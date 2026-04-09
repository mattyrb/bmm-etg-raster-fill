#!/usr/bin/env python3
"""
prep_statewide.py
=================
One-time script: clip CONUS-wide DEM, BpS, and WTD rasters to the dissolved
NWI investigation boundary (with buffer) so per-basin prep can use fast
windowed reads instead of touching the full CONUS files every time.

Also downloads the 3DEP 30-m DEM via py3dep if no CONUS DEM is provided.

Usage
-----
    python prep_statewide.py \
        --bps  /path/to/LF2020_BPS_CONUS.tif \
        --wtd  /path/to/wtd_conus.tif \
        [--dem /path/to/conus_dem.tif]   # optional; downloads 3DEP if omitted

Outputs are written to  <project>/statewide/

License: MIT
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import geopandas as gpd
    from shapely.ops import unary_union
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nInstall: pip install rasterio geopandas shapely")


_here = Path(__file__).resolve().parent
PROJECT_DIR = _here
STATEWIDE_DIR = PROJECT_DIR / "statewide"
NWI_SHP = _here / "NWI_Investigations.shp"

# Buffer (metres) around the dissolved NWI boundary when clipping.
# Ensures edge basins have covariate data right up to and slightly beyond
# their boundary so reprojection doesn't produce NaN strips.
CLIP_BUFFER_M = 10_000  # 10 km


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _clip_raster_to_geometry(
    src_path: Path,
    dst_path: Path,
    clip_geom,
    clip_crs,
    resampling=Resampling.bilinear,
    label: str = "",
):
    """
    Clip a raster to a geometry, reprojecting the geometry to match the raster
    CRS if needed.  Writes a new GeoTIFF with DEFLATE compression.
    """
    _log(f"  Clipping {label or src_path.name} …")

    with rasterio.open(src_path) as src:
        # Reproject clip geometry to raster CRS if needed
        geom_series = gpd.GeoSeries([clip_geom], crs=clip_crs)
        if not geom_series.crs.equals(src.crs):
            geom_series = geom_series.to_crs(src.crs)
        geom = [geom_series.iloc[0].__geo_interface__]

        out_image, out_transform = rio_mask(
            src, geom, crop=True, all_touched=True, nodata=src.nodata,
        )
        out_profile = src.profile.copy()
        out_profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            compress="DEFLATE",
            predictor=2,
        )

    with rasterio.open(dst_path, "w", **out_profile) as dst:
        dst.write(out_image)

    size_mb = dst_path.stat().st_size / 1e6
    _log(f"    → {dst_path.name}  ({size_mb:.1f} MB)")


def _download_3dep(clip_geom, clip_crs, dst_path: Path):
    """
    Download USGS 3DEP 30-m DEM for the clip geometry using py3dep.
    Falls back to instructions if py3dep is not available.
    """
    _log("  Downloading 3DEP 30-m DEM via py3dep …")
    try:
        import py3dep
    except ImportError:
        sys.exit(
            "py3dep is not installed.  Install with:\n"
            "  pip install py3dep\n"
            "Or provide a CONUS DEM with --dem /path/to/dem.tif"
        )

    # py3dep needs a bounding box in EPSG:4326
    geom_series = gpd.GeoSeries([clip_geom], crs=clip_crs)
    geom_4326 = geom_series.to_crs("EPSG:4326")
    bounds = geom_4326.total_bounds  # (minx, miny, maxx, maxy)
    _log(f"    bbox (EPSG:4326): {bounds}")

    # py3dep.get_dem returns an xarray DataArray
    dem = py3dep.get_dem(tuple(bounds), resolution=30, crs="EPSG:4326")

    # Write to GeoTIFF
    _log(f"    Writing DEM to {dst_path.name} …")
    transform = rasterio.transform.from_bounds(
        *bounds, dem.shape[1], dem.shape[0]
    )
    # Actually use the transform from the xarray attrs if available
    if hasattr(dem, "rio"):
        dem.rio.to_raster(dst_path, compress="DEFLATE")
    else:
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": dem.shape[1],
            "height": dem.shape[0],
            "count": 1,
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": np.nan,
            "compress": "DEFLATE",
        }
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(dem.values.astype(np.float32), 1)

    size_mb = dst_path.stat().st_size / 1e6
    _log(f"    → {dst_path.name}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Clip CONUS rasters to NWI investigation extent."
    )
    parser.add_argument("--bps", type=Path, required=True,
                        help="Path to CONUS BpS raster (LF2020)")
    parser.add_argument("--wtd", type=Path, required=True,
                        help="Path to CONUS WTD raster (Ma et al. 2025)")
    parser.add_argument("--dem", type=Path, default=None,
                        help="Path to CONUS DEM raster (optional; downloads "
                             "3DEP 30m if omitted)")
    parser.add_argument("--buffer-m", type=float, default=CLIP_BUFFER_M,
                        help=f"Buffer around NWI boundary (default: {CLIP_BUFFER_M} m)")
    args = parser.parse_args()

    t0 = time.time()
    STATEWIDE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and dissolve NWI boundary ──────────────────────────────────────
    _log("Loading NWI investigation boundaries …")
    if not NWI_SHP.exists():
        sys.exit(f"ERROR: NWI shapefile not found: {NWI_SHP}")

    gdf = gpd.read_file(NWI_SHP)
    _log(f"  {len(gdf)} basin polygons loaded")

    # Dissolve into a single geometry and buffer
    dissolved = unary_union(gdf.geometry)
    # Buffer in the shapefile's native CRS (should be projected, metres)
    if gdf.crs is not None and gdf.crs.is_geographic:
        _log("  WARNING: NWI shapefile is in geographic CRS — "
             "reprojecting to EPSG:5070 for buffering")
        gdf_proj = gdf.to_crs("EPSG:5070")
        dissolved = unary_union(gdf_proj.geometry)
        clip_crs = gdf_proj.crs
    else:
        clip_crs = gdf.crs

    clip_geom = dissolved.buffer(args.buffer_m)
    _log(f"  Dissolved + {args.buffer_m:.0f} m buffer ready")

    # ── Clip BpS ────────────────────────────────────────────────────────────
    _log("Clipping BpS …")
    if not args.bps.exists():
        sys.exit(f"ERROR: BpS raster not found: {args.bps}")
    _clip_raster_to_geometry(
        args.bps, STATEWIDE_DIR / "BpS_statewide.tif",
        clip_geom, clip_crs,
        resampling=Resampling.nearest,
        label="BpS",
    )

    # ── Clip WTD ────────────────────────────────────────────────────────────
    _log("Clipping WTD …")
    if not args.wtd.exists():
        sys.exit(f"ERROR: WTD raster not found: {args.wtd}")
    _clip_raster_to_geometry(
        args.wtd, STATEWIDE_DIR / "WTD_statewide.tif",
        clip_geom, clip_crs,
        resampling=Resampling.bilinear,
        label="WTD",
    )

    # ── DEM: clip from CONUS file or download 3DEP ──────────────────────────
    dem_dst = STATEWIDE_DIR / "DEM_statewide.tif"
    if args.dem is not None:
        _log("Clipping DEM …")
        if not args.dem.exists():
            sys.exit(f"ERROR: DEM raster not found: {args.dem}")
        _clip_raster_to_geometry(
            args.dem, dem_dst,
            clip_geom, clip_crs,
            resampling=Resampling.bilinear,
            label="DEM",
        )
    else:
        _download_3dep(clip_geom, clip_crs, dem_dst)

    # ── Done ────────────────────────────────────────────────────────────────
    _log(f"\nStatewide rasters written to: {STATEWIDE_DIR.resolve()}")
    _log(f"  DEM_statewide.tif")
    _log(f"  BpS_statewide.tif")
    _log(f"  WTD_statewide.tif")
    _log(f"Elapsed: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
