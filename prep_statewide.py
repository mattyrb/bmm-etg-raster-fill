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
NWI_SHP = _here / "NWI_Investigations_EPSG_32611.shp"

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
    target_crs=None,
    resampling=Resampling.bilinear,
    label: str = "",
):
    """
    Clip a raster to a geometry and optionally reproject the clipped result
    into ``target_crs``.  The clip is done in the raster's native CRS to
    minimise resampling artefacts; the final output is then warped into the
    target CRS (by default the NWI shapefile CRS).  Writes a GeoTIFF with
    DEFLATE compression.
    """
    _log(f"  Clipping {label or src_path.name} …")

    with rasterio.open(src_path) as src:
        # Reproject clip geometry to raster CRS for the mask step
        geom_series = gpd.GeoSeries([clip_geom], crs=clip_crs)
        if not geom_series.crs.equals(src.crs):
            geom_series_src = geom_series.to_crs(src.crs)
        else:
            geom_series_src = geom_series
        geom = [geom_series_src.iloc[0].__geo_interface__]

        clipped_image, clipped_transform = rio_mask(
            src, geom, crop=True, all_touched=True, nodata=src.nodata,
        )
        src_crs = src.crs
        src_nodata = src.nodata
        src_dtype = src.dtypes[0]
        src_count = src.count

    # If no target CRS or it matches the source, write the clipped raster as-is
    need_reproject = True
    if target_crs is None:
        need_reproject = False
    else:
        dst_crs = rasterio.crs.CRS.from_user_input(target_crs)
        if dst_crs.equals(src_crs):
            need_reproject = False

    if not need_reproject:
        out_profile = {
            "driver": "GTiff",
            "dtype": src_dtype,
            "count": src_count,
            "crs": src_crs,
            "transform": clipped_transform,
            "width": clipped_image.shape[2],
            "height": clipped_image.shape[1],
            "nodata": src_nodata,
            "compress": "DEFLATE",
            "predictor": 2,
        }
        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(clipped_image)
    else:
        _log(f"    reprojecting {label or src_path.name} "
             f"from {src_crs.to_string()} → {dst_crs.to_string()}")

        src_height = clipped_image.shape[1]
        src_width = clipped_image.shape[2]
        left, bottom, right, top = rasterio.transform.array_bounds(
            src_height, src_width, clipped_transform
        )

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src_width, src_height,
            left=left, bottom=bottom, right=right, top=top,
        )

        out_profile = {
            "driver": "GTiff",
            "dtype": src_dtype,
            "count": src_count,
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "nodata": src_nodata,
            "compress": "DEFLATE",
            "predictor": 2,
        }

        with rasterio.open(dst_path, "w", **out_profile) as dst:
            for b in range(1, src_count + 1):
                reproject(
                    source=clipped_image[b - 1],
                    destination=rasterio.band(dst, b),
                    src_transform=clipped_transform,
                    src_crs=src_crs,
                    src_nodata=src_nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=src_nodata,
                    resampling=resampling,
                )

    size_mb = dst_path.stat().st_size / 1e6
    _log(f"    → {dst_path.name}  ({size_mb:.1f} MB)")


def _download_3dep(clip_geom, clip_crs, dst_path: Path, target_crs=None):
    """
    Download USGS 3DEP 30-m DEM for the clip geometry using py3dep.
    Falls back to instructions if py3dep is not available.  If ``target_crs``
    is provided, the DEM is reprojected from EPSG:4326 into that CRS before
    being written.
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

    # py3dep.get_dem returns an xarray DataArray in EPSG:4326
    dem = py3dep.get_dem(tuple(bounds), resolution=30, crs="EPSG:4326")

    # Build a rasterio-friendly source array + transform in EPSG:4326
    src_transform = rasterio.transform.from_bounds(
        *bounds, dem.shape[1], dem.shape[0]
    )
    src_crs = rasterio.crs.CRS.from_epsg(4326)
    src_arr = dem.values.astype(np.float32)

    # If we don't need to reproject, write directly
    if target_crs is None:
        dst_crs = src_crs
        need_reproject = False
    else:
        dst_crs = rasterio.crs.CRS.from_user_input(target_crs)
        need_reproject = not dst_crs.equals(src_crs)

    if not need_reproject:
        _log(f"    Writing DEM to {dst_path.name} …")
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": src_arr.shape[1],
            "height": src_arr.shape[0],
            "count": 1,
            "crs": dst_crs,
            "transform": src_transform,
            "nodata": np.nan,
            "compress": "DEFLATE",
            "predictor": 2,
        }
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(src_arr, 1)
    else:
        _log(f"    Reprojecting DEM {src_crs.to_string()} → {dst_crs.to_string()}")
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs,
            src_arr.shape[1], src_arr.shape[0],
            left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3],
        )
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": dst_width,
            "height": dst_height,
            "count": 1,
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": np.nan,
            "compress": "DEFLATE",
            "predictor": 2,
        }
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=src_arr,
                destination=rasterio.band(dst, 1),
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

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
    _log(f"  NWI shapefile CRS: {gdf.crs.to_string() if gdf.crs else 'UNKNOWN'}")

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

    # All statewide outputs will be reprojected into the NWI shapefile CRS
    # so downstream per-basin prep can use a single consistent grid.
    target_crs = gdf.crs
    _log(f"  Output CRS for statewide rasters: {target_crs.to_string()}")

    # ── Clip BpS ────────────────────────────────────────────────────────────
    _log("Clipping BpS …")
    if not args.bps.exists():
        sys.exit(f"ERROR: BpS raster not found: {args.bps}")
    _clip_raster_to_geometry(
        args.bps, STATEWIDE_DIR / "BpS_statewide.tif",
        clip_geom, clip_crs,
        target_crs=target_crs,
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
        target_crs=target_crs,
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
            target_crs=target_crs,
            resampling=Resampling.bilinear,
            label="DEM",
        )
    else:
        _download_3dep(clip_geom, clip_crs, dem_dst, target_crs=target_crs)

    # ── Done ────────────────────────────────────────────────────────────────
    _log(f"\nStatewide rasters written to: {STATEWIDE_DIR.resolve()}")
    _log(f"  DEM_statewide.tif")
    _log(f"  BpS_statewide.tif")
    _log(f"  WTD_statewide.tif")
    _log(f"Elapsed: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
