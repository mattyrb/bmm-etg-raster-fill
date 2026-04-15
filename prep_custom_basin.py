#!/usr/bin/env python3
"""
prep_custom_basin.py
====================
Set up a basin directory for a study area that is NOT part of the Nevada
NWI investigation basins.  This is the entry point for applying the ETg
baseline-fill workflow to any geographic area (e.g. Sierra Valley CA,
basins in other states, or international sites).

Unlike prep_basin.py, which clips covariates from pre-built statewide
subsets and reads basin boundaries from the NWI shapefile, this script:

    1. Accepts a user-provided boundary shapefile (or GeoJSON / GPKG).
    2. Clips DEM and BpS from CONUS-scale (or any extent) source rasters.
    3. Optionally clips WTD if a source raster is provided.
    4. Derives HAND from the clipped DEM (same whitebox pipeline as NWI).
    5. Generates a config.toml identical to what prep_basin.py produces.
    6. Copies the boundary shapefile into input/ so etg_baseline_fill.py
       can use it as the training mask (via the ``boundary_shp`` config
       field), removing the dependency on NWI_Investigations.

The resulting basin directory is indistinguishable from an NWI basin
directory — etg_baseline_fill.py, diagnostics.py, and etunit_summary.py
all work exactly the same way.

Usage
-----
    # Minimal: boundary + DEM + BpS + ETg + treatment
    python prep_custom_basin.py SierraValley \\
        --boundary  /path/to/sierra_valley_boundary.shp \\
        --dem       /path/to/CONUS_DEM.tif \\
        --bps       /path/to/LF2020_BPS_CONUS.tif

    # With WTD:
    python prep_custom_basin.py SierraValley \\
        --boundary  /path/to/sierra_valley_boundary.shp \\
        --dem       /path/to/CONUS_DEM.tif \\
        --bps       /path/to/LF2020_BPS_CONUS.tif \\
        --wtd       /path/to/wtd_conus.tif

    # Skip HAND derivation:
    python prep_custom_basin.py SierraValley \\
        --boundary  /path/to/sierra_valley_boundary.shp \\
        --dem       /path/to/local_dem.tif \\
        --bps       /path/to/local_bps.tif \\
        --skip-hand

Outputs
-------
    basins/<basin_key>/
        input/
            DEM.tif
            BpS.tif
            BpS.clr          (QGIS symbology, if bps_lookup.json exists)
            BpS.qml           "
            WTD.tif           (if --wtd provided)
            HAND.tif          (derived from DEM.tif)
            boundary.shp      (copy of your boundary)
        config.toml

After prep, place your ETg raster and treatment shapefile in input/,
review config.toml, and run:

    python etg_baseline_fill.py <basin_key>

License: MIT
"""

import argparse
import shutil
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
    sys.exit(f"Missing dependency: {e}\n"
             "Install: pip install rasterio geopandas shapely")

from prep_statewide import (
    _derive_hand_from_dem,
    _derive_rem_from_dem,
    HAND_STREAM_THRESHOLD_CELLS,
    HAND_BREACH_DIST_CELLS,
    REM_WINDOW_PX,
    REM_PERCENTILE_Q,
)

_here = Path(__file__).resolve().parent
PROJECT_DIR = _here
BASINS_DIR = PROJECT_DIR / "basins"

# Buffer around boundary when clipping covariates (meters).
CLIP_BUFFER_M = 5_000  # 5 km


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _crs_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return rasterio.crs.CRS.from_user_input(a) == \
               rasterio.crs.CRS.from_user_input(b)
    except Exception:
        return str(a) == str(b)


def _clip_raster(
    src_path: Path,
    dst_path: Path,
    clip_geom,
    clip_crs,
    target_crs,
    resampling=Resampling.bilinear,
    label: str = "",
    force: bool = False,
):
    """
    Clip a raster to a geometry and reproject into target_crs.

    If ``dst_path`` already exists, return immediately unless ``force`` is
    True.  This lets users pre-populate input/ with data staged from
    another machine without re-running an expensive clip.
    """
    if dst_path.exists() and not force:
        size_mb = dst_path.stat().st_size / 1e6
        _log(f"  {label or src_path.name}: {dst_path.name} already present "
             f"({size_mb:.1f} MB) — skipping (use --force to rebuild)")
        return
    _log(f"  Clipping {label or src_path.name} …")

    with rasterio.open(src_path) as src:
        geom_series = gpd.GeoSeries([clip_geom], crs=clip_crs)
        if not _crs_equal(geom_series.crs, src.crs):
            geom_series = geom_series.to_crs(src.crs)
        geom = [geom_series.iloc[0].__geo_interface__]

        clipped_image, clipped_transform = rio_mask(
            src, geom, crop=True, all_touched=True, nodata=src.nodata,
        )
        src_crs = src.crs
        src_nodata = src.nodata
        src_dtype = src.dtypes[0]
        src_count = src.count

    need_reproject = (target_crs is not None and
                      not _crs_equal(target_crs, src_crs))

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
        dst_crs = rasterio.crs.CRS.from_user_input(target_crs)
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


def _copy_boundary(boundary_path: Path, dest_dir: Path) -> Path:
    """
    Copy boundary shapefile (and its sidecar files) into ``dest_dir``
    (which should be the basin's source/ directory).
    Returns the destination path of the .shp file.
    """
    stem = boundary_path.stem
    src_dir = boundary_path.parent
    dst_stem = "boundary"

    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx",
                ".geojson", ".gpkg"):
        src_file = src_dir / f"{stem}{ext}"
        if src_file.exists():
            dst_file = dest_dir / f"{dst_stem}{ext}"
            shutil.copy2(src_file, dst_file)

    # Handle GeoJSON / GPKG that don't have sidecars
    if boundary_path.suffix.lower() in (".geojson", ".gpkg"):
        dst = dest_dir / f"{dst_stem}{boundary_path.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(boundary_path, dst)
        return dst

    return dest_dir / f"{dst_stem}.shp"


def _generate_config(basin_dir: Path, basin_key: str,
                     boundary_name: str, has_wtd: bool):
    """Write a default config.toml for a custom basin."""
    config_path = basin_dir / "config.toml"
    if config_path.exists():
        _log(f"  config.toml already exists — preserving your edits")
        return

    # Scan source/ (preferred) then input/ (legacy) for raws
    source_dir = basin_dir / "source"
    input_dir = basin_dir / "input"
    search_dirs = [source_dir, input_dir] if source_dir.exists() else [input_dir]
    etg_candidates, shp_candidates = [], []
    for d in search_dirs:
        etg_candidates += sorted(d.glob("*etg*median*.tif")) + \
                          sorted(d.glob("*ETg*.tif"))
        shp_candidates += sorted(d.glob("*.shp"))
    treatment_shps = [s for s in shp_candidates
                      if s.stem != "boundary" and
                      any(kw in s.stem.lower()
                          for kw in ("etunit", "et_unit", "phreats",
                                     "treatment", "w_ag", "master"))]
    if not treatment_shps:
        treatment_shps = [s for s in shp_candidates if s.stem != "boundary"]

    etg_tif = etg_candidates[0].name if etg_candidates else "# PLACE ETg RASTER HERE"
    treat_shp = treatment_shps[0].name if treatment_shps else "# PLACE TREATMENT SHP HERE"

    # Render the config.toml from basins/_template/config.toml.  To change
    # defaults for new custom basins, edit the template file — not this script.
    import basin_config as _bc
    toml_content = _bc.render_config_template({
        "basin_key":         basin_key,
        "basin_id":          "",
        "basin_name":        basin_key,
        "etg_tif":           etg_tif,
        "treatment_shp":     treat_shp,
        # Custom basins always have an explicit boundary from --boundary.
        "boundary_shp_line": f'boundary_shp = "{boundary_name}"',
        "wtd_tif":           "WTD.tif" if has_wtd else "",
        "use_wtd":           "true" if has_wtd else "false",
    })

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(toml_content)
    _log(f"  → config.toml generated from basins/_template/config.toml "
         f"(review & edit before running fill)")


def prep_custom_basin(
    basin_key: str,
    boundary_path: Path,
    dem_path: Path,
    bps_path: Path,
    wtd_path: Path | None = None,
    awc_path: Path | None = None,
    soil_depth_path: Path | None = None,
    *,
    buffer_m: float = CLIP_BUFFER_M,
    derive_hand: bool = True,
    hand_threshold: int = HAND_STREAM_THRESHOLD_CELLS,
    keep_hand_intermediates: bool = False,
    derive_rem: bool = False,
    rem_window_px: int = REM_WINDOW_PX,
    rem_percentile_q: float = REM_PERCENTILE_Q,
    force: dict | None = None,
):
    """
    Set up a basin directory from user-provided inputs.

    Parameters
    ----------
    basin_key : str
        Directory name under basins/ (e.g. "SierraValley").
    boundary_path : Path
        Shapefile / GeoJSON / GPKG defining the study-area boundary.
    dem_path : Path
        DEM raster (any extent >= study area; will be clipped).
    bps_path : Path
        BpS raster (any extent >= study area; will be clipped).
    wtd_path : Path or None
        WTD raster (optional).
    buffer_m : float
        Buffer around boundary for covariate clipping (meters).
    derive_hand : bool
        Whether to derive HAND from the clipped DEM.
    hand_threshold : int
        Stream initiation threshold in cells for HAND.
    keep_hand_intermediates : bool
        Keep whitebox intermediate files for inspection.
    """
    force = force or {}

    def _force(key: str) -> bool:
        return bool(force.get("all") or force.get(key))

    _log(f"\n{'='*60}")
    _log(f"Preparing custom basin: {basin_key}")

    # ── Read boundary ──────────────────────────────────────────────────
    if not boundary_path.exists():
        sys.exit(f"ERROR: boundary file not found: {boundary_path}")
    gdf_bnd = gpd.read_file(boundary_path)
    if len(gdf_bnd) == 0:
        sys.exit(f"ERROR: boundary file is empty: {boundary_path}")

    bnd_crs = gdf_bnd.crs
    if bnd_crs is None:
        sys.exit(f"ERROR: boundary file has no CRS: {boundary_path}")

    _log(f"  Boundary CRS: {bnd_crs.to_string()}")
    _log(f"  {len(gdf_bnd)} feature(s) in boundary")

    # Determine target CRS: use the boundary CRS if it's projected,
    # otherwise auto-pick a UTM zone from the centroid.
    if bnd_crs.is_projected:
        target_crs = bnd_crs
    else:
        centroid = unary_union(gdf_bnd.geometry).centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = "north" if centroid.y >= 0 else "south"
        epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
        target_crs = rasterio.crs.CRS.from_epsg(epsg)
        _log(f"  Boundary is geographic — auto-selected UTM: EPSG:{epsg}")

    _log(f"  Target CRS for basin: {target_crs.to_string()}")

    # Dissolve and buffer in a projected CRS
    dissolved = unary_union(gdf_bnd.geometry)
    if bnd_crs.is_geographic:
        geom_series = gpd.GeoSeries([dissolved], crs=bnd_crs).to_crs(target_crs)
        dissolved_proj = geom_series.iloc[0]
        clip_crs = target_crs
    else:
        dissolved_proj = dissolved
        clip_crs = bnd_crs

    clip_geom = dissolved_proj.buffer(buffer_m)
    _log(f"  Clip buffer: {buffer_m:.0f} m")

    # ── Create directory structure ─────────────────────────────────────
    #   source/ -- user-supplied raws (ETg, treatment shp, boundary shp)
    #   input/  -- prep-generated clipped covariates
    #   output/ -- fill outputs
    basin_dir = BASINS_DIR / basin_key
    source_dir = basin_dir / "source"
    input_dir = basin_dir / "input"
    output_dir = basin_dir / "output"
    source_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Copy boundary into source/ ─────────────────────────────────────
    _log("  Copying boundary into source/ …")
    bnd_dst = _copy_boundary(boundary_path, source_dir)
    boundary_name = bnd_dst.name
    _log(f"    → {boundary_name}")

    # ── Clip DEM (or download from OpenTopography if no DEM supplied) ──
    dem_dst = input_dir / "DEM.tif"
    if dem_dst.exists() and not _force("dem"):
        size_mb = dem_dst.stat().st_size / 1e6
        _log(f"    → {dem_dst.name} already present ({size_mb:.1f} MB) "
             f"— skipping (pass --force-dem to rebuild)")
    elif dem_path is not None:
        if not dem_path.exists():
            sys.exit(f"ERROR: DEM raster not found: {dem_path}")
        _clip_raster(dem_path, dem_dst,
                     clip_geom, clip_crs, target_crs,
                     Resampling.bilinear, "DEM",
                     force=_force("dem"))
    else:
        _log("  No --dem supplied — downloading COP30 from OpenTopography …")
        import opentopo
        opentopo.download_dem(
            clip_geom, clip_crs, dem_dst,
            target_crs=target_crs,
            demtype="COP30",
        )

    # ── Clip BpS ───────────────────────────────────────────────────────
    if not bps_path.exists():
        sys.exit(f"ERROR: BpS raster not found: {bps_path}")
    bps_dst = input_dir / "BpS.tif"
    _clip_raster(bps_path, bps_dst,
                 clip_geom, clip_crs, target_crs,
                 Resampling.nearest, "BpS",
                 force=_force("bps"))

    # Write QGIS symbology if lookup is available
    def _apply_bps_symbology(lut):
        """Embed color table + RAT in the GeoTIFF and write sidecars."""
        from bps_utils import write_bps_symbology, embed_bps_colortable_and_rat
        embedded = embed_bps_colortable_and_rat(bps_dst, lut)
        write_bps_symbology(bps_dst, lut)
        if embedded:
            _log("    → BpS.tif color table + RAT embedded; "
                 "BpS.clr + BpS.qml sidecars written")
        else:
            _log("    → BpS.clr + BpS.qml sidecars written "
                 "(GDAL embed skipped)")

    try:
        from bps_utils import load_bps_lookup
        bps_lut = load_bps_lookup()
        if bps_lut and bps_dst.exists():
            _apply_bps_symbology(bps_lut)
    except Exception as e:
        _log(f"    (BpS symbology skipped: {e})")

    # If no cached lookup exists, try extracting from the source
    try:
        from bps_utils import extract_bps_lookup, load_bps_lookup as _reload
        if not _reload():
            _log("  Extracting BpS class lookup from source …")
            extract_bps_lookup(bps_path)
            bps_lut = _reload()
            if bps_lut and bps_dst.exists():
                _apply_bps_symbology(bps_lut)
    except Exception:
        pass

    # ── Clip WTD (optional) ────────────────────────────────────────────
    has_wtd = False
    if wtd_path is not None:
        if not wtd_path.exists():
            _log(f"  WARNING: WTD raster not found: {wtd_path} — skipping")
        else:
            _clip_raster(wtd_path, input_dir / "WTD.tif",
                         clip_geom, clip_crs, target_crs,
                         Resampling.bilinear, "WTD",
                         force=_force("wtd"))
            has_wtd = True

    # ── Clip AWC + SoilDepth (optional) ────────────────────────────────
    has_awc = False
    if awc_path is not None:
        if not awc_path.exists():
            _log(f"  WARNING: AWC raster not found: {awc_path} — skipping")
        else:
            _clip_raster(awc_path, input_dir / "AWC.tif",
                         clip_geom, clip_crs, target_crs,
                         Resampling.bilinear, "AWC",
                         force=_force("soil"))
            has_awc = True
    has_soil_depth = False
    if soil_depth_path is not None:
        if not soil_depth_path.exists():
            _log(f"  WARNING: SoilDepth raster not found: {soil_depth_path} "
                 f"— skipping")
        else:
            _clip_raster(soil_depth_path, input_dir / "SoilDepth.tif",
                         clip_geom, clip_crs, target_crs,
                         Resampling.bilinear, "SoilDepth",
                         force=_force("soil"))
            has_soil_depth = True

    # ── Derive HAND ────────────────────────────────────────────────────
    if derive_hand:
        hand_dst = input_dir / "HAND.tif"
        dem_src = input_dir / "DEM.tif"
        if hand_dst.exists() and not _force("hand"):
            size_mb = hand_dst.stat().st_size / 1e6
            _log(f"  HAND.tif already present ({size_mb:.1f} MB) — "
                 "skipping derivation (use --force or --force-hand to rebuild)")
        elif not dem_src.exists():
            _log("  WARNING: DEM.tif missing — cannot derive HAND")
        else:
            try:
                _derive_hand_from_dem(
                    dem_path=dem_src,
                    dst_path=hand_dst,
                    threshold_cells=hand_threshold,
                    keep_intermediates=keep_hand_intermediates,
                )
            except SystemExit as exc:
                _log(f"  WARNING: HAND derivation failed: {exc}")
                if hand_dst.exists():
                    try:
                        hand_dst.unlink()
                    except OSError:
                        pass
    else:
        _log("  HAND derivation skipped (--skip-hand)")

    # ── Derive REM (optional, opt-in) ──────────────────────────────────
    if derive_rem:
        rem_dst = input_dir / "REM.tif"
        dem_src = input_dir / "DEM.tif"
        if rem_dst.exists() and not _force("rem"):
            size_mb = rem_dst.stat().st_size / 1e6
            _log(f"  REM.tif already present ({size_mb:.1f} MB) — "
                 "skipping derivation (use --force or --force-rem to rebuild)")
        elif not dem_src.exists():
            _log("  WARNING: DEM.tif missing — cannot derive REM")
        else:
            try:
                _derive_rem_from_dem(
                    dem_path=dem_src,
                    dst_path=rem_dst,
                    window_px=rem_window_px,
                    percentile_q=rem_percentile_q,
                )
            except SystemExit as exc:
                _log(f"  WARNING: REM derivation failed: {exc}")
                if rem_dst.exists():
                    try:
                        rem_dst.unlink()
                    except OSError:
                        pass

    # ── Generate config.toml ───────────────────────────────────────────
    _generate_config(basin_dir, basin_key, boundary_name, has_wtd)

    _log(f"\n  Done.  Place raw ETg raster and treatment shapefile in:\n"
         f"    {source_dir.resolve()}/\n"
         f"  (Prep-generated covariates are in {input_dir.resolve()}/)\n"
         f"  Then run:\n"
         f"    python etg_baseline_fill.py {basin_key}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Set up a basin directory for a custom (non-NWI) study area."
    )
    parser.add_argument("basin_key",
                        help="Name for this basin (used as directory name, "
                             "e.g. 'SierraValley')")
    parser.add_argument("--boundary", type=Path, required=True,
                        help="Shapefile / GeoJSON / GPKG defining the "
                             "study-area boundary polygon")
    parser.add_argument("--dem", type=Path, default=None,
                        help="DEM raster (any extent >= study area).  "
                             "If omitted, COP30 is downloaded for the "
                             "basin bbox from OpenTopography.")
    parser.add_argument("--bps", type=Path, required=True,
                        help="BpS raster (any extent >= study area)")
    parser.add_argument("--wtd", type=Path, default=None,
                        help="WTD raster (optional)")
    parser.add_argument("--awc", type=Path, default=None,
                        help="gSSURGO AWC raster (optional)")
    parser.add_argument("--soil-depth", type=Path, default=None,
                        help="gSSURGO depth-to-restrictive-layer raster (optional)")
    parser.add_argument("--buffer-m", type=float, default=CLIP_BUFFER_M,
                        help=f"Buffer around boundary for clipping "
                             f"(default: {CLIP_BUFFER_M} m)")
    parser.add_argument("--skip-hand", action="store_true",
                        help="Skip HAND derivation")
    parser.add_argument("--hand-threshold", type=int,
                        default=HAND_STREAM_THRESHOLD_CELLS,
                        help=f"Stream threshold in cells for HAND "
                             f"(default: {HAND_STREAM_THRESHOLD_CELLS})")
    parser.add_argument("--keep-hand-intermediates", action="store_true",
                        help="Keep whitebox intermediate files")
    parser.add_argument("--derive-rem", action="store_true",
                        help="Derive a Relative Elevation Model (REM) from "
                             "the DEM via a percentile filter (opt-in).")
    parser.add_argument("--rem-window-px", type=int, default=REM_WINDOW_PX,
                        help=f"REM percentile-filter window side in pixels "
                             f"(default: {REM_WINDOW_PX}, ≈2 km at 30 m)")
    parser.add_argument("--rem-percentile-q", type=float, default=REM_PERCENTILE_Q,
                        help=f"REM percentile (0-100) used as valley-floor proxy "
                             f"(default: {REM_PERCENTILE_Q})")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild every covariate even if already present")
    parser.add_argument("--force-dem",  action="store_true", help="Rebuild DEM.tif")
    parser.add_argument("--force-bps",  action="store_true", help="Rebuild BpS.tif")
    parser.add_argument("--force-wtd",  action="store_true", help="Rebuild WTD.tif")
    parser.add_argument("--force-hand", action="store_true", help="Rebuild HAND.tif")
    parser.add_argument("--force-soil", action="store_true",
                        help="Rebuild AWC.tif / SoilDepth.tif")
    parser.add_argument("--force-rem",  action="store_true", help="Rebuild REM.tif")
    args = parser.parse_args()

    force_dict = {
        "all":  bool(args.force),
        "dem":  bool(args.force_dem),
        "bps":  bool(args.force_bps),
        "wtd":  bool(args.force_wtd),
        "hand": bool(args.force_hand),
        "soil": bool(args.force_soil),
        "rem":  bool(args.force_rem),
    }

    t0 = time.time()
    BASINS_DIR.mkdir(parents=True, exist_ok=True)

    prep_custom_basin(
        basin_key=args.basin_key,
        boundary_path=args.boundary,
        dem_path=args.dem,
        bps_path=args.bps,
        wtd_path=args.wtd,
        awc_path=args.awc,
        soil_depth_path=args.soil_depth,
        buffer_m=args.buffer_m,
        derive_hand=not args.skip_hand,
        hand_threshold=args.hand_threshold,
        keep_hand_intermediates=args.keep_hand_intermediates,
        derive_rem=args.derive_rem,
        rem_window_px=args.rem_window_px,
        rem_percentile_q=args.rem_percentile_q,
        force=force_dict,
    )

    _log(f"\nElapsed: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
