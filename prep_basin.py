#!/usr/bin/env python3
"""
prep_basin.py
=============
Per-basin setup: clips DEM, BpS, and WTD from statewide subsets to a single
basin boundary, derives HAND (Height Above Nearest Drainage) from the
clipped basin DEM, and generates a default ``config.toml`` if one doesn't
exist.

HAND is derived per-basin rather than statewide because whitebox-tools'
ElevationAboveStream runs out of memory on the full Nevada DEM.  Each
basin DEM (clipped with a 5 km buffer) easily fits in RAM.

Reads the basin polygon from NWI_Investigations_EPSG_32611.shp using the
``Basin`` field as the key (e.g. "101_SierraValley").

Usage
-----
    # Prep a single basin:
    python prep_basin.py 101_SierraValley

    # Prep all basins that have ETg + treatment data placed in their dirs:
    python prep_basin.py --all

    # List available basin keys from the NWI shapefile:
    python prep_basin.py --list

    # Skip HAND derivation entirely:
    python prep_basin.py 101_SierraValley --skip-hand

    # Tune HAND stream threshold (larger = sparser stream network):
    python prep_basin.py 101_SierraValley --hand-threshold 500

Outputs
-------
    basins/<basin_key>/
        input/
            DEM.tif
            BpS.tif
            WTD.tif
            HAND.tif     (derived from DEM.tif via whitebox-tools)
        config.toml   (created only if missing — never overwritten)

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
    from rasterio.enums import Resampling
    import geopandas as gpd
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n"
             "Install: pip install rasterio geopandas shapely")

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 fallback

# Reuse the HAND derivation implementation from prep_statewide so we have
# a single source of truth for the whitebox pipeline.  Running HAND
# per-basin is the only workable option — whitebox's ElevationAboveStream
# tries to allocate ~25 GB on the full statewide DEM and runs out of
# memory, but each basin DEM easily fits in RAM.
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
STATEWIDE_DIR = PROJECT_DIR / "statewide"
BASINS_DIR = PROJECT_DIR / "basins"
NWI_SHP = _here / "NWI_Investigations_EPSG_32611.shp"

# Fields in NWI_Investigations_EPSG_32611.shp
BASIN_KEY_FIELD = "Basin"
BASIN_ID_FIELD  = "BasinID"
BASIN_NAME_FIELD = "BasinName"

# Buffer around basin polygon when clipping covariates (meters).
# Larger than the treatment buffer so there's always covariate data
# available at the edges of the treatment zone.
BASIN_CLIP_BUFFER_M = 5_000  # 5 km


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_nwi() -> gpd.GeoDataFrame:
    """Load NWI shapefile, return GeoDataFrame."""
    if not NWI_SHP.exists():
        sys.exit(f"ERROR: NWI shapefile not found: {NWI_SHP}")
    return gpd.read_file(NWI_SHP)


def _crs_equal(a, b) -> bool:
    """Compare two CRS-like objects (rasterio.crs.CRS lacks ``.equals()``)."""
    if a is None or b is None:
        return a is None and b is None
    try:
        return rasterio.crs.CRS.from_user_input(a) == \
               rasterio.crs.CRS.from_user_input(b)
    except Exception:
        return str(a) == str(b)


def _clip_from_statewide(
    src_name: str,
    dst_path: Path,
    clip_geom,
    clip_crs,
    resampling=Resampling.bilinear,
    force: bool = False,
    quiet_missing: bool = False,
):
    """
    Clip a statewide raster to a basin polygon + buffer.

    If ``dst_path`` already exists and ``force`` is False, the existing
    file is preserved (a log note is emitted).  Pass ``force=True`` to
    always re-clip.

    If the statewide source is missing, logs a warning (or stays silent
    if ``quiet_missing=True``, useful for optional covariates like soil)
    and returns False.
    """
    if dst_path.exists() and not force:
        size_mb = dst_path.stat().st_size / 1e6
        _log(f"    → {dst_path.name} already present ({size_mb:.1f} MB) "
             f"— skipping (pass --force or --force-{dst_path.stem.lower()} "
             f"to rebuild)")
        return True

    src_path = STATEWIDE_DIR / src_name
    if not src_path.exists():
        if not quiet_missing:
            _log(f"  WARNING: {src_path.name} not found in statewide/ — skipping")
        else:
            _log(f"    ({src_path.name} not in statewide/ — optional, skipping)")
        return False

    with rasterio.open(src_path) as src:
        # Reproject clip geometry to raster CRS
        geom_series = gpd.GeoSeries([clip_geom], crs=clip_crs)
        if not _crs_equal(geom_series.crs, src.crs):
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
    return True


def _generate_default_config(basin_dir: Path, basin_key: str,
                              basin_id: str, basin_name: str):
    """
    Write a default config.toml for a basin.  NEVER overwrites an existing one.
    """
    config_path = basin_dir / "config.toml"
    if config_path.exists():
        _log(f"  config.toml already exists — preserving your edits")
        return

    # Scan source/ (preferred) then input/ (legacy) for likely ETg and
    # treatment files.  New basins put raws in source/ and prep-script
    # covariates in input/; legacy basins had everything in input/.
    source_dir = basin_dir / "source"
    input_dir = basin_dir / "input"
    search_dirs = [source_dir, input_dir] if source_dir.exists() else [input_dir]
    etg_candidates, shp_candidates = [], []
    for d in search_dirs:
        etg_candidates += sorted(d.glob("*etg*median*.tif")) + \
                          sorted(d.glob("*ETg*.tif"))
        shp_candidates += sorted(d.glob("*.shp"))

    # Filter shp candidates: prefer files that look like treatment/ET-unit shapefiles
    treatment_shps = [s for s in shp_candidates
                      if any(kw in s.stem.lower()
                             for kw in ("etunit", "et_unit", "phreats", "treatment",
                                        "w_ag", "master"))]
    if not treatment_shps:
        treatment_shps = shp_candidates

    etg_tif = etg_candidates[0].name if etg_candidates else "# PLACE ETg RASTER HERE"
    treat_shp = treatment_shps[0].name if treatment_shps else "# PLACE TREATMENT SHP HERE"

    toml_content = f"""\
# ETg Baseline Fill -- Basin Configuration
# Basin: {basin_key}
# Generated automatically.  Edit freely -- this file is never overwritten.

[basin]
basin_key  = "{basin_key}"
basin_id   = "{basin_id}"
basin_name = "{basin_name}"

[source]
# User-supplied files -- filenames are relative to the source/ directory.
# Drop your raw ETg raster(s) and treatment shapefile into source/ and
# reference them here.  prep_basin.py does NOT copy or modify these.
# The ETg raster:
etg_tif       = "{etg_tif}"
# Treatment polygons shapefile:
treatment_shp = "{treat_shp}"
# Optional basin boundary shapefile.  If omitted, the fill script tries
# the statewide NWI shapefile.  Set for custom (non-NWI) study areas.
# boundary_shp = "boundary.shp"

[inputs]
# Prep-generated covariates -- filenames are relative to the input/
# directory.  prep_basin.py writes these by clipping from statewide/.
dem_tif       = "DEM.tif"
bps_tif       = "BpS.tif"
wtd_tif       = "WTD.tif"
hand_tif      = "HAND.tif"
# gSSURGO soil covariates.  prep_basin.py clips these from
# statewide/AWC_statewide.tif and statewide/SoilDepth_statewide.tif if
# present (see prep_statewide.py --awc / --soil-depth).  If the
# statewide sources were never built, the fill script silently skips
# whichever covariate is missing.
awc_tif         = "AWC.tif"
soil_depth_tif  = "SoilDepth.tif"
# Relative Elevation Model (optional).  Opt-in only -- derive it by
# running prep_basin.py with --derive-rem or --force-rem, then flip
# use_rem=true below.  See README for the percentile-filter definition.
rem_tif         = "REM.tif"

[treatment]
# Buffer distance (meters) around treatment polygons.
buffer_m         = 90.0
# Gaussian feather width (pixels) outside the treatment boundary.
feather_width_px = 4
# Shapefile attribute names identifying treatment polygons.
# A polygon is treated if either attribute is > 0.
attr_scale       = "scale_fctr"
attr_replace     = "rplc_rt"

[adjustment]
# Expert adjustment knob -- scale the modeled baseline up or down.
# 1.0 = no change, 0.8 = reduce baseline 20%, 1.2 = increase 20%.
# This is a basin-wide default; individual polygons can override it
# via the "adj_fctr" column in the treatment shapefile.
baseline_adjust    = 1.0
# Name of the per-polygon override column in the treatment shapefile.
# If a polygon has a value > 0 in this column, it overrides the
# basin-wide default above.  Set to "" to disable per-polygon overrides.
attr_adjust        = "adj_fctr"

[model]
# Include water table depth as a covariate? Set to false if the WTD product
# is unreliable for this basin.
use_wtd          = true
# Include Height Above Nearest Drainage as a covariate?  HAND is derived
# from the 3DEP DEM in prep_statewide.py and captures distance-to-drainage
# in elevation units, which is informative for groundwater ET in
# phreatophyte / playa-fringe systems.  Set to false to drop it.
use_hand         = true
# Include gSSURGO soil covariates (Available Water Capacity in the top
# ~1 m + depth to restrictive layer)?  Each is loaded only if its raster
# is present in input/ AND reprojects to at least one valid pixel.  Set
# to false to drop both entirely regardless of file presence.
use_soil         = true
# Include Relative Elevation Model as a covariate?  REM captures height
# above local valley floor via a percentile-filter of the DEM; it is
# OFF by default.  To enable: run prep_basin.py --derive-rem to produce
# input/REM.tif, then set use_rem = true.
use_rem          = false
# When the terrain residual model's CV R² is negative, the fill script
# falls back to spatially weighted BpS class means instead of flat
# basin-wide averages.  This radius (in pixels) controls the Gaussian
# smoothing window.  At 30 m, 33 px ~ 1 km.  Set to 0 for flat means.
spatial_fallback_radius_px = 33
# "lgbm" (LightGBM, recommended) or "rf" (RandomForest)
backend          = "lgbm"
# Exclude training pixels on slopes steeper than this (degrees).
# Set to 0 or remove to disable.
max_slope_deg    = 5.0
# Maximum training pixels (memory/speed cap). Set to 0 for no limit.
max_train_pixels = 500000
random_seed      = 42

[crs_overrides]
# If a covariate raster has a malformed or missing CRS, specify it here.
# Key = raster filename stem (without extension), value = CRS string.
# Example:
# "WTD" = "+proj=lcc +lat_1=30 +lat_2=60 +lon_0=-97.0 +lat_0=40.0 +a=6370000.0 +b=6370000.0 +units=m +no_defs"
"""

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(toml_content)
    _log(f"  → config.toml generated (review & edit before running fill)")


def prep_one_basin(
    basin_key: str,
    gdf_nwi: gpd.GeoDataFrame,
    *,
    derive_hand: bool = True,
    hand_threshold: int = HAND_STREAM_THRESHOLD_CELLS,
    hand_breach_dist: int = HAND_BREACH_DIST_CELLS,
    keep_hand_intermediates: bool = False,
    derive_rem: bool = False,
    rem_window_px: int = REM_WINDOW_PX,
    rem_percentile_q: float = REM_PERCENTILE_Q,
    force: dict | None = None,
):
    """Set up a single basin directory: clip covariates + generate config.

    ``force`` is a dict keyed by covariate stem (``"dem"``, ``"bps"``,
    ``"wtd"``, ``"hand"``, ``"soil"``, ``"rem"``) whose truthy values
    cause the corresponding covariate to be rebuilt even if the output
    file already exists.  Missing keys default to False.  Pass
    ``{"all": True}`` to force-rebuild everything.
    """
    force = force or {}
    def _force(key: str) -> bool:
        return bool(force.get("all") or force.get(key))

    _log(f"\n{'='*60}")
    _log(f"Preparing basin: {basin_key}")

    # Find basin in NWI shapefile
    match = gdf_nwi[gdf_nwi[BASIN_KEY_FIELD] == basin_key]
    if len(match) == 0:
        _log(f"  ERROR: '{basin_key}' not found in NWI shapefile (Basin field)")
        return False
    row = match.iloc[0]
    basin_id = str(row[BASIN_ID_FIELD]).strip()
    basin_name = str(row[BASIN_NAME_FIELD]).strip()
    geom = row.geometry

    # Create directory structure
    #   source/  -- user-supplied raws (you drop ETg + treatment shp here)
    #   input/   -- prep-generated clipped covariates
    #   output/  -- fill outputs
    basin_dir = BASINS_DIR / basin_key
    source_dir = basin_dir / "source"
    input_dir = basin_dir / "input"
    output_dir = basin_dir / "output"
    source_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Buffer the basin geometry for covariate clipping
    clip_crs = gdf_nwi.crs
    if clip_crs is not None and clip_crs.is_geographic:
        # Reproject to a projected CRS for buffering
        geom_series = gpd.GeoSeries([geom], crs=clip_crs).to_crs("EPSG:5070")
        clip_geom = geom_series.iloc[0].buffer(BASIN_CLIP_BUFFER_M)
        clip_crs = geom_series.crs
    else:
        clip_geom = geom.buffer(BASIN_CLIP_BUFFER_M)

    # ── Clip covariates from statewide ──────────────────────────────────────
    _log("  Clipping DEM …")
    _clip_from_statewide("DEM_statewide.tif", input_dir / "DEM.tif",
                         clip_geom, clip_crs, Resampling.bilinear,
                         force=_force("dem"))

    _log("  Clipping BpS …")
    bps_dst = input_dir / "BpS.tif"
    _clip_from_statewide("BpS_statewide.tif", bps_dst,
                         clip_geom, clip_crs, Resampling.nearest,
                         force=_force("bps"))

    # Write QGIS-compatible symbology sidecars (.clr + .qml) so the BpS
    # raster renders with LANDFIRE class names and colours in QGIS.
    try:
        from bps_utils import load_bps_lookup, write_bps_symbology
        bps_lut = load_bps_lookup()
        if bps_lut and bps_dst.exists():
            write_bps_symbology(bps_dst, bps_lut)
            _log("    → BpS.clr + BpS.qml (QGIS symbology)")
    except Exception as e:
        _log(f"    (BpS symbology sidecar skipped: {e})")

    _log("  Clipping WTD …")
    _clip_from_statewide("WTD_statewide.tif", input_dir / "WTD.tif",
                         clip_geom, clip_crs, Resampling.bilinear,
                         force=_force("wtd"))

    # ── gSSURGO soil covariates (optional -- only if statewide built) ───────
    _log("  Clipping soil covariates (AWC, SoilDepth) if available …")
    _clip_from_statewide("AWC_statewide.tif", input_dir / "AWC.tif",
                         clip_geom, clip_crs, Resampling.bilinear,
                         force=_force("soil"), quiet_missing=True)
    _clip_from_statewide("SoilDepth_statewide.tif", input_dir / "SoilDepth.tif",
                         clip_geom, clip_crs, Resampling.bilinear,
                         force=_force("soil"), quiet_missing=True)

    # ── Derive HAND from the basin's clipped DEM ────────────────────────────
    # whitebox-tools' ElevationAboveStream runs out of memory on the full
    # statewide DEM, so we derive HAND per-basin.  The basin DEM is already
    # clipped with a BASIN_CLIP_BUFFER_M (5 km) buffer, which gives the
    # stream network a little surrounding drainage context at basin edges.
    if derive_hand:
        hand_dst = input_dir / "HAND.tif"
        dem_src  = input_dir / "DEM.tif"
        if hand_dst.exists() and not _force("hand"):
            size_mb = hand_dst.stat().st_size / 1e6
            _log(f"    → HAND.tif already present ({size_mb:.1f} MB) "
                 f"— skipping (pass --force-hand to rebuild)")
        elif not dem_src.exists():
            _log("  WARNING: DEM.tif missing — cannot derive HAND")
        else:
            try:
                _derive_hand_from_dem(
                    dem_path=dem_src,
                    dst_path=hand_dst,
                    threshold_cells=hand_threshold,
                    breach_dist_cells=hand_breach_dist,
                    keep_intermediates=keep_hand_intermediates,
                )
            except SystemExit as exc:
                # _derive_hand_from_dem calls sys.exit() on failure; we don't
                # want one bad basin to abort a --all batch run, so catch it
                # here and continue without HAND for this basin.
                _log(f"  WARNING: HAND derivation failed for {basin_key}: {exc}")
                if hand_dst.exists():
                    try:
                        hand_dst.unlink()
                    except OSError:
                        pass
    else:
        _log("  HAND derivation skipped (--skip-hand)")

    # ── Derive REM from the basin's clipped DEM (opt-in) ────────────────────
    # REM is OFF by default.  Only runs when --derive-rem (or --force-rem)
    # is passed.  Uses a percentile filter over the DEM (fast, robust in
    # closed basins with no integrated drainage).
    if derive_rem or _force("rem"):
        rem_dst = input_dir / "REM.tif"
        dem_src = input_dir / "DEM.tif"
        if rem_dst.exists() and not _force("rem"):
            size_mb = rem_dst.stat().st_size / 1e6
            _log(f"    → REM.tif already present ({size_mb:.1f} MB) "
                 f"— skipping (pass --force-rem to rebuild)")
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
            except Exception as exc:
                _log(f"  WARNING: REM derivation failed for {basin_key}: {exc}")
                if rem_dst.exists():
                    try:
                        rem_dst.unlink()
                    except OSError:
                        pass

    # ── Generate config.toml ────────────────────────────────────────────────
    _generate_default_config(basin_dir, basin_key, basin_id, basin_name)

    _log(f"  Done.  Place raw ETg raster and treatment shapefile in:\n"
         f"    {source_dir.resolve()}/\n"
         f"  Prep-generated covariates live in:\n"
         f"    {input_dir.resolve()}/")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Clip covariates and generate config for one or more basins."
    )
    parser.add_argument("basins", nargs="*",
                        help="Basin key(s) from NWI shapefile Basin field "
                             "(e.g. 101_SierraValley)")
    parser.add_argument("--all", action="store_true",
                        help="Prep ALL basins in the NWI shapefile")
    parser.add_argument("--list", action="store_true",
                        help="List available basin keys and exit")
    parser.add_argument("--only-missing", action="store_true",
                        help="With --all: skip basins that already have DEM.tif")
    parser.add_argument("--skip-hand", action="store_true",
                        help="Skip HAND derivation (whitebox-tools).  Useful "
                             "for testing or if whitebox isn't installed.")
    parser.add_argument("--hand-threshold", type=int,
                        default=HAND_STREAM_THRESHOLD_CELLS,
                        help=f"Stream initiation threshold in cells for HAND "
                             f"derivation (default: {HAND_STREAM_THRESHOLD_CELLS} "
                             f"cells ≈ 0.9 km² at 30 m).")
    parser.add_argument("--hand-breach-dist", type=int,
                        default=HAND_BREACH_DIST_CELLS,
                        help=f"Max breach search distance in cells for "
                             f"BreachDepressionsLeastCost (default: "
                             f"{HAND_BREACH_DIST_CELLS} cells).")
    parser.add_argument("--keep-hand-intermediates", action="store_true",
                        help="Keep breached DEM / flow-accumulation / streams "
                             "rasters from HAND derivation for inspection.")
    parser.add_argument("--derive-rem", action="store_true",
                        help="Derive Relative Elevation Model (REM) from the "
                             "clipped DEM.  Opt-in only -- REM is OFF by "
                             "default.  Also set use_rem=true in config.toml "
                             "to include it in the fill model.")
    parser.add_argument("--rem-window-px", type=int, default=REM_WINDOW_PX,
                        help=f"REM percentile-filter window radius in pixels "
                             f"(default: {REM_WINDOW_PX} ~ 2 km at 30 m).")
    parser.add_argument("--rem-percentile-q", type=float,
                        default=REM_PERCENTILE_Q,
                        help=f"REM percentile for the valley-floor reference "
                             f"(default: {REM_PERCENTILE_Q}; 5 = robust min).")
    # Force flags.  By default any covariate whose output file already
    # exists is skipped.  Pass --force to rebuild everything, or a
    # specific --force-<name> flag to rebuild just that covariate.
    parser.add_argument("--force", action="store_true",
                        help="Rebuild all covariates even if output files "
                             "already exist.")
    parser.add_argument("--force-dem",  action="store_true", help="Rebuild DEM.tif.")
    parser.add_argument("--force-bps",  action="store_true", help="Rebuild BpS.tif.")
    parser.add_argument("--force-wtd",  action="store_true", help="Rebuild WTD.tif.")
    parser.add_argument("--force-hand", action="store_true", help="Rebuild HAND.tif.")
    parser.add_argument("--force-soil", action="store_true",
                        help="Rebuild AWC.tif and SoilDepth.tif.")
    parser.add_argument("--force-rem",  action="store_true",
                        help="Rebuild REM.tif (implies --derive-rem).")
    args = parser.parse_args()

    gdf_nwi = _load_nwi()

    if args.list:
        keys = sorted(gdf_nwi[BASIN_KEY_FIELD].dropna().unique())
        print(f"\n{len(keys)} basins in NWI_Investigations_EPSG_32611.shp:\n")
        for k in keys:
            row = gdf_nwi[gdf_nwi[BASIN_KEY_FIELD] == k].iloc[0]
            print(f"  {k:40s}  ({row[BASIN_NAME_FIELD]})")
        return

    if not args.all and not args.basins:
        parser.print_help()
        sys.exit(1)

    # Check statewide rasters exist.  HAND is derived per-basin from the
    # basin's clipped DEM, so there's no HAND_statewide.tif to check.
    for name in ("DEM_statewide.tif", "BpS_statewide.tif",
                 "WTD_statewide.tif"):
        p = STATEWIDE_DIR / name
        if not p.exists():
            _log(f"WARNING: {p} not found — run prep_statewide.py first")

    t0 = time.time()
    BASINS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        keys = sorted(gdf_nwi[BASIN_KEY_FIELD].dropna().unique())
    else:
        keys = args.basins

    n_ok, n_fail = 0, 0
    for key in keys:
        if args.only_missing:
            dem_path = BASINS_DIR / key / "input" / "DEM.tif"
            if dem_path.exists():
                continue
        if prep_one_basin(
            key, gdf_nwi,
            derive_hand=not args.skip_hand,
            hand_threshold=args.hand_threshold,
            hand_breach_dist=args.hand_breach_dist,
            keep_hand_intermediates=args.keep_hand_intermediates,
            derive_rem=args.derive_rem or args.force_rem,
            rem_window_px=args.rem_window_px,
            rem_percentile_q=args.rem_percentile_q,
            force={
                "all":  args.force,
                "dem":  args.force_dem,
                "bps":  args.force_bps,
                "wtd":  args.force_wtd,
                "hand": args.force_hand,
                "soil": args.force_soil,
                "rem":  args.force_rem,
            },
        ):
            n_ok += 1
        else:
            n_fail += 1

    _log(f"\n{'='*60}")
    _log(f"Prepped {n_ok} basins, {n_fail} failed.  "
         f"Elapsed: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
