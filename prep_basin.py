#!/usr/bin/env python3
"""
prep_basin.py
=============
Per-basin setup: clips DEM, BpS, and WTD from statewide subsets to a single
basin boundary, and generates a default ``config.toml`` if one doesn't exist.

Reads the basin polygon from NWI_Investigations.shp using the ``Basin`` field
as the key (e.g. "101_SierraValley").

Usage
-----
    # Prep a single basin:
    python prep_basin.py 101_SierraValley

    # Prep all basins that have ETg + treatment data placed in their dirs:
    python prep_basin.py --all

    # List available basin keys from the NWI shapefile:
    python prep_basin.py --list

Outputs
-------
    basins/<basin_key>/
        input/
            DEM.tif
            BpS.tif
            WTD.tif
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

_here = Path(__file__).resolve().parent
PROJECT_DIR = _here
STATEWIDE_DIR = PROJECT_DIR / "statewide"
BASINS_DIR = PROJECT_DIR / "basins"
NWI_SHP = _here / "NWI_Investigations.shp"

# Fields in NWI_Investigations.shp
BASIN_KEY_FIELD = "Basin"
BASIN_ID_FIELD  = "BasinID"
BASIN_NAME_FIELD = "BasinName"

# Buffer around basin polygon when clipping covariates (metres).
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


def _clip_from_statewide(
    src_name: str,
    dst_path: Path,
    clip_geom,
    clip_crs,
    resampling=Resampling.bilinear,
):
    """
    Clip a statewide raster to a basin polygon + buffer.
    """
    src_path = STATEWIDE_DIR / src_name
    if not src_path.exists():
        _log(f"  WARNING: {src_path.name} not found in statewide/ — skipping")
        return False

    with rasterio.open(src_path) as src:
        # Reproject clip geometry to raster CRS
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

    # Scan input/ for likely ETg and treatment files
    input_dir = basin_dir / "input"
    etg_candidates = sorted(input_dir.glob("*etg*median*.tif")) + \
                     sorted(input_dir.glob("*ETg*.tif"))
    raw_candidates = sorted(input_dir.glob("*raw*etg*.tif")) + \
                     sorted(input_dir.glob("*raw*.tif"))
    shp_candidates = sorted(input_dir.glob("*.shp"))

    # Filter shp candidates: prefer files that look like treatment/ET-unit shapefiles
    treatment_shps = [s for s in shp_candidates
                      if any(kw in s.stem.lower()
                             for kw in ("etunit", "et_unit", "phreats", "treatment",
                                        "w_ag", "master"))]
    if not treatment_shps:
        treatment_shps = shp_candidates

    etg_tif = etg_candidates[0].name if etg_candidates else "# PLACE ETg RASTER HERE"
    raw_tif = raw_candidates[0].name if raw_candidates else ""
    treat_shp = treatment_shps[0].name if treatment_shps else "# PLACE TREATMENT SHP HERE"

    toml_content = f"""\
# ETg Baseline Fill — Basin Configuration
# Basin: {basin_key}
# Generated automatically.  Edit freely — this file is never overwritten.

[basin]
basin_key  = "{basin_key}"
basin_id   = "{basin_id}"
basin_name = "{basin_name}"

[inputs]
# Filenames are relative to the input/ directory for this basin.
# The burned (adjusted) ETg raster:
etg_tif       = "{etg_tif}"
# The raw (unmodified) ETg raster (leave blank if same as etg_tif):
etg_raw_tif   = "{raw_tif}"
# Treatment polygons shapefile:
treatment_shp = "{treat_shp}"
# Covariates (auto-generated by prep_basin.py; usually no need to edit):
dem_tif       = "DEM.tif"
bps_tif       = "BpS.tif"
wtd_tif       = "WTD.tif"

[treatment]
# Buffer distance (metres) around treatment polygons.
buffer_m         = 90.0
# Gaussian feather width (pixels) outside the treatment boundary.
feather_width_px = 4
# Shapefile attribute names identifying treatment polygons.
# A polygon is treated if either attribute is > 0.
attr_scale       = "scale_fctr"
attr_replace     = "rplc_rt"

[adjustment]
# Expert adjustment knob — scale the modeled baseline up or down.
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
# is unreliable for this basin. When false, the model uses only elevation
# and slope as terrain features.
use_wtd          = true
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

    with open(config_path, "w") as f:
        f.write(toml_content)
    _log(f"  → config.toml generated (review & edit before running fill)")


def prep_one_basin(basin_key: str, gdf_nwi: gpd.GeoDataFrame):
    """Set up a single basin directory: clip covariates + generate config."""
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
    basin_dir = BASINS_DIR / basin_key
    input_dir = basin_dir / "input"
    output_dir = basin_dir / "output"
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
                         clip_geom, clip_crs, Resampling.bilinear)

    _log("  Clipping BpS …")
    _clip_from_statewide("BpS_statewide.tif", input_dir / "BpS.tif",
                         clip_geom, clip_crs, Resampling.nearest)

    _log("  Clipping WTD …")
    _clip_from_statewide("WTD_statewide.tif", input_dir / "WTD.tif",
                         clip_geom, clip_crs, Resampling.bilinear)

    # ── Generate config.toml ────────────────────────────────────────────────
    _generate_default_config(basin_dir, basin_key, basin_id, basin_name)

    _log(f"  Done.  Place ETg and treatment files in:\n"
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
    args = parser.parse_args()

    gdf_nwi = _load_nwi()

    if args.list:
        keys = sorted(gdf_nwi[BASIN_KEY_FIELD].dropna().unique())
        print(f"\n{len(keys)} basins in NWI_Investigations.shp:\n")
        for k in keys:
            row = gdf_nwi[gdf_nwi[BASIN_KEY_FIELD] == k].iloc[0]
            print(f"  {k:40s}  ({row[BASIN_NAME_FIELD]})")
        return

    if not args.all and not args.basins:
        parser.print_help()
        sys.exit(1)

    # Check statewide rasters exist
    for name in ("DEM_statewide.tif", "BpS_statewide.tif", "WTD_statewide.tif"):
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
        if prep_one_basin(key, gdf_nwi):
            n_ok += 1
        else:
            n_fail += 1

    _log(f"\n{'='*60}")
    _log(f"Prepped {n_ok} basins, {n_fail} failed.  "
         f"Elapsed: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
