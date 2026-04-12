#!/usr/bin/env python3
"""
etg_baseline_fill.py
====================
Data-driven estimation of counterfactual groundwater ET (ETg) for irrigated and
augmented areas.  Replaces uniform burn-in rates with a spatially varying baseline
predicted from terrain and ecological covariates.

Any polygon in the treatment shapefile whose ``scale_fctr`` or ``rplc_rt``
attribute is > 0 is treated: the original ETg is replaced with the
model-predicted baseline.  Gaussian edge feathering blends replaced values
with the surrounding landscape.  Polygons where both attributes are zero
are untouched (phreatophyte / riparian / wetland).

Workflow (8 steps)
------------------
1. Read ETg raster as template grid (CRS, extent, resolution).
2. Rasterize treatment shapefile → single treatment_zone mask (buffered).
3. Reproject / resample DEM and BpS to ETg grid.
4. Compute slope from DEM (Horn's method).
5. Train two-stage model on outside-treatment pixels:
       a. Per-BpS-class mean ETg.
       b. LightGBM / RandomForest residual model on elevation + slope.
6. Predict baseline ETg for all pixels.
7. Fill treatment zones with baseline + Gaussian edge feathering.
8. Per-polygon summary CSV.

Usage
-----
    cd <project folder>
    python etg_baseline_fill.py SierraValley
    python etg_baseline_fill.py PineValley

Configuration lives in config.py.  Study areas are defined there.

License: MIT (see LICENSE)
"""

import csv
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

# ── Geospatial imports -------------------------------------------------------
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize
    from rasterio.warp import reproject
    import geopandas as gpd
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\n"
        "Install with:  pip install rasterio geopandas fiona shapely pyproj"
    )

# ── ML imports ---------------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
except ImportError:
    sys.exit("Missing scikit-learn.  Install with:  pip install scikit-learn")

# ── Config --------------------------------------------------------------------
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
# Flexible config: supports both legacy config.py (SierraValley/PineValley)
# and new per-basin TOML configs via basin_config.py.
cfg = None  # set by _load_cfg() in main()


# =============================================================================
# Helper utilities
# =============================================================================

_logger = logging.getLogger("etg_fill")


def _log(msg: str) -> None:
    """Timestamped log to both console and per-basin log file."""
    stamped = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(stamped, flush=True)
    _logger.info(msg)


def _setup_logging(out_dir: Path, study_area: str) -> None:
    """Configure per-basin file logging (appends if file exists)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{study_area}_run.log"
    # Remove old handlers from previous runs in the same process
    _logger.handlers.clear()
    _logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    _logger.addHandler(fh)


def _read_raster(path: Path):
    """Return (array, profile) reading band-1 as float32."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None:
        arr[arr == np.float32(nodata)] = np.nan
    return arr, profile


def _write_raster(arr: np.ndarray, profile: dict, path: Path,
                  dtype: str = "float32") -> None:
    """Write a single-band GeoTIFF with compression."""
    prof = profile.copy()
    prof.update(
        dtype=dtype,
        count=1,
        compress=cfg.COMPRESS,
        nodata=np.nan if "float" in dtype else 255,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    if "float" in dtype:
        prof["predictor"] = cfg.PREDICTOR_TIFF
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
    _log(f"  → wrote {path.name}  ({arr.shape[1]}×{arr.shape[0]})")


def _match_raster(src_path: Path, ref_profile: dict, resampling, dst_path: Path,
                  categorical: bool = False) -> np.ndarray:
    """
    Reproject *src_path* to match the grid defined by *ref_profile*.
    Returns the reprojected array and writes to *dst_path*.
    """
    dst_crs       = ref_profile["crs"]
    dst_transform = ref_profile["transform"]
    dst_w         = ref_profile["width"]
    dst_h         = ref_profile["height"]

    with rasterio.open(src_path) as src:
        src_nodata    = src.nodata
        src_transform = src.transform
        src_crs       = src.crs

        stem = src_path.stem
        if stem in cfg.CRS_OVERRIDES:
            from rasterio.crs import CRS as _CRS
            src_crs = _CRS.from_string(cfg.CRS_OVERRIDES[stem])
            _log(f"    CRS override for {stem}: → {src_crs}")

        src_arr = src.read(1)

    if categorical:
        dst_arr    = np.zeros((dst_h, dst_w), dtype=np.int32)
        dst_nodata = 0
    else:
        dst_arr    = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
        dst_nodata = np.nan

    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata if not categorical else 0,
    )

    out_profile = ref_profile.copy()
    out_profile.update(
        dtype=dst_arr.dtype.name,
        count=1,
        nodata=dst_nodata if not categorical else 0,
        compress=cfg.COMPRESS,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    if not categorical:
        out_profile["predictor"] = cfg.PREDICTOR_TIFF
    with rasterio.open(dst_path, "w", **out_profile) as dst:
        dst.write(dst_arr, 1)

    _log(f"  → matched {src_path.name} → {dst_path.name}")
    return dst_arr


def _slope_from_dem(dem: np.ndarray, cellsize: float) -> np.ndarray:
    """Slope in degrees via Horn's method (3×3 finite-difference)."""
    padded = np.pad(dem, 1, mode="constant", constant_values=np.nan)
    dy, dx = np.gradient(padded, cellsize)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    return slope_deg[1:-1, 1:-1].astype(np.float32)


# =============================================================================
# Main workflow
# =============================================================================

def _load_cfg(study_area: str):
    """
    Load the right config module based on the study_area argument.

    - If a ``basins/<study_area>/config.toml`` exists → use basin_config.py
    - Otherwise fall back to the legacy config.py (SierraValley / PineValley)
    """
    global cfg
    basins_dir = _here / "basins"
    toml_path = basins_dir / study_area / "config.toml"

    if toml_path.exists():
        import basin_config as _cfg
        _cfg.load_basin(study_area)
        cfg = _cfg
    else:
        import config as _cfg
        _cfg.load_study_area(study_area)
        cfg = _cfg


def main(study_area: str | None = None) -> None:
    # ── Resolve study area ──────────────────────────────────────────────────
    if study_area is None:
        if len(sys.argv) > 1:
            study_area = sys.argv[1]
        else:
            # List available from both legacy and basin configs
            import basin_config as _bc
            import config as _legacy
            all_areas = sorted(set(_bc.available_areas()) |
                               set(_legacy.available_areas()))
            sys.exit(
                f"Usage: python etg_baseline_fill.py <study_area>\n"
                f"  Available: {', '.join(all_areas)}"
            )
    _load_cfg(study_area)
    _log(f"═══ Study area: {cfg.STUDY_AREA_NAME} ═══")

    t0 = time.time()
    out_dir = cfg.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir, cfg.STUDY_AREA_NAME)

    # ── 1. Read ETg (template) ───────────────────────────────────────────────
    _log("1 · Reading ETg raster (template grid) …")
    etg, etg_prof = _read_raster(cfg.ETG_TIF)
    original_valid = np.isfinite(etg)  # preserve original extent mask
    _log(f"    shape={etg.shape}  CRS={etg_prof['crs']}  "
         f"valid px={original_valid.sum():,}")

    grid_shape = (etg_prof["height"], etg_prof["width"])
    grid_transform = etg_prof["transform"]

    # ── 2. Rasterize treatment shapefile ─────────────────────────────────────
    _log("2 · Rasterizing treatment zones …")
    gdf = gpd.read_file(cfg.TREATMENT_SHP)

    # Reproject to ETg CRS if needed
    etg_crs = etg_prof["crs"]
    if gdf.crs is None:
        _log("    WARNING: shapefile has no CRS – assuming it matches ETg")
    elif not gdf.crs.equals(etg_crs):
        _log(f"    reprojecting from {gdf.crs} → {etg_crs}")
        gdf = gdf.to_crs(etg_crs)

    # Ensure attribute columns exist and fill NaN with 0
    for col in (cfg.ATTR_SCALE, cfg.ATTR_REPLACE):
        if col not in gdf.columns:
            sys.exit(f"ERROR: attribute '{col}' not found in {cfg.TREATMENT_SHP.name}")
        gdf[col] = gdf[col].fillna(0).astype(float)

    # Any polygon with either attribute > 0 is a treatment polygon
    gdf["_is_treated"] = (gdf[cfg.ATTR_SCALE] > 0) | (gdf[cfg.ATTR_REPLACE] > 0)
    n_treated = int(gdf["_is_treated"].sum())
    n_skip    = len(gdf) - n_treated
    _log(f"    polygons:  {n_treated} treatment  |  {n_skip} untouched")

    # ── 2a. Treatment-zone mask (with optional buffer) ───────────────────────
    gdf_treat = gdf[gdf["_is_treated"]].copy()
    if cfg.BUFFER_M > 0 and len(gdf_treat) > 0:
        _log(f"    buffering treatment polygons by {cfg.BUFFER_M} CRS-units")
        gdf_treat["geometry"] = gdf_treat.geometry.buffer(cfg.BUFFER_M)

    treat_shapes = [
        (geom, 1)
        for geom in gdf_treat.geometry
        if geom is not None and geom.is_valid
    ]
    treatment_zone = rasterize(
        treat_shapes,
        out_shape=grid_shape,
        transform=grid_transform,
        fill=0,
        dtype=np.uint8,
    ) if treat_shapes else np.zeros(grid_shape, dtype=np.uint8)

    # ── 2b. Per-pixel adjustment factor raster ─────────────────────────────
    # Start with the basin-wide default; overlay per-polygon overrides.
    basin_adjust = getattr(cfg, "BASELINE_ADJUST", 1.0)
    attr_adjust  = getattr(cfg, "ATTR_ADJUST", "adj_fctr")
    adjust_raster = np.full(grid_shape, basin_adjust, dtype=np.float32)

    has_per_poly_adj = (
        attr_adjust
        and attr_adjust in gdf.columns
        and gdf[gdf["_is_treated"]][attr_adjust].fillna(0).astype(float).gt(0).any()
    )
    if has_per_poly_adj:
        _log(f"   2b · Rasterizing per-polygon adjustment factors "
             f"(column '{attr_adjust}') …")
        gdf_adj = gdf[gdf["_is_treated"]].copy()
        gdf_adj[attr_adjust] = gdf_adj[attr_adjust].fillna(0).astype(float)
        # Only burn polygons that actually have an override (> 0)
        adj_shapes = []
        for _, r in gdf_adj[gdf_adj[attr_adjust] > 0].iterrows():
            geom = r.geometry
            if geom is not None and geom.is_valid:
                adj_shapes.append((geom, float(r[attr_adjust])))
        if adj_shapes:
            # Burn per-polygon values onto the adjustment raster
            per_poly_adj = rasterize(
                adj_shapes,
                out_shape=grid_shape,
                transform=grid_transform,
                fill=0.0,
                dtype="float32",
            )
            # Where a polygon override was burned (> 0), use it; else keep default
            override_mask = per_poly_adj > 0
            adjust_raster[override_mask] = per_poly_adj[override_mask]
            n_override_px = int(override_mask.sum())
            _log(f"    per-polygon overrides applied to {n_override_px:,} pixels")
    else:
        if attr_adjust and attr_adjust not in gdf.columns:
            _log(f"   2b · Per-polygon adjustment column '{attr_adjust}' not found "
                 f"in shapefile — using basin-wide default ({basin_adjust})")
        else:
            _log(f"   2b · No per-polygon adjustment overrides — "
                 f"using basin-wide default ({basin_adjust})")

    adjustment_active = (basin_adjust != 1.0) or has_per_poly_adj
    if adjustment_active:
        _log(f"    expert adjustment ACTIVE — basin default: {basin_adjust}")

    _write_raster(treatment_zone, etg_prof, out_dir / "treatment_zone.tif", dtype="uint8")
    n_treat_px = int(treatment_zone.sum())
    _log(f"    treatment-zone pixels: {n_treat_px:,}")

    # ── 2d. Prepare training ETg ────────────────────────────────────────────
    if cfg.ETG_RAW_TIF is not None:
        _log("   2d · Matching raw ETg raster to grid …")
        etg_raw = _match_raster(
            cfg.ETG_RAW_TIF, etg_prof, Resampling.bilinear,
            out_dir / "ETg_raw_matched.tif",
        )
    else:
        _log("   2d · Building training ETg (NaN-ing all treatment pixels) …")
        etg_raw = etg.copy()
        # The input raster has modified (scaled/replaced) values inside treatment
        # zones, so we can't trust those pixels for training or as raw values.
        # Mark them all as NaN — the model will predict baseline for them.
        treat_mask = treatment_zone.astype(bool)
        etg_raw[treat_mask] = np.nan
        n_masked = int(treat_mask.sum())
        _log(f"    masked {n_masked:,} treatment-zone pixels as NaN")

    # ── 3. Reproject / resample DEM and BpS ──────────────────────────────────
    _log("3 · Matching DEM to ETg grid …")
    dem = _match_raster(
        cfg.DEM_TIF, etg_prof, Resampling.bilinear,
        out_dir / "DEM_matched.tif",
    )
    _log("   Matching BpS to ETg grid …")
    bps = _match_raster(
        cfg.BPS_TIF, etg_prof, Resampling.nearest,
        out_dir / "BpS_matched.tif",
        categorical=True,
    )

    # Optional: water table depth (Ma et al., 2025)
    wtd = None
    use_wtd = getattr(cfg, "USE_WTD", True)
    if not use_wtd:
        _log("   WTD disabled in config (use_wtd = false) — skipping")
    elif getattr(cfg, "WTD_TIF", None) is not None and cfg.WTD_TIF is not None:
        _log("   Matching WTD to ETg grid …")
        wtd_candidate = _match_raster(
            cfg.WTD_TIF, etg_prof, Resampling.bilinear,
            out_dir / "WTD_matched.tif",
        )
        n_wtd_valid = int(np.isfinite(wtd_candidate).sum())
        if n_wtd_valid > 0:
            wtd = wtd_candidate
            _log(f"    WTD valid pixels: {n_wtd_valid:,}  "
                 f"range: {np.nanmin(wtd):.1f} – {np.nanmax(wtd):.1f} m")
        else:
            _log("  ⚠ WTD raster has 0 valid pixels after reprojection — "
                 "check extent / nodata / CRS.  Continuing WITHOUT WTD.")
            # Diagnose: report the source raster's extent and nodata value
            with rasterio.open(cfg.WTD_TIF) as _src:
                _log(f"    WTD source bounds : {_src.bounds}")
                _log(f"    WTD source CRS    : {_src.crs}")
                _log(f"    WTD source nodata : {_src.nodata}")
                _log(f"    WTD source shape  : {_src.shape}")
                _raw = _src.read(1)
                _finite = np.isfinite(_raw.astype(np.float32))
                if _src.nodata is not None:
                    _finite &= (_raw != _src.nodata)
                _log(f"    WTD source finite pixels (pre-reproject): {int(_finite.sum()):,}")
    else:
        _log("   WTD raster not configured — skipping")

    # Optional: Height Above Nearest Drainage (HAND) — derived from DEM in
    # prep_statewide.py via whitebox-tools, encoded in metres above the nearest
    # downslope stream cell.  Informative for groundwater ET in phreatophyte
    # / playa-fringe systems where distance-to-drainage matters.
    hand = None
    use_hand = getattr(cfg, "USE_HAND", True)
    if not use_hand:
        _log("   HAND disabled in config (use_hand = false) — skipping")
    elif getattr(cfg, "HAND_TIF", None) is not None and cfg.HAND_TIF is not None:
        _log("   Matching HAND to ETg grid …")
        hand_candidate = _match_raster(
            cfg.HAND_TIF, etg_prof, Resampling.bilinear,
            out_dir / "HAND_matched.tif",
        )
        n_hand_valid = int(np.isfinite(hand_candidate).sum())
        if n_hand_valid > 0:
            hand = hand_candidate
            _log(f"    HAND valid pixels: {n_hand_valid:,}  "
                 f"range: {np.nanmin(hand):.1f} – {np.nanmax(hand):.1f} m")
        else:
            _log("  ⚠ HAND raster has 0 valid pixels after reprojection — "
                 "check extent / nodata / CRS.  Continuing WITHOUT HAND.")
            # Diagnose: report the source raster's metadata so the user can
            # see what went wrong (mirrors the WTD diagnostic block).
            try:
                with rasterio.open(cfg.HAND_TIF) as _src:
                    _log(f"    HAND source bounds : {_src.bounds}")
                    _log(f"    HAND source CRS    : {_src.crs}")
                    _log(f"    HAND source nodata : {_src.nodata}")
                    _log(f"    HAND source shape  : {_src.shape}")
                    _log(f"    HAND source dtype  : {_src.dtypes[0]}")
                    _raw = _src.read(1)
                    _finite = np.isfinite(_raw.astype(np.float32))
                    if _src.nodata is not None:
                        _finite &= (_raw != _src.nodata)
                    _log(f"    HAND source finite pixels (pre-reproject): "
                         f"{int(_finite.sum()):,}")
                    if _finite.any():
                        _vals = _raw[_finite]
                        _log(f"    HAND source value range: "
                             f"{float(_vals.min()):.2f} – {float(_vals.max()):.2f}")
            except Exception as _e:
                _log(f"    (could not diagnose HAND source: {_e})")
    else:
        _log("   HAND raster not configured — skipping")

    # ── 4. Derive slope ──────────────────────────────────────────────────────
    _log("4 · Computing slope from DEM …")
    cellsize = abs(etg_prof["transform"].a)
    slope = _slope_from_dem(dem, cellsize)
    _write_raster(slope, etg_prof, out_dir / "slope_matched.tif")

    # ── 4b. Basin boundary mask (constrain training to within the basin) ────
    # If the NWI shapefile is available, rasterize the basin polygon so
    # training only draws from within-basin pixels.
    basin_mask = None
    nwi_path = _here / "NWI_Investigations_EPSG_32611.shp"
    if nwi_path.exists():
        _log("  4b · Rasterizing basin boundary (NWI) for training mask …")
        gdf_nwi = gpd.read_file(nwi_path)
        basin_key = cfg.STUDY_AREA_NAME
        nwi_match = gdf_nwi[gdf_nwi["Basin"] == basin_key]
        if len(nwi_match) == 0:
            # Try matching on BasinName for legacy study areas
            compare_key = (basin_key.split("_", 1)[-1]
                           if "_" in basin_key else basin_key)
            nwi_match = gdf_nwi[
                gdf_nwi["BasinName"].str.replace(" ", "") == compare_key
            ]
        if len(nwi_match) > 0:
            nwi_geom = nwi_match.iloc[0].geometry
            if nwi_match.crs is not None and not nwi_match.crs.equals(etg_crs):
                nwi_series = gpd.GeoSeries([nwi_geom], crs=nwi_match.crs)
                nwi_geom = nwi_series.to_crs(etg_crs).iloc[0]
            basin_mask = rasterize(
                [(nwi_geom, 1)],
                out_shape=grid_shape,
                transform=grid_transform,
                fill=0,
                dtype=np.uint8,
            ).astype(bool)
            n_basin_px = int(basin_mask.sum())
            _log(f"    basin boundary pixels: {n_basin_px:,}")
        else:
            _log(f"    WARNING: basin key '{basin_key}' not found in NWI shapefile "
                 f"— training will use all valid pixels")
    else:
        _log("    NWI shapefile not found — training will use all valid pixels")

    # ── 5. Build training set (outside ALL treatment zones, within basin) ───
    _log("5 · Assembling training data …")
    # Train on etg_raw (recovered/true ETg), NOT the adjusted raster.
    # Exclude treatment zones, steep slopes, and out-of-basin pixels.
    valid = (
        (treatment_zone == 0)
        & np.isfinite(etg_raw)
        & np.isfinite(dem)
        & np.isfinite(slope)
        & (bps > 0)
        & (etg_raw > 0)
    )
    if basin_mask is not None:
        n_before_basin = int(valid.sum())
        valid &= basin_mask
        n_after_basin = int(valid.sum())
        _log(f"    basin boundary filter: {n_before_basin:,} → {n_after_basin:,} "
             f"({n_before_basin - n_after_basin:,} out-of-basin pixels excluded)")
    if wtd is not None:
        valid &= np.isfinite(wtd)
    if hand is not None:
        valid &= np.isfinite(hand)
    max_slope = getattr(cfg, "MAX_SLOPE_DEG", None)
    if max_slope is not None:
        n_before_slope = int(valid.sum())
        valid &= (slope <= max_slope)
        n_after_slope = int(valid.sum())
        _log(f"    slope filter (≤{max_slope}°): {n_before_slope:,} → {n_after_slope:,} "
             f"({n_before_slope - n_after_slope:,} steep pixels excluded)")
    n_valid = int(valid.sum())
    _log(f"    valid training pixels: {n_valid:,}")

    # Graceful skip: if too few training pixels, write a skip marker and return
    MIN_TRAIN_PIXELS = 50
    if n_valid < MIN_TRAIN_PIXELS:
        msg = (f"WARNING: only {n_valid} valid training pixels "
               f"(minimum {MIN_TRAIN_PIXELS}) — skipping basin.")
        _log(msg)
        skip_path = out_dir / f"{cfg.STUDY_AREA_NAME}_SKIPPED.txt"
        skip_path.write_text(f"{msg}\n")
        _log(f"  → wrote {skip_path.name}")
        return

    idx = np.where(valid.ravel())[0]
    if cfg.MAX_TRAIN_PIXELS and n_valid > cfg.MAX_TRAIN_PIXELS:
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        idx = rng.choice(idx, size=cfg.MAX_TRAIN_PIXELS, replace=False)
        _log(f"    sub-sampled to {len(idx):,} pixels")

    y_train    = etg_raw.ravel()[idx]
    bps_flat   = bps.ravel()[idx].astype(np.int32)
    elev_flat  = dem.ravel()[idx]
    slope_flat = slope.ravel()[idx]
    wtd_flat   = wtd.ravel()[idx] if wtd is not None else None
    hand_flat  = hand.ravel()[idx] if hand is not None else None

    # ── 5a. Per-BpS-class mean ETg ───────────────────────────────────────────
    _log("   5a · Computing per-BpS mean ETg …")
    unique_bps = np.unique(bps_flat)
    bps_mean_map = {}
    for b in unique_bps:
        bps_mean_map[b] = float(np.nanmean(y_train[bps_flat == b]))
    _log(f"    BpS classes in training data: {len(unique_bps)}")

    bps_mean_train = np.array([bps_mean_map.get(b, 0.0) for b in bps_flat],
                              dtype=np.float32)
    residual_train = y_train - bps_mean_train

    # ── 5b. Train residual model on terrain features ───────────────────────
    # Build the feature stack dynamically.  Elevation + slope are always
    # included; WTD and HAND are added only if their rasters loaded cleanly
    # and the user hasn't disabled them in config.
    feat_columns = [elev_flat, slope_flat]
    feature_names = ["elevation", "slope"]
    if wtd_flat is not None:
        feat_columns.append(wtd_flat)
        feature_names.append("wtd")
    if hand_flat is not None:
        feat_columns.append(hand_flat)
        feature_names.append("hand")
    import pandas as pd
    X_train = pd.DataFrame(np.column_stack(feat_columns), columns=feature_names)

    _log(f"   5b · Training {cfg.MODEL_BACKEND.upper()} on residuals …")
    if cfg.MODEL_BACKEND == "lgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            sys.exit("LightGBM requested but not installed.  pip install lightgbm")
        model = lgb.LGBMRegressor(
            n_estimators=cfg.LGBM_N_ESTIMATORS,
            max_depth=cfg.LGBM_MAX_DEPTH,
            learning_rate=cfg.LGBM_LEARNING_RATE,
            num_leaves=cfg.LGBM_NUM_LEAVES,
            min_child_samples=cfg.LGBM_MIN_CHILD,
            n_jobs=cfg.LGBM_N_JOBS,
            random_state=cfg.RANDOM_SEED,
            verbose=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=cfg.RF_N_ESTIMATORS,
            max_depth=cfg.RF_MAX_DEPTH,
            min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF,
            n_jobs=cfg.RF_N_JOBS,
            random_state=cfg.RANDOM_SEED,
        )

    # ── Early stopping (LightGBM only) ────────────────────────────────────
    # Hold out 20% for early-stopping validation so the model stops adding
    # trees once the validation loss plateaus.  This prevents over-training
    # on small basins where the terrain signal is weak.
    from sklearn.model_selection import train_test_split
    n_trees_used = None     # will be set to actual trees after early stopping
    if cfg.MODEL_BACKEND == "lgbm" and len(X_train) >= 200:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train, residual_train,
            test_size=0.2, random_state=cfg.RANDOM_SEED,
        )
        _log(f"    early-stopping split: {len(X_fit):,} train / "
             f"{len(X_val):,} validation")
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=0),   # suppress per-round logging
            ],
        )
        n_trees_used = model.best_iteration_ if model.best_iteration_ > 0 \
            else model.n_estimators
        _log(f"    early stopping: {n_trees_used} / "
             f"{cfg.LGBM_N_ESTIMATORS} trees used")
        # Refit on the FULL training set with the optimal number of trees
        # so that no data is wasted for the final predictions.
        model.set_params(n_estimators=n_trees_used)
        model.fit(X_train, residual_train)
    else:
        model.fit(X_train, residual_train)

    _log("    cross-validating (3-fold) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = cross_val_score(model, X_train, residual_train, cv=3,
                             scoring="r2", n_jobs=cfg.RF_N_JOBS)
    cv_mean = cv.mean()
    _log(f"    residual-model CV R²: {cv_mean:.4f} ± {cv.std():.4f}")

    imp = model.feature_importances_
    for name, val in zip(feature_names, imp):
        _log(f"      {name:>12s}  importance = {val:.4f}")

    # ── Fallback decision: skip residual model if it hurts ─────────────────
    # A negative CV R² means the residual predictions make the baseline
    # WORSE than the BpS class means alone.  In that case we zero out
    # the residual so the final baseline is just the per-BpS mean.
    use_residual = cv_mean >= 0.0
    if not use_residual:
        _log(f"    ⚠ CV R² is negative ({cv_mean:.4f}) — the residual model "
             f"is hurting predictions.  Falling back to BpS-mean-only baseline.")

    # ── 6. Predict baseline ETg (for replacement zones) ──────────────────────
    _log("6 · Predicting baseline ETg …")

    # Full-raster BpS mean layer
    bps_mean_full = np.zeros_like(etg)
    for b, m in bps_mean_map.items():
        bps_mean_full[bps == b] = m
    global_mean = float(np.nanmean(y_train))
    unseen = ~np.isin(bps, list(bps_mean_map.keys())) & (bps > 0)
    bps_mean_full[unseen] = global_mean

    # Predict residual in chunks (or skip if residual model hurts)
    residual_pred = np.full_like(etg, np.nan)
    if use_residual:
        pred_mask = np.isfinite(dem) & np.isfinite(slope) & (bps > 0)
        if wtd is not None:
            pred_mask &= np.isfinite(wtd)
        if hand is not None:
            pred_mask &= np.isfinite(hand)
        pred_idx = np.where(pred_mask.ravel())[0]

        # Pre-flatten the covariate rasters once
        dem_r   = dem.ravel()
        slope_r = slope.ravel()
        wtd_r   = wtd.ravel()  if wtd  is not None else None
        hand_r  = hand.ravel() if hand is not None else None

        CHUNK = 500_000
        for start in range(0, len(pred_idx), CHUNK):
            chunk_idx = pred_idx[start:start + CHUNK]
            cols = [dem_r[chunk_idx], slope_r[chunk_idx]]
            if wtd_r is not None:
                cols.append(wtd_r[chunk_idx])
            if hand_r is not None:
                cols.append(hand_r[chunk_idx])
            X_chunk = pd.DataFrame(np.column_stack(cols), columns=feature_names)
            residual_pred.ravel()[chunk_idx] = model.predict(X_chunk).astype(np.float32)
    else:
        # BpS-mean-only: residual is zero everywhere so baseline = bps_mean_full
        residual_pred = np.zeros_like(etg)

    baseline_raw = bps_mean_full + residual_pred
    n_neg_clipped = int(((baseline_raw < 0) & np.isfinite(baseline_raw)).sum())
    baseline = np.maximum(baseline_raw, 0.0)
    if n_neg_clipped > 0:
        _log(f"    ⚠ {n_neg_clipped:,} pixels had negative baseline predictions "
             f"— clipped to 0.0")
    sa = cfg.STUDY_AREA_NAME   # short alias for file naming
    _write_raster(baseline, etg_prof, out_dir / f"{sa}_ETg_baseline_pred.tif")

    # ── 7. Build final ETg raster ────────────────────────────────────────────
    # All treatment-zone pixels in etg_raw are NaN (unless a raw raster was
    # supplied).  We start from etg_raw (= etg outside treatment, NaN inside)
    # and fill treatment pixels with the appropriate method.
    _log("7 · Building final ETg raster …")

    etg_treated = etg_raw.copy()

    # ── 7a. Replace all treatment-zone pixels with adjusted baseline ─────
    treat_mask = (treatment_zone == 1) & np.isfinite(baseline)
    n_replaced = int(treat_mask.sum())

    if adjustment_active:
        # Apply per-pixel adjustment: baseline * adjustment_factor
        adjusted_baseline = baseline * adjust_raster
        # Re-clip negatives introduced by very low adjustment factors
        adjusted_baseline = np.maximum(adjusted_baseline, 0.0)
        adj_vals = adjust_raster[treat_mask]
        _log(f"    adjustment factors in treatment zone — "
             f"min: {np.min(adj_vals):.3f}  max: {np.max(adj_vals):.3f}  "
             f"mean: {np.mean(adj_vals):.3f}")
    else:
        adjusted_baseline = baseline.copy()

    # ── Downward-only cap ────────────────────────────────────────────────
    # The baseline replacement must never INCREASE ETg in a treatment zone
    # relative to the original input raster.  If the model predicts a
    # higher rate than the hydrologist's burned-in value, we keep the
    # original value instead.  This guarantees the workflow only removes
    # irrigation-inflated signal — it never adds ET.
    cap_mask = treat_mask & np.isfinite(etg) & (adjusted_baseline > etg)
    n_capped = int(cap_mask.sum())
    if n_capped > 0:
        adjusted_baseline[cap_mask] = etg[cap_mask]
        _log(f"    downward-only cap applied to {n_capped:,} pixels "
             f"(baseline exceeded original input ETg)")

    etg_treated[treat_mask] = adjusted_baseline[treat_mask]
    if adjustment_active:
        _log(f"    treatment pixels filled with adjusted baseline: {n_replaced:,}"
             f"  /  {n_treat_px:,}")
    else:
        _log(f"    treatment pixels filled with baseline: {n_replaced:,}"
             f"  /  {n_treat_px:,}")

    # ── 7b. Edge feathering (Gaussian blend OUTSIDE the treatment boundary) ──
    # Create a smooth transition in a ring just outside the treatment zone,
    # blending the raw ETg toward the baseline-replaced values at the boundary.
    # Inside the treatment zone: 100 % baseline (weight = 1).
    # At the treatment boundary: weight transitions from 1 → 0 moving outward.
    # Beyond the feather distance: 100 % raw ETg (weight = 0, unchanged).
    if cfg.FEATHER_WIDTH_PX > 0:
        sigma = float(cfg.FEATHER_WIDTH_PX)
        _log(f"   7b · Gaussian feathering OUTSIDE treatment boundary "
             f"(sigma={sigma:.0f} px, ~{sigma * cellsize:.0f} m) …")

        treat_bool = treatment_zone.astype(bool)
        # Distance from each non-treatment pixel to the nearest treatment pixel
        outside_zone = ~treat_bool
        dist_outside = distance_transform_edt(outside_zone)

        # Gaussian weight: 1 at the boundary (dist=0), decaying to 0 far away
        # Only applies to the ring of pixels just outside the treatment zone.
        blend_weight = np.exp(-(dist_outside / sigma) ** 2).astype(np.float32)

        # Inside treatment: weight = 1 (fully baseline)
        blend_weight[treat_bool] = 1.0

        # Build feather band mask: outside-zone pixels within meaningful range
        feather_band = outside_zone & (blend_weight > 0.01)

        # Blend: in the feather band, mix adjusted baseline with raw ETg.
        # etg_treated already has adjusted baseline inside treatment and raw
        # outside, so for the feather band we blend adjusted baseline with raw.
        etg_final = etg_treated.copy()
        fb = feather_band & np.isfinite(etg_raw) & np.isfinite(adjusted_baseline)
        blended = (
            blend_weight[fb] * adjusted_baseline[fb]
            + (1.0 - blend_weight[fb]) * etg_raw[fb]
        )
        # Downward-only: feathering must never raise ETg above the raw value
        blended = np.minimum(blended, etg_raw[fb])
        etg_final[fb] = blended

        n_feathered = int(((blend_weight > 0.01) & (blend_weight < 0.99)
                           & outside_zone).sum())
        _log(f"    feather-band pixels (partial blend outside boundary): "
             f"{n_feathered:,}")

        # Write the blend weight raster for inspection
        # Show weight for treatment + feather band; NaN elsewhere
        blend_out = np.where(treat_bool | feather_band, blend_weight, np.nan)
        _write_raster(blend_out.astype(np.float32), etg_prof,
                      out_dir / "feather_weight.tif")
    else:
        _log("    feathering disabled (FEATHER_WIDTH_PX = 0)")
        etg_final = etg_treated

    # Enforce original raster extent: any pixel that was nodata in the input
    # stays nodata in the output, even if the buffer or feathering filled it.
    n_clipped_extent = int((np.isfinite(etg_final) & ~original_valid).sum())
    if n_clipped_extent > 0:
        etg_final[~original_valid] = np.nan
        _log(f"    extent mask applied: {n_clipped_extent:,} pixels outside "
             f"original ETg extent set back to nodata")

    _write_raster(etg_final, etg_prof, out_dir / f"{sa}_ETg_final.tif")

    # ── 7d. Per-pixel percent change raster and figure ──────────────────────
    _log("   7d · Computing per-pixel percent change (raw → final) …")
    # Use etg_raw (the unmodified Landsat-derived ETg) as the reference
    pct_change = np.full_like(etg_raw, np.nan, dtype=np.float32)
    denom_valid = np.isfinite(etg_raw) & (etg_raw > 0) & np.isfinite(etg_final)
    pct_change[denom_valid] = (
        100.0 * (etg_final[denom_valid] - etg_raw[denom_valid]) / etg_raw[denom_valid]
    )
    _write_raster(pct_change, etg_prof, out_dir / f"{sa}_ETg_pct_change.tif")

    # Summary stats on treatment zones only
    pct_treat = pct_change[treatment_zone.astype(bool) & np.isfinite(pct_change)]
    if len(pct_treat) > 0:
        _log(f"    treatment-zone % change — "
             f"mean: {np.mean(pct_treat):+.1f}%  "
             f"median: {np.median(pct_treat):+.1f}%  "
             f"p10: {np.percentile(pct_treat, 10):+.1f}%  "
             f"p90: {np.percentile(pct_treat, 90):+.1f}%")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 8))
        vabs = min(np.nanpercentile(np.abs(pct_treat), 98), 100) if len(pct_treat) > 0 else 50
        im = ax.imshow(pct_change, cmap="RdBu", vmin=-vabs, vmax=vabs)
        ax.set_title("Per-pixel ETg change: raw → final (% change)", fontsize=13, pad=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04,
                     pad=0.04, shrink=0.6, label="% change (negative = ETg reduced)")
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        fig.savefig(out_dir / f"{sa}_diag_pct_change_map.png", dpi=150)
        plt.close(fig)
        _log(f"  → wrote diag_pct_change_map.png")
    except Exception as e:
        _log(f"    (skipped percent-change figure: {e})")

    # ── 7e. Metadata / provenance file ──────────────────────────────────────
    _log("   7e · Writing run metadata …")
    import datetime

    meta_lines = [
        "# ETg Baseline Fill — Run Metadata",
        f"# Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"# Study area: {cfg.STUDY_AREA_NAME}",
        "",
        "[inputs]",
        f"study_area       = {cfg.STUDY_AREA_NAME}",
        f"etg_tif          = {cfg.ETG_TIF}",
        f"etg_raw_tif      = {cfg.ETG_RAW_TIF}",
        f"dem_tif          = {cfg.DEM_TIF}",
        f"bps_tif          = {cfg.BPS_TIF}",
        f"wtd_tif          = {getattr(cfg, 'WTD_TIF', None)}",
        f"hand_tif         = {getattr(cfg, 'HAND_TIF', None)}",
        f"treatment_shp    = {cfg.TREATMENT_SHP}",
        "",
        "[grid]",
        f"crs              = {etg_prof['crs']}",
        f"shape            = {grid_shape}",
        f"pixel_size       = {abs(etg_prof['transform'].a):.1f} m",
        "",
        "[treatment]",
        f"buffer_m         = {cfg.BUFFER_M}",
        f"feather_width_px = {cfg.FEATHER_WIDTH_PX}",
        f"baseline_adjust  = {basin_adjust}",
        f"attr_adjust      = {attr_adjust}",
        f"adjustment_active = {'yes' if adjustment_active else 'no'}",
        f"per_polygon_overrides = {'yes' if has_per_poly_adj else 'no'}",
        "",
        "[model]",
        f"backend          = {cfg.MODEL_BACKEND}",
        f"use_wtd          = {getattr(cfg, 'USE_WTD', True)}",
        f"use_hand         = {getattr(cfg, 'USE_HAND', True)}",
        f"max_slope_deg    = {getattr(cfg, 'MAX_SLOPE_DEG', None)}",
        f"max_train_pixels = {cfg.MAX_TRAIN_PIXELS}",
        f"random_seed      = {cfg.RANDOM_SEED}",
    ]

    # Add model-specific hyperparameters
    if cfg.MODEL_BACKEND == "lgbm":
        meta_lines += [
            f"lgbm_n_estimators  = {cfg.LGBM_N_ESTIMATORS}",
            f"lgbm_trees_used    = {n_trees_used if n_trees_used is not None else cfg.LGBM_N_ESTIMATORS}",
            f"lgbm_max_depth     = {cfg.LGBM_MAX_DEPTH}",
            f"lgbm_learning_rate = {cfg.LGBM_LEARNING_RATE}",
            f"lgbm_num_leaves    = {cfg.LGBM_NUM_LEAVES}",
            f"lgbm_min_child     = {cfg.LGBM_MIN_CHILD}",
            f"early_stopping     = {'yes' if n_trees_used is not None else 'no (too few samples)'}",
        ]
    else:
        meta_lines += [
            f"rf_n_estimators     = {cfg.RF_N_ESTIMATORS}",
            f"rf_max_depth        = {cfg.RF_MAX_DEPTH}",
            f"rf_min_samples_leaf = {cfg.RF_MIN_SAMPLES_LEAF}",
        ]
    meta_lines.append(f"use_residual_model = {'yes' if use_residual else 'no (CV R² < 0, BpS-mean only)'}")

    meta_lines += [
        "",
        "[crs_overrides]",
    ]
    for stem, crs_val in cfg.CRS_OVERRIDES.items():
        meta_lines.append(f"{stem} = {crs_val}")

    meta_lines += [
        "",
        "[training]",
        f"total_valid_pixels    = {n_valid:,}",
        f"basin_boundary_mask   = {'yes' if basin_mask is not None else 'no (NWI not found)'}",
        f"features              = {feature_names}",
        f"cv_r2_mean            = {cv.mean():.4f}",
        f"cv_r2_std             = {cv.std():.4f}",
    ]

    # Feature importance
    meta_lines.append("")
    meta_lines.append("[feature_importance]")
    for name, val in zip(feature_names, imp):
        meta_lines.append(f"{name:20s} = {val:.4f}")

    meta_lines += [
        "",
        "[results]",
        f"treatment_pixels      = {n_treat_px:,}",
        f"bps_classes_trained   = {len(unique_bps)}",
        f"negative_baseline_clipped = {n_neg_clipped:,}",
    ]

    if len(pct_treat) > 0:
        meta_lines += [
            "",
            "[pct_change_treatment_zones]",
            f"mean   = {np.mean(pct_treat):+.2f}%",
            f"median = {np.median(pct_treat):+.2f}%",
            f"p10    = {np.percentile(pct_treat, 10):+.2f}%",
            f"p90    = {np.percentile(pct_treat, 90):+.2f}%",
        ]

    meta_path = out_dir / f"{sa}_run_metadata.txt"
    with open(meta_path, "w") as f:
        f.write("\n".join(meta_lines) + "\n")
    _log(f"  → wrote {meta_path.name}")

    # ── 8. Per-polygon summary table ────────────────────────────────────────
    _log("8 · Computing per-polygon summary (implied scale factors) …")

    # Re-read the original (un-buffered) shapefile for zonal stats
    gdf_orig = gpd.read_file(cfg.TREATMENT_SHP)
    etg_crs_obj = etg_prof["crs"]
    if gdf_orig.crs is not None and not gdf_orig.crs.equals(etg_crs_obj):
        gdf_orig = gdf_orig.to_crs(etg_crs_obj)
    for col in (cfg.ATTR_SCALE, cfg.ATTR_REPLACE):
        if col in gdf_orig.columns:
            gdf_orig[col] = gdf_orig[col].fillna(0).astype(float)

    # Determine an ID column for the output
    id_col = None
    for candidate in ("DRI_ID", "UniqueID", "ET_unit", "FID"):
        if candidate in gdf_orig.columns:
            id_col = candidate
            break
    if id_col is None:
        gdf_orig["_row_id"] = range(len(gdf_orig))
        id_col = "_row_id"

    rows = []
    for i, row in gdf_orig.iterrows():
        geom = row.geometry
        if geom is None or not geom.is_valid:
            continue

        # Is this a treatment polygon?
        is_treated = False
        for col in (cfg.ATTR_SCALE, cfg.ATTR_REPLACE):
            if col in gdf_orig.columns and row.get(col, 0) > 0:
                is_treated = True
                break

        # Rasterize this single polygon to a mini-mask
        mini_mask = rasterize(
            [(geom, 1)],
            out_shape=grid_shape,
            transform=grid_transform,
            fill=0,
            dtype=np.uint8,
        )
        px = (mini_mask == 1)
        n_px = int(px.sum())
        if n_px == 0:
            continue

        base_vals  = baseline[px & np.isfinite(baseline)]
        final_vals = etg_final[px & np.isfinite(etg_final)]
        input_vals = etg[px & np.isfinite(etg)]

        mean_base  = float(np.mean(base_vals))   if len(base_vals) > 0   else np.nan
        mean_final = float(np.mean(final_vals))  if len(final_vals) > 0  else np.nan
        mean_input = float(np.mean(input_vals))  if len(input_vals) > 0  else np.nan

        # Determine the effective adjustment factor for this polygon
        poly_adj = basin_adjust  # default
        if is_treated and has_per_poly_adj and attr_adjust in gdf_orig.columns:
            poly_adj_val = float(row.get(attr_adjust, 0) or 0)
            if poly_adj_val > 0:
                poly_adj = poly_adj_val

        rec = {
            "polygon_id": row[id_col],
            "n_pixels": n_px,
            "treatment": "replaced" if is_treated else "none",
            "adj_factor": round(poly_adj, 4) if is_treated else "",
            "mean_input_ETg": round(mean_input, 4),
            "mean_baseline_ETg": round(mean_base, 4),
            "mean_final_ETg": round(mean_final, 4),
        }

        # For Descrip / crop info if available
        for extra in ("Descrip", "crop_major", "ET_unit"):
            if extra in gdf_orig.columns:
                rec[extra] = row[extra]

        rows.append(rec)

    csv_path = out_dir / f"{sa}_polygon_summary.csv"
    if rows:
        keys = rows[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        _log(f"  → wrote {csv_path.name}  ({len(rows)} polygons)")
    else:
        _log("    WARNING: no polygons produced summary rows")

    # ── Summary stats ────────────────────────────────────────────────────────
    _log("── Summary ─────────────────────────────────────────")

    def _stats(arr, label):
        a = arr[np.isfinite(arr)]
        if len(a) == 0:
            _log(f"  {label:40s}  (no valid pixels)")
            return
        _log(f"  {label:40s}  n={len(a):>8,}  "
             f"mean={np.mean(a):.3f}  med={np.median(a):.3f}  std={np.std(a):.3f}")

    treat_bool = treatment_zone.astype(bool)
    _stats(etg[~treat_bool],             "Outside treatment (training)")
    _stats(etg_raw[treat_bool],          "Treatment zones – raw ETg")
    _stats(baseline[treat_bool],         "Treatment zones – model baseline")
    _stats(etg_final[treat_bool],        "Treatment zones – final")

    # Total volume change vs raw input
    raw_sum = np.nansum(etg_raw[np.isfinite(etg_raw)])
    final_sum = np.nansum(etg_final[np.isfinite(etg_final)])
    pct = 100 * (final_sum - raw_sum) / raw_sum if raw_sum > 0 else 0
    _log(f"  Total ETg volume change vs raw: {pct:+.2f}%")

    _log(f"  Elapsed: {time.time() - t0:.1f} s")
    _log("Done.  Outputs in:  " + str(out_dir.resolve()))


if __name__ == "__main__":
    main()
