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

Configuration lives in per-basin TOML files under basins/<key>/config.toml
(loaded via basin_config.py).

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
# Config is loaded per-basin from basins/<key>/config.toml via basin_config.py.
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


def _spatially_weighted_bps_mean(
    etg_train: np.ndarray,
    bps: np.ndarray,
    valid_mask: np.ndarray,
    unique_bps: np.ndarray,
    bps_mean_map: dict,
    radius_px: int = 33,
) -> np.ndarray:
    """
    Compute a spatially weighted per-BpS mean ETg layer.

    For each BpS class, builds a raster of known ETg values at training
    pixel locations, applies a Gaussian spatial smooth (sigma = radius_px/3
    so that ~99% of weight falls within ``radius_px``), then divides by a
    similarly smoothed count to produce a local weighted mean.

    Where the local window contains no training pixels of the same class
    (e.g. a class that only exists inside the treatment zone), the basin-
    wide class mean is used as the fallback.

    Parameters
    ----------
    etg_train : 2-D array
        ETg values at training pixel locations; NaN elsewhere.
    bps : 2-D array (int)
        BpS class codes for every pixel.
    valid_mask : 2-D bool array
        True where training-eligible pixels exist.
    unique_bps : 1-D array
        Unique BpS codes present in the training data.
    bps_mean_map : dict
        {bps_code: basin-wide mean ETg} — fallback for sparse areas.
    radius_px : int
        Approximate radius (in pixels) of the spatial window.  The
        Gaussian sigma is set to radius_px / 3 so the window captures
        ~99% of the weight.  At 30 m pixels, the default 33 px ≈ 1 km.

    Returns
    -------
    2-D float32 array
        Spatially varying per-BpS mean ETg, same shape as ``bps``.
    """
    from scipy.ndimage import gaussian_filter

    sigma = max(radius_px / 3.0, 1.0)
    result = np.zeros(bps.shape, dtype=np.float32)
    global_mean = float(np.nanmean(etg_train[valid_mask]))

    for b in unique_bps:
        class_mask = (bps == b)
        train_mask = class_mask & valid_mask & np.isfinite(etg_train)

        n_train = int(train_mask.sum())
        if n_train == 0:
            # No training pixels for this class — use basin-wide mean
            result[class_mask] = bps_mean_map.get(b, global_mean)
            continue

        # Build value and count layers for this class
        vals = np.zeros(bps.shape, dtype=np.float64)
        counts = np.zeros(bps.shape, dtype=np.float64)
        vals[train_mask] = etg_train[train_mask]
        counts[train_mask] = 1.0

        # Gaussian smooth both — division gives a weighted local mean
        vals_smooth = gaussian_filter(vals, sigma=sigma, mode="constant")
        counts_smooth = gaussian_filter(counts, sigma=sigma, mode="constant")

        # Where counts_smooth is near zero, no nearby training pixels exist
        has_local = counts_smooth > 1e-10
        local_mean = np.where(has_local,
                              vals_smooth / counts_smooth,
                              bps_mean_map.get(b, global_mean))

        result[class_mask] = local_mean[class_mask].astype(np.float32)

    return result


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
    Load the per-basin TOML config via basin_config.py.  Errors out if the
    basin has no config.toml (run prep_basin.py or prep_custom_basin.py first).
    """
    global cfg
    basins_dir = _here / "basins"
    toml_path = basins_dir / study_area / "config.toml"
    if not toml_path.exists():
        import basin_config as _bc
        available = _bc.available_areas()
        sys.exit(
            f"ERROR: no config.toml found for '{study_area}' at {toml_path}.\n"
            f"  Run prep_basin.py (or prep_custom_basin.py) first.\n"
            f"  Available basins: {', '.join(available) if available else '(none)'}"
        )
    import basin_config as _cfg
    _cfg.load_basin(study_area)
    cfg = _cfg


def main(study_area: str | None = None) -> None:
    # ── Resolve study area ──────────────────────────────────────────────────
    if study_area is None:
        # Fall through to CLI parser for argv-style invocation.
        _cli()
        return
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
    # Start from the input ETg raster and NaN out every treatment-zone pixel
    # so they're excluded from training (the model will predict baseline
    # values for them later).  ``etg_raw`` is the working copy with treatment
    # pixels removed; ``etg`` keeps the original values for reference /
    # diagnostics.
    _log("   2d · Building training ETg (NaN-ing all treatment pixels) …")
    etg_raw = etg.copy()
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
    # prep_statewide.py via whitebox-tools, encoded in meters above the nearest
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

    # Optional: Relative Elevation Model (REM) — derived from DEM via a low-
    # percentile moving window (prep_basin --derive-rem).  Opt-in alternative
    # to HAND for closed Great Basin sub-basins without integrated drainage.
    rem = None
    use_rem = getattr(cfg, "USE_REM", False)
    if not use_rem:
        _log("   REM disabled in config (use_rem = false) — skipping")
    elif getattr(cfg, "REM_TIF", None) is not None:
        _log("   Matching REM to ETg grid …")
        rem_candidate = _match_raster(
            cfg.REM_TIF, etg_prof, Resampling.bilinear,
            out_dir / "REM_matched.tif",
        )
        n_rem_valid = int(np.isfinite(rem_candidate).sum())
        if n_rem_valid > 0:
            rem = rem_candidate
            _log(f"    REM valid pixels: {n_rem_valid:,}  "
                 f"range: {np.nanmin(rem):.1f} – {np.nanmax(rem):.1f} m")
        else:
            _log("  ⚠ REM raster has 0 valid pixels after reprojection — "
                 "continuing WITHOUT REM.")
    else:
        _log("   REM raster not configured — skipping")

    # Optional: Soil covariates from gSSURGO — Available Water Capacity (AWC)
    # in the top ~1 m and depth to restrictive layer.  Both proxy for the
    # sub-surface moisture-holding capacity and rooting depth that partly
    # control natural ET, independently of terrain position.  Loaded and
    # gated in parallel: each is included only if its raster is present AND
    # reprojects to at least one valid pixel on the ETg grid.
    def _load_optional_covariate(tif_path, label, match_path):
        """Reproject onto ETg grid; return array or None if unusable."""
        if tif_path is None:
            _log(f"   {label} raster not configured — skipping")
            return None
        _log(f"   Matching {label} to ETg grid …")
        arr = _match_raster(
            tif_path, etg_prof, Resampling.bilinear, match_path,
        )
        n_valid_px = int(np.isfinite(arr).sum())
        if n_valid_px == 0:
            _log(f"  ⚠ {label} raster has 0 valid pixels after reprojection "
                 "— continuing WITHOUT it.")
            return None
        _log(f"    {label} valid pixels: {n_valid_px:,}  "
             f"range: {np.nanmin(arr):.2f} – {np.nanmax(arr):.2f}")
        return arr

    awc = None
    soil_depth = None
    use_soil = getattr(cfg, "USE_SOIL", True)
    if not use_soil:
        _log("   Soil (AWC + depth) disabled in config (use_soil = false) — skipping")
    else:
        awc = _load_optional_covariate(
            getattr(cfg, "AWC_TIF", None), "AWC",
            out_dir / "AWC_matched.tif",
        )
        soil_depth = _load_optional_covariate(
            getattr(cfg, "SOIL_DEPTH_TIF", None), "SoilDepth",
            out_dir / "SoilDepth_matched.tif",
        )

    # ── 4. Derive slope ──────────────────────────────────────────────────────
    _log("4 · Computing slope from DEM …")
    cellsize = abs(etg_prof["transform"].a)
    slope = _slope_from_dem(dem, cellsize)
    _write_raster(slope, etg_prof, out_dir / "slope_matched.tif")

    # ── 4b. Basin boundary mask (constrain training to within the basin) ────
    # Priority:  (1) boundary_shp from config.toml  →  (2) NWI shapefile.
    # Custom boundary_shp lets non-NWI study areas (e.g. Sierra Valley CA)
    # define their own training mask without needing the NWI shapefile at all.
    basin_mask = None
    boundary_shp = getattr(cfg, "BOUNDARY_SHP", None)
    boundary_configured = getattr(cfg, "BOUNDARY_SHP_CONFIGURED", False)
    boundary_missing = getattr(cfg, "BOUNDARY_SHP_MISSING", False)
    nwi_path = _here / "NWI_Investigations_EPSG_32611.shp"

    # If the user configured a boundary path but the file is missing, warn
    # loudly — that's almost certainly a broken config, not intent.
    if boundary_configured and boundary_missing:
        _log(f"  4b · WARNING: boundary_shp configured but file missing — "
             f"cannot build training mask; using all valid pixels")

    if boundary_shp is not None and Path(boundary_shp).exists():
        # ── Custom boundary from config.toml ──────────────────────────
        _log(f"  4b · Rasterizing basin boundary (custom: {Path(boundary_shp).name}) …")
        gdf_bnd = gpd.read_file(boundary_shp)
        if len(gdf_bnd) == 0:
            _log("    WARNING: boundary shapefile is empty — using all valid pixels")
        else:
            from shapely.ops import unary_union
            bnd_geom = unary_union(gdf_bnd.geometry)
            if gdf_bnd.crs is not None and not gdf_bnd.crs.equals(etg_crs):
                bnd_series = gpd.GeoSeries([bnd_geom], crs=gdf_bnd.crs)
                bnd_geom = bnd_series.to_crs(etg_crs).iloc[0]
            basin_mask = rasterize(
                [(bnd_geom, 1)],
                out_shape=grid_shape,
                transform=grid_transform,
                fill=0,
                dtype=np.uint8,
            ).astype(bool)
            n_basin_px = int(basin_mask.sum())
            _log(f"    basin boundary pixels: {n_basin_px:,}")

    elif nwi_path.exists():
        # ── Fall back to NWI shapefile ────────────────────────────────
        # Custom (non-NWI) basins won't match the NWI shapefile by design.
        # A basin_key without an NWI-style numeric prefix (e.g. "053_Pine…")
        # is almost certainly a custom basin, so skip the NWI lookup
        # quietly rather than emitting a scary "not found" warning.
        basin_key = cfg.STUDY_AREA_NAME
        looks_like_nwi = (
            "_" in basin_key
            and basin_key.split("_", 1)[0].isdigit()
        )
        if not looks_like_nwi:
            _log(f"  4b · Custom basin ({basin_key}) — no boundary_shp "
                 f"configured and basin_key is not NWI-formatted; "
                 f"training will use all valid pixels")
        else:
            _log("  4b · Rasterizing basin boundary (NWI) for training mask …")
            gdf_nwi = gpd.read_file(nwi_path)
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
                _log(f"    WARNING: basin key '{basin_key}' is NWI-formatted "
                     f"but not found in NWI shapefile — training will use "
                     f"all valid pixels")
    else:
        _log("    No basin boundary found (no boundary_shp in config, "
             "no NWI shapefile) — training will use all valid pixels")

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
    if rem is not None:
        valid &= np.isfinite(rem)
    if awc is not None:
        valid &= np.isfinite(awc)
    if soil_depth is not None:
        valid &= np.isfinite(soil_depth)
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
    rem_flat   = rem.ravel()[idx] if rem is not None else None
    awc_flat   = awc.ravel()[idx] if awc is not None else None
    soil_depth_flat = soil_depth.ravel()[idx] if soil_depth is not None else None

    # ── 5a. Per-BpS-class mean ETg ───────────────────────────────────────────
    _log("   5a · Computing per-BpS mean ETg …")
    unique_bps = np.unique(bps_flat)
    bps_mean_map = {}
    for b in unique_bps:
        bps_mean_map[b] = float(np.nanmean(y_train[bps_flat == b]))
    _log(f"    BpS classes in training data: {len(unique_bps)}")

    # Load BpS class names for readable logging and metadata
    try:
        from bps_utils import load_bps_lookup, bps_name as _bps_name
        _bps_lut = load_bps_lookup()
    except Exception:
        _bps_lut = {}
        def _bps_name(code, lut=None):
            return f"BpS {code}"

    # Track within-class std alongside the mean so we can see whether each
    # BpS class carries any residual signal at all.
    bps_std_map: dict[int, float] = {}
    bps_count_map: dict[int, int] = {}
    for b in unique_bps:
        vals = y_train[bps_flat == b]
        bps_std_map[b] = float(np.nanstd(vals)) if vals.size else 0.0
        bps_count_map[b] = int(vals.size)

    for b in sorted(bps_mean_map, key=lambda x: -bps_mean_map[x]):
        _log(f"      {_bps_name(b, _bps_lut):50s}  "
             f"code={b:<6d}  mean={bps_mean_map[b]:.4f} ft  "
             f"std={bps_std_map[b]:.4f}  "
             f"n={bps_count_map[b]:,}")

    bps_mean_train = np.array([bps_mean_map.get(b, 0.0) for b in bps_flat],
                              dtype=np.float32)
    residual_train = y_train - bps_mean_train

    # ── Residual-target signal diagnostic ─────────────────────────────────
    # Compare the std of the residual target to the std of the raw target.
    # A small ratio means the BpS class mean already explains most of the
    # variance, so there's little left for the terrain model to learn.
    # Across a Huntington-style dataset (piecewise-constant per polygon),
    # expect ratios well below 0.5; a negative CV R² in that regime is
    # "no signal to fit" rather than "model failed to fit".
    y_std        = float(np.nanstd(y_train))        if y_train.size else 0.0
    resid_std    = float(np.nanstd(residual_train)) if residual_train.size else 0.0
    resid_ratio  = (resid_std / y_std) if y_std > 0 else float("nan")
    _log(f"    residual-target diagnostic: "
         f"std(y)={y_std:.4f}  std(residual)={resid_std:.4f}  "
         f"ratio={resid_ratio:.3f}   (ratio ≪ 1 ⇒ BpS mean already "
         f"explains most variance)")

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
    if rem_flat is not None:
        feat_columns.append(rem_flat)
        feature_names.append("rem")
    if awc_flat is not None:
        feat_columns.append(awc_flat)
        feature_names.append("awc")
    if soil_depth_flat is not None:
        feat_columns.append(soil_depth_flat)
        feature_names.append("soil_depth")
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
    # WORSE than the BpS class means alone.  In that case we fall back to
    # a spatially weighted per-BpS mean (local window smoothing) rather
    # than a flat basin-wide class average.
    use_residual = cv_mean >= 0.0
    use_spatial_fallback = False
    if not use_residual:
        spatial_radius = getattr(cfg, "SPATIAL_FALLBACK_RADIUS_PX", 33)
        if spatial_radius > 0:
            use_spatial_fallback = True
            _log(f"    ⚠ CV R² is negative ({cv_mean:.4f}) — the residual model "
                 f"is hurting predictions.  Falling back to spatially weighted "
                 f"BpS means (radius = {spatial_radius} px ≈ "
                 f"{spatial_radius * abs(etg_prof['transform'].a):.0f} m).")
        else:
            _log(f"    ⚠ CV R² is negative ({cv_mean:.4f}) — the residual model "
                 f"is hurting predictions.  Falling back to flat BpS-mean baseline "
                 f"(spatial fallback disabled: radius = 0).")

    # ── 6. Predict baseline ETg (for replacement zones) ──────────────────────
    _log("6 · Predicting baseline ETg …")

    # Build the BpS mean layer — either spatially weighted or flat.
    global_mean = float(np.nanmean(y_train))

    if use_spatial_fallback:
        # Spatially weighted: for each BpS class, nearby training pixels
        # contribute more than distant ones via Gaussian smoothing.
        _log("   6a · Computing spatially weighted BpS means …")
        etg_train_raster = np.full_like(etg, np.nan)
        etg_train_raster.ravel()[idx] = y_train
        bps_mean_full = _spatially_weighted_bps_mean(
            etg_train=etg_train_raster,
            bps=bps,
            valid_mask=valid,
            unique_bps=unique_bps,
            bps_mean_map=bps_mean_map,
            radius_px=spatial_radius,
        )
        # Handle unseen BpS classes (not in training data)
        unseen = ~np.isin(bps, list(bps_mean_map.keys())) & (bps > 0)
        bps_mean_full[unseen] = global_mean
        _log(f"    spatial BpS mean range: "
             f"{float(np.nanmin(bps_mean_full[bps_mean_full > 0])):.4f} – "
             f"{float(np.nanmax(bps_mean_full)):.4f} ft")
    else:
        # Flat: single basin-wide mean per BpS class
        bps_mean_full = np.zeros_like(etg)
        for b, m in bps_mean_map.items():
            bps_mean_full[bps == b] = m
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
        if rem is not None:
            pred_mask &= np.isfinite(rem)
        if awc is not None:
            pred_mask &= np.isfinite(awc)
        if soil_depth is not None:
            pred_mask &= np.isfinite(soil_depth)
        pred_idx = np.where(pred_mask.ravel())[0]

        # Pre-flatten the covariate rasters once
        dem_r   = dem.ravel()
        slope_r = slope.ravel()
        wtd_r   = wtd.ravel()  if wtd  is not None else None
        hand_r  = hand.ravel() if hand is not None else None
        rem_r   = rem.ravel()  if rem  is not None else None
        awc_r   = awc.ravel()  if awc  is not None else None
        sdp_r   = soil_depth.ravel() if soil_depth is not None else None

        CHUNK = 500_000
        for start in range(0, len(pred_idx), CHUNK):
            chunk_idx = pred_idx[start:start + CHUNK]
            cols = [dem_r[chunk_idx], slope_r[chunk_idx]]
            if wtd_r is not None:
                cols.append(wtd_r[chunk_idx])
            if hand_r is not None:
                cols.append(hand_r[chunk_idx])
            if rem_r is not None:
                cols.append(rem_r[chunk_idx])
            if awc_r is not None:
                cols.append(awc_r[chunk_idx])
            if sdp_r is not None:
                cols.append(sdp_r[chunk_idx])
            X_chunk = pd.DataFrame(np.column_stack(cols), columns=feature_names)
            residual_pred.ravel()[chunk_idx] = model.predict(X_chunk).astype(np.float32)
    else:
        # No residual model — baseline comes entirely from BpS means
        # (either spatially weighted or flat depending on above)
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
    # All treatment-zone pixels in etg_raw are NaN.  We start from etg_raw
    # (= etg outside treatment, NaN inside) and fill treatment pixels with
    # the appropriate method.
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
        #
        # Irrigation-pull guard: when a feather-band pixel sits on untreated
        # irrigated ag (raw ETg > baseline, e.g. an adjacent parcel the
        # treatment polygon didn't capture), the naive weighted mean pulls
        # the treatment edge UP toward the irrigation value, creating a
        # visible bright ring around the polygon.  Clip raw at the baseline
        # before blending so the feather can only pull edges *down* toward
        # natural values, never *up* toward irrigation signal.
        etg_final = etg_treated.copy()
        fb = feather_band & np.isfinite(etg_raw) & np.isfinite(adjusted_baseline)
        raw_for_blend = np.minimum(etg_raw[fb], adjusted_baseline[fb])
        blended = (
            blend_weight[fb] * adjusted_baseline[fb]
            + (1.0 - blend_weight[fb]) * raw_for_blend
        )
        # Redundant downward-only safety (guards against any numerical drift)
        blended = np.minimum(blended, etg_raw[fb])
        etg_final[fb] = blended

        # Diagnostic: how many feather-band pixels had their raw clipped
        # (i.e. were irrigation-pull candidates)?
        n_irrig_clipped = int((etg_raw[fb] > adjusted_baseline[fb]).sum())
        if n_irrig_clipped > 0:
            _log(f"    irrigation-pull guard: {n_irrig_clipped:,} feather "
                 f"pixels had raw > baseline and were clipped in the blend")

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
    _log("   7d · Computing per-pixel percent change (input → final) …")
    # Reference = the original INPUT ETg raster (etg), NOT etg_raw.
    # etg_raw has every treatment pixel NaN-masked for training (see §2d),
    # which would make pct_change NaN across entire treatment interiors —
    # only the feathering ring would show values.  `etg` retains the input
    # values inside treatment zones, so pct_change shows the true magnitude
    # of the fill's effect everywhere it changed the raster.
    pct_change = np.full_like(etg, np.nan, dtype=np.float32)
    denom_valid = np.isfinite(etg) & (etg > 0) & np.isfinite(etg_final)
    pct_change[denom_valid] = (
        100.0 * (etg_final[denom_valid] - etg[denom_valid]) / etg[denom_valid]
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
        f"dem_tif          = {cfg.DEM_TIF}",
        f"bps_tif          = {cfg.BPS_TIF}",
        f"wtd_tif          = {getattr(cfg, 'WTD_TIF', None)}",
        f"hand_tif         = {getattr(cfg, 'HAND_TIF', None)}",
        f"rem_tif          = {getattr(cfg, 'REM_TIF', None)}",
        f"awc_tif          = {getattr(cfg, 'AWC_TIF', None)}",
        f"soil_depth_tif   = {getattr(cfg, 'SOIL_DEPTH_TIF', None)}",
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
        f"use_rem          = {getattr(cfg, 'USE_REM', False)}",
        f"use_soil         = {getattr(cfg, 'USE_SOIL', True)}",
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
    if use_residual:
        residual_status = "yes"
    elif use_spatial_fallback:
        residual_status = (f"no (CV R² < 0, spatially weighted BpS means, "
                           f"radius={spatial_radius}px)")
    else:
        residual_status = "no (CV R² < 0, flat BpS-mean only)"
    meta_lines.append(f"use_residual_model = {residual_status}")
    meta_lines.append(f"spatial_fallback_radius_px = "
                      f"{getattr(cfg, 'SPATIAL_FALLBACK_RADIUS_PX', 33)}")

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
        f"y_std                 = {y_std:.4f}   # std of raw ETg target",
        f"residual_std          = {resid_std:.4f}   # std after subtracting BpS class mean",
        f"residual_signal_ratio = {resid_ratio:.4f}   # residual_std / y_std "
        f"(ratio ≪ 1 ⇒ no signal for residual model)",
    ]

    # Feature importance
    meta_lines.append("")
    meta_lines.append("[feature_importance]")
    for name, val in zip(feature_names, imp):
        meta_lines.append(f"{name:20s} = {val:.4f}")

    # BpS class breakdown
    meta_lines.append("")
    meta_lines.append("[bps_class_means]")
    meta_lines.append("# code  mean_ETg_ft  std_ETg_ft  n_train_pixels  class_name")
    for b in sorted(bps_mean_map, key=lambda x: -bps_mean_map[x]):
        meta_lines.append(
            f"{b:<8d} = {bps_mean_map[b]:.4f}  "
            f"std={bps_std_map[b]:.4f}  "
            f"n={bps_count_map[b]:<8,d}  {_bps_name(b, _bps_lut)}"
        )

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
    # Report stats against the ORIGINAL input ETg raster (etg), not etg_raw.
    # etg_raw has treatment pixels NaN-masked for training (see §2d), so
    # using it here would print "(no valid pixels)" in treatment zones and
    # also corrupt the volume-change denominator (treatment pixels would
    # count as zero in the baseline).  etg preserves the original values.
    _stats(etg[treat_bool],              "Treatment zones – original input")
    _stats(baseline[treat_bool],         "Treatment zones – model baseline")
    _stats(etg_final[treat_bool],        "Treatment zones – final")

    # Total volume change: compare final raster against the ORIGINAL input
    # ETg raster over the same pixel set (pixels valid in both rasters).
    # This gives the true "how much did we change the raw ETg?" number.
    both_valid = np.isfinite(etg) & np.isfinite(etg_final)
    orig_sum  = float(np.sum(etg[both_valid]))
    final_sum = float(np.sum(etg_final[both_valid]))
    pct = 100 * (final_sum - orig_sum) / orig_sum if orig_sum > 0 else 0
    _log(f"  Total ETg volume change vs original input: {pct:+.2f}%  "
         f"(over {int(both_valid.sum()):,} pixels valid in both rasters)")
    # Also report the treatment-zone-only change, which is the actionable
    # number for buy-back accounting.
    tz = both_valid & treat_bool
    if tz.any():
        orig_tz  = float(np.sum(etg[tz]))
        final_tz = float(np.sum(etg_final[tz]))
        pct_tz = 100 * (final_tz - orig_tz) / orig_tz if orig_tz > 0 else 0
        _log(f"  Treatment-zone ETg volume change:       {pct_tz:+.2f}%  "
             f"(over {int(tz.sum()):,} treatment pixels)")

    _log(f"  Elapsed: {time.time() - t0:.1f} s")
    _log("Done.  Outputs in:  " + str(out_dir.resolve()))


def _cli() -> None:
    """Argparse-driven entry point. Supports single-basin or batch runs."""
    import argparse
    import basin_config as _bc

    ap = argparse.ArgumentParser(
        prog="etg_baseline_fill.py",
        description="Fill treatment-zone ETg with a data-driven natural baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python etg_baseline_fill.py 053_PineValley\n"
            "  python etg_baseline_fill.py --all\n"
            "  python etg_baseline_fill.py --all --skip 053_PineValley 131_BuffaloValley\n"
            "  python etg_baseline_fill.py --only 042_MarysRiverArea 073A_LovelockValley\n"
            "  python etg_baseline_fill.py --list\n"
        ),
    )
    ap.add_argument("study_area", nargs="?",
                    help="Basin key to process (e.g. 053_PineValley).")
    ap.add_argument("--all", action="store_true",
                    help="Process every basin that has a config.toml.")
    ap.add_argument("--only", nargs="+", metavar="KEY",
                    help="Process only these basin keys (mutually exclusive with --all).")
    ap.add_argument("--skip", nargs="+", metavar="KEY", default=[],
                    help="When used with --all, skip these basin keys.")
    ap.add_argument("--list", action="store_true",
                    help="List available basins and exit.")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="Abort the batch on the first failure (default: continue).")
    args = ap.parse_args()

    available = _bc.available_areas()

    if args.list:
        if not available:
            print("(no basins have a config.toml yet)")
        else:
            print(f"{len(available)} basin(s) with config.toml:")
            for k in available:
                print(f"  {k}")
        return

    # Resolve which basins to run
    if args.all and args.only:
        ap.error("--all and --only are mutually exclusive")
    if args.all:
        targets = [k for k in available if k not in set(args.skip)]
        if not targets:
            sys.exit("No basins to run (after applying --skip).")
    elif args.only:
        missing = [k for k in args.only if k not in available]
        if missing:
            sys.exit(
                f"Basin(s) not found (no config.toml): {', '.join(missing)}\n"
                f"  Available: {', '.join(available) if available else '(none)'}"
            )
        targets = list(args.only)
    elif args.study_area:
        if args.study_area not in available:
            sys.exit(
                f"Basin '{args.study_area}' not found (no config.toml).\n"
                f"  Available: {', '.join(available) if available else '(none)'}"
            )
        targets = [args.study_area]
    else:
        ap.print_help()
        sys.exit(
            f"\nERROR: supply <study_area>, --all, --only, or --list.\n"
            f"Available basins: {', '.join(available) if available else '(none)'}"
        )

    # Single-basin: run directly so exceptions surface the usual way.
    if len(targets) == 1:
        main(targets[0])
        return

    # Batch run
    print(f"\n═══ BATCH: {len(targets)} basin(s) ═══")
    for i, k in enumerate(targets, 1):
        print(f"  [{i:3d}/{len(targets)}] {k}")
    print()

    batch_t0 = time.time()
    ok, failed = [], []
    for i, key in enumerate(targets, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(targets)}]  {key}")
        print("=" * 60)
        try:
            main(key)
            ok.append(key)
        except SystemExit as e:
            # _load_cfg and similar use sys.exit on missing data; treat as failure.
            msg = str(e) if e.code not in (None, 0) else "SystemExit"
            failed.append((key, msg))
            print(f"\n⚠ {key} aborted: {msg}")
            if args.stop_on_error:
                break
        except Exception as e:  # pragma: no cover — defensive
            failed.append((key, f"{type(e).__name__}: {e}"))
            print(f"\n⚠ {key} failed: {type(e).__name__}: {e}")
            if args.stop_on_error:
                raise

    # Summary
    elapsed = time.time() - batch_t0
    print(f"\n{'=' * 60}")
    print(f"Batch complete in {elapsed/60:.1f} min")
    print(f"  succeeded: {len(ok)}")
    print(f"  failed:    {len(failed)}")
    if failed:
        print("\nFailures:")
        for k, msg in failed:
            print(f"  • {k}: {msg}")
    print("=" * 60)

    if failed and not ok:
        sys.exit(1)


if __name__ == "__main__":
    _cli()
