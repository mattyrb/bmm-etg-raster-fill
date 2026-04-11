"""
config.py  –  All user-tunable parameters for the ETg baseline-fill workflow.

Study areas are defined in the STUDY_AREAS dictionary.  Each entry maps a
short name (used on the command line and in output folder/file names) to the
input file paths and any area-specific overrides for that basin.

The active study area is selected at runtime:
    python etg_baseline_fill.py SierraValley
    python etg_baseline_fill.py PineValley

Model parameters, feathering, thresholds, etc. are shared across all areas
unless overridden per-area.
"""

from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent

# ── Study areas ─────────────────────────────────────────────────────────────
# Each entry MUST contain at minimum:
#   etg_tif, dem_tif, bps_tif, treatment_shp
# Optional keys (set to None or omit to skip):
#   etg_raw_tif, wtd_tif, crs_overrides
# Any shared parameter (model, feathering, etc.) can be overridden per-area
# by including it here; otherwise the global default below is used.

STUDY_AREAS = {

    "SierraValley": {
        "data_dir":      PROJECT_DIR / "SierraValley_Data",
        "etg_tif":       "HASV1_SierraValley_ETunits_MGv4_no_surface_etg_adj_long_term_median_ft_burned_replacement_rates.tif",
        "etg_raw_tif":   "HASV1_SierraValley_phreats_w_ag_raw_etg_adj_long_term_median_ft_1984_2025.tif",
        "dem_tif":       "DEM_SierraValley.tif",
        "bps_tif":       "LF2020_BPS_SierraValley.tif",
        "wtd_tif":       "wtd_mean_estimate_RF_additional_inputs_dummy_drop0LP_1s_SierraValley.tif",
        "treatment_shp": "SierraValley_ETunits_v5_SWscale_down.shp",
        "crs_overrides": {
            "LF2020_BPS_SierraValley": "EPSG:5070",
            "wtd_mean_estimate_RF_additional_inputs_dummy_drop0LP_1s_SierraValley":
                "+proj=lcc +lat_1=30 +lat_2=60 +lon_0=-97.0 +lat_0=40.0000076294444 "
                "+a=6370000.0 +b=6370000.0 +units=m +no_defs",
        },
    },

    "PineValley": {
        "data_dir":      PROJECT_DIR / "PineValley_Data",
        "etg_tif":       "HA053_NV_phreats_MASTER_v11_PineValley_053_w_ag_etg_adj_long_term_median_ft_1984_2025.tif",
        "etg_raw_tif":   "HA053_NV_phreats_MASTER_v11_PineValley_053_w_ag_etg_adj_long_term_median_ft_1984_2025.tif",
        "dem_tif":       "SRTMGL1.tiff",
        "bps_tif":       "LF2020_BPS_CONUS_PineValley.tif",
        "wtd_tif":       "wtd_mean_estimate_RF_additional_inputs_dummy_drop0LP_1s_CONUS2_m_v_20240813_PineValley.tif",
        "treatment_shp": "NV_phreats_MASTER_v11_PineValley_053_w_ag.shp",
        "crs_overrides": {
            "LF2020_BPS_CONUS_PineValley": "EPSG:5070",
            "wtd_mean_estimate_RF_additional_inputs_dummy_drop0LP_1s_CONUS2_m_v_20240813_PineValley":
                "+proj=lcc +lat_1=30 +lat_2=60 +lon_0=-97.0 +lat_0=40.0000076294444 "
                "+a=6370000.0 +b=6370000.0 +units=m +no_defs",
        },
    },
}

# ── Active study area (set by load_study_area()) ────────────────────────────
# These module-level attributes are populated at runtime so that the rest of
# the codebase can continue to reference cfg.ETG_TIF, cfg.OUT_DIR, etc.
STUDY_AREA_NAME = None
DATA_DIR        = None
ETG_TIF         = None
ETG_RAW_TIF     = None
DEM_TIF         = None
BPS_TIF         = None
WTD_TIF         = None
HAND_TIF        = None
TREATMENT_SHP   = None
OUT_DIR         = None
CRS_OVERRIDES   = {}


def load_study_area(name: str) -> None:
    """
    Activate a study area by name.  Populates module-level path attributes
    so downstream code (etg_baseline_fill, diagnostics, etunit_summary) can
    use ``cfg.ETG_TIF``, ``cfg.OUT_DIR``, etc. without changes.

    Parameters
    ----------
    name : str
        Key in STUDY_AREAS (e.g. "SierraValley", "PineValley").
    """
    import sys as _sys

    if name not in STUDY_AREAS:
        available = ", ".join(sorted(STUDY_AREAS.keys()))
        _sys.exit(f"ERROR: unknown study area '{name}'.  Available: {available}")

    area = STUDY_AREAS[name]
    d = area["data_dir"]

    # Resolve each path; None stays None
    def _resolve(key):
        val = area.get(key)
        if val is None:
            return None
        return d / val

    g = globals()
    g["STUDY_AREA_NAME"] = name
    g["DATA_DIR"]        = d
    g["ETG_TIF"]         = _resolve("etg_tif")
    g["ETG_RAW_TIF"]     = _resolve("etg_raw_tif")
    g["DEM_TIF"]         = _resolve("dem_tif")
    g["BPS_TIF"]         = _resolve("bps_tif")
    g["WTD_TIF"]         = _resolve("wtd_tif")
    g["HAND_TIF"]        = _resolve("hand_tif")
    g["TREATMENT_SHP"]   = _resolve("treatment_shp")
    g["CRS_OVERRIDES"]   = area.get("crs_overrides", {})

    # Output directory: output/<StudyAreaName>/
    g["OUT_DIR"] = PROJECT_DIR / "output" / name

    # Validate that required inputs exist
    for key in ("ETG_TIF", "DEM_TIF", "BPS_TIF", "TREATMENT_SHP"):
        p = g[key]
        if p is None:
            _sys.exit(f"ERROR: '{key}' is not configured for study area '{name}'.\n"
                      f"  Edit the '{name}' entry in config.py.")
        if not p.exists():
            _sys.exit(f"ERROR: {key} not found: {p}")


def available_areas() -> list[str]:
    """Return sorted list of configured study area names."""
    return sorted(STUDY_AREAS.keys())


# ── Shapefile attribute names ───────────────────────────────────────────────
# A polygon is treated (baseline-replaced) if either attribute is > 0.
ATTR_SCALE    = "scale_fctr"
ATTR_REPLACE  = "rplc_rt"

# ── Expert adjustment ───────────────────────────────────────────────────────
# Scale the modeled baseline up or down.  1.0 = no change.  0.8 = reduce 20%.
# Individual polygons can override this via an "adj_fctr" column in the
# treatment shapefile.  If the column exists and a polygon's value is > 0,
# that per-polygon value is used instead of this basin-wide default.
BASELINE_ADJUST = 1.0
ATTR_ADJUST     = "adj_fctr"

# ── Treatment-zone options ──────────────────────────────────────────────────
# Buffer distance (CRS units, typically metres) applied outward from ALL
# treatment polygons before rasterizing.  Ensures the modeled baseline
# extends slightly beyond the mapped boundary so edge feathering blends
# smoothly.  Set to 0 for no buffer.  Example: 90 = three 30-m pixels.
BUFFER_M = 90.0   # metres (set to 0.0 to disable)

# ── Model options ────────────────────────────────────────────────────────────
# Include water table depth (WTD) as a covariate?  Set to False to exclude it
# for basins where the WTD product is unreliable or unavailable.
USE_WTD = True

# Include Height Above Nearest Drainage (HAND) as a covariate?  HAND is
# derived from the DEM in prep_statewide.py and captures distance-to-drainage
# in elevation units.  Set to False for basins where HAND is unreliable.
USE_HAND = True

# Which model backend to use:  "rf"  (RandomForest)  or  "lgbm"  (LightGBM)
MODEL_BACKEND = "lgbm"

# RandomForest hyper-parameters
RF_N_ESTIMATORS   = 200
RF_MAX_DEPTH      = 12
RF_MIN_SAMPLES_LEAF = 20
RF_N_JOBS         = -1     # use all cores

# LightGBM hyper-parameters
LGBM_N_ESTIMATORS  = 300
LGBM_MAX_DEPTH     = 8
LGBM_LEARNING_RATE = 0.05
LGBM_NUM_LEAVES    = 63
LGBM_MIN_CHILD     = 50
LGBM_N_JOBS        = -1

# ── Training options ─────────────────────────────────────────────────────────
# Maximum slope (degrees) for a pixel to be included in the training set.
# Steep hillslopes have deep water tables and negligible groundwater ET, so
# including them biases the model toward low values in the valley-bottom areas
# where phreatophyte ET actually occurs.  Set to None to disable the filter.
MAX_SLOPE_DEG = 5.0

# Maximum number of training pixels to sample (to keep memory/time reasonable).
# Set to None to use ALL outside-treatment valid pixels.
MAX_TRAIN_PIXELS = 500_000

# Random seed for reproducibility
RANDOM_SEED = 42

# ── Edge feathering / blending ───────────────────────────────────────────────
# Smooth the transition at treatment-zone boundaries so replaced/scaled values
# blend gradually into the surrounding landscape instead of hard edges.
#
# FEATHER_WIDTH_PX controls the characteristic width of the Gaussian transition
# band IN PIXELS measured inward from the treatment-zone boundary.  Within this
# band, the final ETg is a weighted blend:
#     final = w * treated  +  (1-w) * surrounding
# where w follows a Gaussian curve:  w = 1 − exp(−(d / sigma)²)
# with sigma = FEATHER_WIDTH_PX.  This gives a smooth, natural-looking ramp
# that starts gently at the boundary and approaches 100% treated deep inside.
# At d = sigma, w ≈ 0.63;  at d = 2×sigma, w ≈ 0.98.
#
# Set to 0 to disable feathering (hard edges).
FEATHER_WIDTH_PX = 4   # pixels (e.g. 5 × 30 m = 150 m; effective transition ~2× this)

# ── GeoTIFF output options ──────────────────────────────────────────────────
COMPRESS = "DEFLATE"
PREDICTOR_TIFF = 2        # horizontal differencing for float data
