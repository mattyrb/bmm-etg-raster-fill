"""
basin_config.py  –  Read per-basin TOML config and expose the same interface
that etg_baseline_fill.py, diagnostics.py, and etunit_summary.py expect.

Replaces the old config.py for the scaled-up multi-basin workflow.
The old config.py is kept for backward compatibility with the two original
study areas (SierraValley, PineValley) but new basins should use this module.

Usage in downstream scripts:

    import basin_config as cfg

    cfg.load_basin("101_SierraValley")
    # or
    cfg.load_basin_from_toml(Path("basins/101_SierraValley/config.toml"))

    # Now use cfg.ETG_TIF, cfg.OUT_DIR, cfg.BUFFER_M, etc.
"""

import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # Python < 3.11
    except ImportError:
        sys.exit("Python 3.11+ or 'tomli' package required for TOML support.\n"
                 "Install with:  pip install tomli")

_here = Path(__file__).resolve().parent
PROJECT_DIR = _here
BASINS_DIR = PROJECT_DIR / "basins"

# ── Module-level attributes (populated by load_basin) ──────────────────────
# These mirror the interface of the old config.py so fill/diagnostics/summary
# scripts work without changes.

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

# Treatment
ATTR_SCALE       = "scale_fctr"
ATTR_REPLACE     = "rplc_rt"
BUFFER_M         = 90.0
FEATHER_WIDTH_PX = 4

# Expert adjustment
BASELINE_ADJUST  = 1.0    # basin-wide scalar (1.0 = no change)
ATTR_ADJUST      = "adj_fctr"  # per-polygon override column in shapefile

# Model
USE_WTD          = True   # include WTD as a covariate (set False to drop it)
USE_HAND         = True   # include HAND as a covariate (set False to drop it)
MODEL_BACKEND    = "lgbm"
MAX_SLOPE_DEG    = 5.0
MAX_TRAIN_PIXELS = 500_000
RANDOM_SEED      = 42

# RandomForest
RF_N_ESTIMATORS     = 200
RF_MAX_DEPTH        = 12
RF_MIN_SAMPLES_LEAF = 20
RF_N_JOBS           = -1

# LightGBM
LGBM_N_ESTIMATORS  = 300
LGBM_MAX_DEPTH     = 8
LGBM_LEARNING_RATE = 0.05
LGBM_NUM_LEAVES    = 63
LGBM_MIN_CHILD     = 50
LGBM_N_JOBS        = -1

# Output
COMPRESS        = "DEFLATE"
PREDICTOR_TIFF  = 2


def load_basin(basin_key: str) -> None:
    """
    Load configuration for a basin by key (e.g. "101_SierraValley").
    Reads  basins/<basin_key>/config.toml.
    """
    toml_path = BASINS_DIR / basin_key / "config.toml"
    if not toml_path.exists():
        sys.exit(
            f"ERROR: config.toml not found for basin '{basin_key}'.\n"
            f"  Expected: {toml_path}\n"
            f"  Run prep_basin.py first, or create config.toml manually."
        )
    load_basin_from_toml(toml_path)


def load_basin_from_toml(toml_path: Path) -> None:
    """
    Load configuration from a specific TOML file.
    Populates module-level attributes so downstream code can use
    ``cfg.ETG_TIF``, ``cfg.OUT_DIR``, etc.
    """
    toml_path = Path(toml_path).resolve()
    basin_dir = toml_path.parent
    input_dir = basin_dir / "input"

    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    g = globals()

    # ── Basin identity ──────────────────────────────────────────────────────
    basin_section = raw.get("basin", {})
    basin_key = basin_section.get("basin_key", basin_dir.name)
    g["STUDY_AREA_NAME"] = basin_key
    g["DATA_DIR"] = input_dir

    # ── Inputs ──────────────────────────────────────────────────────────────
    inputs = raw.get("inputs", {})

    def _resolve_input(key, fallback=None):
        val = inputs.get(key, fallback)
        if val is None or val == "" or val.startswith("#"):
            return None
        p = input_dir / val
        return p if p.exists() else None

    g["ETG_TIF"]       = _resolve_input("etg_tif")
    g["ETG_RAW_TIF"]   = _resolve_input("etg_raw_tif")
    g["DEM_TIF"]       = _resolve_input("dem_tif", "DEM.tif")
    g["BPS_TIF"]       = _resolve_input("bps_tif", "BpS.tif")
    g["WTD_TIF"]       = _resolve_input("wtd_tif", "WTD.tif")
    g["HAND_TIF"]      = _resolve_input("hand_tif", "HAND.tif")
    g["TREATMENT_SHP"] = _resolve_input("treatment_shp")

    # ── Output directory ────────────────────────────────────────────────────
    g["OUT_DIR"] = basin_dir / "output"

    # ── Treatment parameters ────────────────────────────────────────────────
    treat = raw.get("treatment", {})
    g["BUFFER_M"]         = float(treat.get("buffer_m", 90.0))
    g["FEATHER_WIDTH_PX"] = int(treat.get("feather_width_px", 4))
    g["ATTR_SCALE"]       = treat.get("attr_scale", "scale_fctr")
    g["ATTR_REPLACE"]     = treat.get("attr_replace", "rplc_rt")

    # ── Expert adjustment parameters ───────────────────────────────────────
    adj = raw.get("adjustment", {})
    g["BASELINE_ADJUST"]  = float(adj.get("baseline_adjust", 1.0))
    g["ATTR_ADJUST"]      = adj.get("attr_adjust", "adj_fctr")

    # ── Model parameters ────────────────────────────────────────────────────
    model = raw.get("model", {})
    g["USE_WTD"]          = bool(model.get("use_wtd", True))
    g["USE_HAND"]         = bool(model.get("use_hand", True))
    g["MODEL_BACKEND"]    = model.get("backend", "lgbm")
    g["MAX_SLOPE_DEG"]    = _float_or_none(model.get("max_slope_deg", 5.0))
    g["MAX_TRAIN_PIXELS"] = _int_or_none(model.get("max_train_pixels", 500_000))
    g["RANDOM_SEED"]      = int(model.get("random_seed", 42))

    # LightGBM overrides
    lgbm = raw.get("lgbm", {})
    g["LGBM_N_ESTIMATORS"]  = int(lgbm.get("n_estimators", 300))
    g["LGBM_MAX_DEPTH"]     = int(lgbm.get("max_depth", 8))
    g["LGBM_LEARNING_RATE"] = float(lgbm.get("learning_rate", 0.05))
    g["LGBM_NUM_LEAVES"]    = int(lgbm.get("num_leaves", 63))
    g["LGBM_MIN_CHILD"]     = int(lgbm.get("min_child", 50))
    g["LGBM_N_JOBS"]        = int(lgbm.get("n_jobs", -1))

    # RandomForest overrides
    rf = raw.get("rf", {})
    g["RF_N_ESTIMATORS"]     = int(rf.get("n_estimators", 200))
    g["RF_MAX_DEPTH"]        = int(rf.get("max_depth", 12))
    g["RF_MIN_SAMPLES_LEAF"] = int(rf.get("min_samples_leaf", 20))
    g["RF_N_JOBS"]           = int(rf.get("n_jobs", -1))

    # ── CRS overrides ──────────────────────────────────────────────────────
    g["CRS_OVERRIDES"] = raw.get("crs_overrides", {})

    # ── Validate required inputs ────────────────────────────────────────────
    missing = []
    for key in ("ETG_TIF", "DEM_TIF", "BPS_TIF", "TREATMENT_SHP"):
        if g[key] is None:
            missing.append(key)
    if missing:
        sys.exit(
            f"ERROR: Missing required inputs for basin '{basin_key}':\n"
            + "\n".join(f"  - {k}: check config.toml [inputs] and "
                        f"files in {input_dir}" for k in missing)
        )


def available_areas() -> list[str]:
    """Return sorted list of basin keys that have a config.toml."""
    if not BASINS_DIR.exists():
        return []
    return sorted(
        d.name for d in BASINS_DIR.iterdir()
        if d.is_dir() and (d / "config.toml").exists()
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _float_or_none(val):
    if val is None or val == 0:
        return None
    return float(val)


def _int_or_none(val):
    if val is None or val == 0:
        return None
    return int(val)
