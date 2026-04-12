# Changelog -- bmm-etg-raster-fill

All notable changes to this project are documented in this file.

## [0.6.0] - 2026-04-12

BpS symbology, spatially weighted fallback, custom study areas.

### Added
- **Spatially weighted BpS mean fallback**: when the terrain residual
  model's CV R² is negative, the fill script now computes per-BpS class
  means using a Gaussian spatial window (default ~1 km radius) instead
  of a flat basin-wide average.  Each treatment pixel draws more heavily
  from nearby training pixels of the same vegetation class, producing
  spatial gradients within each class rather than uniform rates.  The
  window radius is configurable via `spatial_fallback_radius_px` in
  config.toml (set to 0 to revert to flat class means).
- **BpS symbology preservation**: `prep_statewide.py` now extracts the
  LANDFIRE raster attribute table (class names + RGB colours) from the
  source BpS raster and caches it as `statewide/bps_lookup.json`.
  `prep_basin.py` and `prep_custom_basin.py` write QGIS-compatible
  `.clr` and `.qml` sidecar files alongside each basin's `BpS.tif` so
  it renders with named, coloured classes in QGIS automatically.
- **BpS class names in logs and metadata**: `etg_baseline_fill.py` now
  logs per-class mean ETg with human-readable LANDFIRE class names
  (e.g. "Great Basin Xeric Mixed Sagebrush Shrubland" instead of just
  code 10190) and writes a `[bps_class_means]` section in run metadata.
- **Custom study areas** (`prep_custom_basin.py`): new script for
  setting up basin directories outside the Nevada NWI framework.
  Accepts any boundary shapefile, clips DEM/BpS/WTD from CONUS-scale
  sources, derives HAND, and copies the boundary into `input/` so the
  fill script can use it as the training mask.  Auto-detects UTM zone
  from the boundary if the CRS is geographic.
- **`boundary_shp` in config.toml**: `basin_config.py` and
  `etg_baseline_fill.py` now support a `boundary_shp` field under
  `[inputs]`.  When present, the fill script uses this boundary for the
  training mask instead of requiring the NWI shapefile.  NWI basins are
  unaffected (NWI is still the fallback).
- New module `bps_utils.py` with `extract_bps_lookup()`,
  `load_bps_lookup()`, `write_bps_symbology()`, and `bps_name()`.

## [0.5.0] - 2026-04-12

Early stopping and automatic BpS-mean fallback.

### Added
- LightGBM early stopping: when the training set has at least 200 pixels,
  a 20% validation split is held out and LightGBM stops adding trees once
  validation loss fails to improve for 20 rounds.  The model is then
  refitted on the full training set with the optimal tree count so no
  data is wasted.  The log reports trees used vs. configured maximum.
- Automatic BpS-mean-only fallback: if 3-fold cross-validation R² is
  negative (meaning the residual model is making predictions worse, not
  better), the residual is zeroed out and the baseline becomes the
  per-BpS class mean only.  A warning is logged and the run metadata
  records `use_residual_model = no`.
- Run metadata now logs `lgbm_trees_used`, `early_stopping`, and
  `use_residual_model` in the `[model]` section.

### Fixed
- HAND derivation: clipped DEMs now have nodata pixels filled with an
  elevation wall (max elevation + 1000 m) before whitebox processing,
  so D8 flow routes inward toward the valley floor rather than dying at
  the polygon boundary.  The wall is masked back to NaN after HAND is
  computed.
- Switched whitebox pipeline from `BreachDepressionsLeastCost` to
  `FillDepressions` with `fix_flats=True`.  The breach algorithm spent
  hours searching for least-cost paths through the elevation wall;
  `FillDepressions` completes in seconds with equivalent results for
  HAND derivation.
- Added file-existence checks between each whitebox pipeline step so a
  failed step stops the chain immediately instead of cascading.
- TOML template in `prep_basin.py` now writes with explicit
  `encoding="utf-8"` and uses ASCII-only characters in comments, fixing
  a `UnicodeDecodeError` on Windows where `tomllib` expected UTF-8 but
  the file was written in cp1252.
- `basin_config.py` now auto-repairs existing cp1252-encoded config files
  by re-encoding them to UTF-8 on first load.
- HAND diagnostic block added to `etg_baseline_fill.py`: if HAND has 0
  valid pixels after reprojection, the source raster's bounds, CRS,
  nodata value, shape, dtype, and finite pixel count are logged.
- LGBMRegressor feature-names warning eliminated by wrapping both
  training and prediction arrays in `pd.DataFrame` with consistent
  column names.

## [0.4.3] - 2026-04-11

Move HAND derivation from statewide to per-basin.

### Changed
- HAND (Height Above Nearest Drainage) is now derived inside
  `prep_basin.py` from each basin's 5 km-buffered clipped DEM, rather
  than on the full statewide DEM.  whitebox-tools' `ElevationAboveStream`
  attempted to allocate ~25 GB of RAM on the Nevada-wide 30 m DEM and
  ran out of memory; per-basin DEMs are small enough that the same
  pipeline runs in seconds with ordinary memory usage.
- `prep_basin.py` imports `_derive_hand_from_dem`,
  `HAND_STREAM_THRESHOLD_CELLS`, and `HAND_BREACH_DIST_CELLS` from
  `prep_statewide.py`, so there is still a single source of truth for
  the whitebox pipeline.
- New `prep_basin.py` CLI flags: `--skip-hand`, `--hand-threshold`,
  `--hand-breach-dist`, `--keep-hand-intermediates` (mirror of the old
  flags on `prep_statewide.py`).
- If HAND derivation fails for an individual basin in a `--all` batch
  run, the basin continues without HAND rather than aborting the batch.

### Removed
- HAND no longer produced at the statewide level: `HAND_statewide.tif`
  and the corresponding `--skip-hand` / `--hand-threshold` /
  `--hand-breach-dist` / `--keep-hand-intermediates` flags are gone
  from `prep_statewide.py`.  The `_derive_hand_from_dem()` helper is
  retained and exported.
- `HAND_statewide.tif` removed from the `prep_basin.py` statewide
  presence check.

### Fixed
- Whitebox HAND pipeline now uses absolute paths for every tool input
  and output, with `set_verbose_mode(True)` plus a callback that
  captures whitebox's stdout.  If a tool silently fails or produces no
  output, the script logs the work-directory contents and the tail of
  whitebox's captured messages instead of just "output not produced".

## [0.4.2] - 2026-04-11

HAND (Height Above Nearest Drainage) covariate.

### Added
- `prep_statewide.py` now derives a statewide HAND raster from the 3DEP DEM
  using the standard whitebox-tools pipeline:
  `BreachDepressionsLeastCost → D8FlowAccumulation → ExtractStreams →
  ElevationAboveStream`. Outputs `statewide/HAND_statewide.tif` on the same
  grid and CRS as `DEM_statewide.tif`. Tunable via new CLI flags
  `--skip-hand`, `--hand-threshold` (stream-extraction flow accumulation
  threshold in cells; default 1000), `--hand-breach-dist` (max breach
  distance in cells; default 50, chosen to keep closed Great Basin sub-basins
  from being bridged through low passes), and `--keep-hand-intermediates`.
- `prep_basin.py` now clips `HAND_statewide.tif` to each basin into
  `basins/<key>/input/HAND.tif` alongside DEM/BpS/WTD.
- `etg_baseline_fill.py` adds HAND as a fourth covariate alongside
  elevation, slope, and WTD. The training mask, sampler, feature stack, and
  chunked prediction loop are all refactored to dynamically include or
  exclude WTD and HAND based on availability.
- `use_hand` toggle in basin `config.toml` `[model]` section (default `true`),
  mirroring `use_wtd`. Set to `false` to drop HAND for basins where the
  derived raster is unreliable.
- `HAND_TIF` and `USE_HAND` exposed by both `basin_config.py` and legacy
  `config.py`.
- `whitebox` added to `environment.yml` (pip section) for HAND derivation.

### Changed
- Run metadata (`<basin>_run_metadata.txt`) now logs `hand_tif`, `use_wtd`,
  and `use_hand` alongside the existing inputs/model fields. The `features`
  list in the `[training]` section reflects the actual covariate stack used
  (e.g. `['elevation', 'slope', 'wtd', 'hand']`).

## [0.4.1] - 2026-04-10

Downward-only replacement cap and statewide CRS alignment.

### Added
- Downward-only cap in `etg_baseline_fill.py` Step 7a: the adjusted baseline
  is clipped per-pixel to the original input ETg, so the fill can only
  *reduce* ETg in treatment zones, never raise it. A capped-pixel count is
  written to the run log.
- Matching cap in Step 7b feathering: blended values outside the treatment
  boundary are clamped to the raw ETg so feathering cannot introduce upward
  adjustment either.
- `prep_statewide.py` now reads the NWI shapefile CRS and reprojects all
  clipped statewide rasters (DEM, BpS, WTD, 3DEP download) into that CRS.
  The target CRS and source CRS are logged for each raster.

### Changed
- Basin boundary shapefile renamed `NWI_Investigations.shp` →
  `NWI_Investigations_EPSG_32611.shp` (EPSG:32611 / UTM 11N WGS84).
  References updated in `prep_statewide.py`, `prep_basin.py`,
  `etg_baseline_fill.py`, `.gitignore`, and `README.md`.

## [0.4.0] - 2026-04-09

Expert adjustment knob for hydrologist-driven baseline tuning.

### Added
- `baseline_adjust` parameter in `config.toml` `[adjustment]` section: basin-wide
  multiplicative scalar (default 1.0 = no change) applied to modeled baseline
- Per-polygon override via `adj_fctr` column in treatment shapefile: if present
  and > 0 for a polygon, overrides the basin-wide default for that polygon's pixels
- `BASELINE_ADJUST` and `ATTR_ADJUST` added to both `basin_config.py` and
  legacy `config.py`
- Per-pixel adjustment factor raster built at Step 2b from basin default +
  polygon-level overrides
- `adj_factor` column in polygon summary CSV recording the effective adjustment
  for each treatment polygon
- Adjustment metadata persisted in `run_metadata.txt` `[treatment]` section

### Changed
- Step 7a applies adjustment before filling treatment zones:
  `adjusted_baseline = baseline * adjustment_factor`
- Feathering blends the adjusted baseline (not the raw model prediction) with
  surrounding ETg

## [0.3.0] - 2026-04-08

Scaled architecture for statewide multi-basin processing.

### Added
- `prep_statewide.py`: one-time script to clip CONUS DEM, BpS, and WTD rasters
  to the dissolved NWI investigation boundary (with 10 km buffer)
- `prep_basin.py`: per-basin setup (clips covariates from statewide subsets,
  generates default `config.toml`, creates directory structure)
- `basin_config.py`: TOML configuration reader exposing the same interface as
  the legacy `config.py`, enabling per-basin parameter customization
- `run_all.py`: orchestrator for batch-processing multiple basins (prep, fill,
  diagnostics, ET unit summary) with `--dry-run`, `--list`, and selective flags
- Per-basin `config.toml` files with all tunable parameters (buffer, feathering,
  slope threshold, model backend, CRS overrides, etc.)
- Basin boundary mask: training data is now constrained to within-basin pixels
  using the NWI investigation polygon, preventing out-of-basin contamination
- Per-basin log files (`{key}_run.log`) for debugging batch runs
- Graceful skip for basins with fewer than 50 valid training pixels (writes a
  `_SKIPPED.txt` marker instead of crashing)
- Cross-basin summary CSV (`cross_basin_summary.csv`) collecting key metrics
  (training pixels, CV R-squared, treatment pixels, percent change, feature
  importances) from all processed basins
- Feature importances persisted in run metadata `[feature_importance]` section
- Negative-baseline clipping count logged and saved to metadata
- `py3dep` and `tomli` added to `environment.yml`

### Changed
- All three downstream scripts (`etg_baseline_fill.py`, `diagnostics.py`,
  `etunit_summary.py`) now auto-detect TOML-based basin configs vs. legacy
  `config.py` based on whether `basins/{name}/config.toml` exists
- Basin key derived from `Basin` field in the NWI shapefile (e.g.,
  `101_SierraValley`, `137A_BigSmokyValley`)

## [0.2.0] - 2026-04-08

Simplified treatment logic and corrected feathering direction.

### Changed
- Removed the scale vs. replace distinction: all treatment-zone polygons now
  receive full replacement with the modeled baseline
- Removed `SCALE_SOURCE`, `USE_THRESHOLD`, `THRESHOLD_FT` config parameters
- Removed `scale_raster.tif`, `replace_flag.tif`, `scale_factor_effective.tif`
  output rasters
- Renamed `REPLACE_BUFFER_M` to `BUFFER_M` (applies to all treatment polygons)
- Gaussian feathering now applies *outside* the treatment boundary (previously
  was inside), creating a smooth transition from baseline into surrounding raw ETg
- Diagnostics simplified: single "treatment" category replaces separate
  replace/scale traces in histograms, scatter plots, boxplots, and maps
- Polygon summary CSV no longer includes `assigned_scale_fctr` or `scale_source`
  columns; treatment column is `"replaced"` or `"none"`

## [0.1.0] - 2026-04-02

Initial proof-of-concept release for the Sierra Valley study area.

### Added
- Two-stage baseline ETg model (BpS class means + LightGBM/RandomForest terrain
  residual) trained on outside-treatment pixels
- Dual treatment-zone support: full replacement zones and scale-factor zones,
  controlled by shapefile attributes (`rplc_rt`, `scale_fctr`)
- Gaussian edge feathering for replacement zones with configurable sigma
- Three scale-factor modes: `modeled`, `assigned`, and `average`
- CRS override mechanism for rasters with malformed coordinate systems (e.g.,
  LANDFIRE BpS "EngineeringCRS" issue)
- Configurable replacement-zone polygon buffering
- Per-polygon summary CSV with input, baseline, and final mean ETg
- ET-unit-level summary CSV with area, volume, and rate statistics
- Diagnostic suite: histograms, scatter plots, BpS box-plots, map panels,
  difference maps, treatment-type maps, feather weight visualization
- Conda environment specification (`environment.yml`)
- AI-assisted development disclosure
