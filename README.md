# bmm-etg-raster-fill

Data-driven estimation of counterfactual groundwater evapotranspiration (ETg) for
irrigated and augmented areas in closed-basin groundwater budgets.

The resulting ETg product refines basin water budgets and supports decomposition
of Landsat-derived total ET into its natural and applied-water components
(`ETa_applied = ETa_total − ETg`).

This tool replaces the previous manual approach of burning uniform ETg replacement
rates or scale factors into 30-meter Landsat-derived ETg rasters. Instead, it builds
a statistical model of what ETg *would* look like in irrigated areas if the
irrigation were absent, using terrain and ecological covariates that do not carry
an irrigation signal.

Developed at the [Desert Research Institute (DRI)](https://www.dri.edu/) for
Nevada's NWI investigation basins (257 hydrographic areas extending into adjacent
states).  The workflow also supports custom study areas outside the NWI framework
via `prep_custom_basin.py` (see Quick Start, step 7).

> **AI disclosure:** This codebase was developed collaboratively with Claude (Anthropic).
> All scientific decisions, model design choices, and domain validation were made by DRI
> research staff. Claude assisted with code implementation, debugging, and documentation
> under human direction and review. See [AI_DISCLOSURE.md](AI_DISCLOSURE.md) for details.


## What this tool does, in plain language

In arid basins across Nevada, some of the water that evaporates or is transpired by
plants comes from groundwater. Measuring this "groundwater ET" across an entire
valley is important for managing the water budget, and satellites make it possible:
we can estimate how much water each 30-meter patch of land is losing to the
atmosphere by looking at how green the vegetation is in Landsat imagery.

The challenge is irrigated farmland. A center-pivot alfalfa field shows up as very
green in the satellite image, and the ET estimate is high -- but most of that water
came from a well or a ditch, not from the natural water table. To build an accurate
groundwater budget, we need to answer the question: *if that field had never been
irrigated, how much groundwater ET would the native vegetation at that spot
naturally produce?*

Previously, a hydrologist would draw a polygon around each irrigated area, pick a
single number (a "replacement rate") based on professional judgment, and paste that
number uniformly across every pixel in the polygon. This is reasonable, but it means
a pixel at the edge of a field next to a hillslope gets the same value as a pixel in
the middle of a wet meadow -- there's no spatial variation within each polygon.

This tool replaces that manual step with a statistical model. It works by looking at
all the land *outside* the irrigated areas -- the sagebrush flats, the un-irrigated
meadows, the hillslopes -- and learning the relationship between each pixel's ET and
two things the model can observe everywhere:

1. **What type of vegetation naturally grows there** (from a national vegetation
   map called LANDFIRE Biophysical Settings). Sagebrush uses less water than
   riparian willows, for example.

2. **The terrain** (elevation and slope from a digital elevation model). Valley
   bottoms with shallow water tables tend to have higher natural ET than steep
   hillslopes, even within the same vegetation type.

Once the model learns these patterns from the non-irrigated landscape, it predicts
what each irrigated pixel's ET *would* be if the irrigation were removed. The result
is a spatially varying "natural baseline" that respects the actual topography and
ecology at every pixel, rather than a single flat number per polygon.

All treatment-zone polygons are fully replaced with the modeled baseline. Gaussian
edge feathering smooths the transition *outside* the treatment boundary so replaced
values blend gradually into the surrounding landscape with no abrupt jumps.


## Background

### The Beamer-Minor Model

This tool builds on the Beamer-Minor Model (BMM), a place-based approach for
estimating groundwater evapotranspiration (ETg) from phreatophyte and riparian
vegetation across the Great Basin. The original framework established an empirical
relationship between Landsat-derived vegetation indices and flux-tower-calibrated
ET measurements to produce spatially distributed ETg estimates at 30-meter
resolution (Beamer et al., 2013). Minor (2019) extended this work to a
basin-by-basin methodology for estimating annual ETg across Great Basin
hydrographic areas, refining the regression models and expanding the approach to
operational scales. The combined BMM framework has been applied across Nevada and
northeastern California, including basin-scale groundwater discharge assessments
for the Humboldt River Basin (Huntington et al., 2022) and site-specific
monitoring at contamination and remediation sites (Huntington and Bromley, 2023).

A key step in the BMM workflow is the delineation of "ET units" -- polygons
representing distinct land-use and vegetation categories (irrigated cropland,
meadow, phreatophyte shrubland, riparian, wetland, etc.). Within irrigated ET
units, the Landsat-derived ETg signal includes both the natural groundwater
contribution and the enhancement from applied irrigation water. To isolate the
groundwater component, the legacy workflow assigns each irrigated polygon a
**replacement rate** (a uniform ETg value that overwrites the Landsat estimate)
chosen by an analyst based on field knowledge and professional judgment,
then "burned" into the raster.

### The problem

This burn-in approach is defensible at the polygon level but lacks spatial realism:
every pixel inside a given polygon receives the same ETg value, regardless of its
position on the landscape. Valley-bottom pixels near streams and hillslope pixels
at polygon margins get identical rates, when in reality natural ETg varies
continuously with topography, depth to water table, and vegetation community.

### This tool's approach

Rather than using a single number per polygon, this tool asks: *What would ETg
look like at each 30-meter pixel if the irrigation were removed?*

It answers that question with a two-stage statistical model trained exclusively on
pixels **outside** irrigated areas **and within the basin boundary**:

1. **Ecological-setting baseline.** Every pixel belongs to a LANDFIRE Biophysical
   Settings (BpS) vegetation class. The model computes the average ETg for each
   BpS class from the non-irrigated training pixels. This captures the dominant
   signal -- different plant communities use water at different rates.

2. **Terrain-driven refinement.** Within a single BpS class, ETg still varies:
   valley floors with shallow water tables tend to have higher ETg than hillslopes.
   A gradient-boosted tree model (LightGBM) or random forest learns the residual
   pattern `(ETg - BpS_mean)` as a function of elevation and slope derived from a
   DEM.

The predicted baseline is `BpS_mean + terrain_residual`, clamped at zero. This
value varies smoothly across the landscape and responds to real topographic
gradients, producing a more realistic counterfactual than a single uniform rate.

NDVI is intentionally **excluded** as a predictor because it carries the irrigation
signal we are trying to remove.

### Treatment handling

Any polygon in the treatment shapefile with `scale_fctr > 0` or `rplc_rt > 0` is
treated: its pixels are fully replaced with the modeled baseline. Polygons where
both attributes are zero are left untouched.

All treatment polygons are buffered outward (configurable, default 90 m) to
exclude irrigation edge effects from both training data and the replacement zone.
Gaussian feathering applies *outside* the treatment boundary, blending the
baseline-replaced values smoothly into the surrounding raw ETg landscape.


## Repository structure

```
project/
    statewide/                  Nevada-extent covariate rasters (one-time prep)
        DEM_statewide.tif       (optional; if missing, prep_basin.py pulls
                                 per-basin COP30 from OpenTopography)
        BpS_statewide.tif
        WTD_statewide.tif

    basins/                     Per-basin directories (257 NWI basins)
        101_SierraValley/
            config.toml         Per-basin configuration (auto-generated, editable)
            source/             Raw ETg raster(s) + treatment shapefile (you drop these in)
            input/              Prep-generated clipped covariates (DEM, BpS, WTD, HAND, ...)
            output/             Fill results, diagnostics, logs
        053_PineValley/
            ...
        _template/
            config.toml         # Template used by prep scripts to
                                # generate each basin's config.toml.
                                # Edit to change defaults for new basins.

    prep_statewide.py       One-time: clip CONUS rasters to NWI extent
    prep_basin.py           Per-basin: clip covariates + generate config.toml (NWI)
    prep_custom_basin.py    Set up a basin from any boundary shapefile (non-NWI)
    basin_config.py         TOML config reader (per-basin config interface)
    opentopo.py             OpenTopography COP30 downloader (per-basin DEM fallback)
    etg_baseline_fill.py    Main workflow (training, prediction, fill, feathering)
    diagnostics.py          Post-run plots and distribution summaries
    etunit_summary.py       ET-unit-level summary CSV (area, volume, rates)
    run_all.py              Orchestrator: batch-process multiple basins
    fetch_wtd.py            Download Ma et al. (2025) WTD from HydroFrame
    NWI_Investigations_EPSG_32611.shp  Basin boundaries, 257 areas (EPSG:32611 / UTM 11N)
    environment.yml         Conda environment specification
    README.md               This file
    WALKTHROUGH.md          Step-by-step guide for new users
    AI_DISCLOSURE.md        AI-assisted development disclosure
    CHANGELOG.md            Version history
    LICENSE                 MIT license
```


## Quick start

### 1. Install dependencies

```bash
conda env create -f environment.yml --solver=classic
conda activate bmm-etg-raster-fill
```

If you already have a compatible environment, the key packages are: `numpy`,
`rasterio`, `geopandas`, `fiona`, `shapely`, `pyproj`, `scipy`, `scikit-learn`,
`lightgbm`, `matplotlib`, `py3dep`, `whitebox`, and `tomli` (Python < 3.11).

### 2. Prepare statewide covariates (one time)

Clip CONUS-wide BpS and WTD rasters to the NWI investigation extent. You can
optionally pass `--dem /path/to/CONUS_DEM.tif` to build a statewide DEM too
(recommended if you already have a mosaicked COP30 / SRTM / 3DEP file on
disk). If you omit `--dem`, **no** `DEM_statewide.tif` is built; instead
`prep_basin.py` will download a per-basin COP30 tile from OpenTopography for
each basin's bbox. The per-basin path avoids py3dep's memory issues on
Nevada-scale AOIs.

```bash
python prep_statewide.py \
    --bps  /path/to/LF2020_BPS_CONUS.tif \
    --wtd  /path/to/wtd_conus.tif
    # optional: --dem /path/to/CONUS_DEM.tif
```

This writes `BpS_statewide.tif` and `WTD_statewide.tif` (plus
`DEM_statewide.tif` if `--dem` was supplied) to the `statewide/` directory,
all reprojected to the NWI shapefile CRS (EPSG:32611).

**OpenTopography API key.** Per-basin COP30 downloads use the OpenTopography
Global DEM API. A project-default key is embedded in `opentopo.py` for
convenience; override it by setting the `OPENTOPOGRAPHY_API_KEY` environment
variable. Get your own free key at https://portal.opentopography.org.

HAND (Height Above Nearest Drainage) is derived *per-basin* during
`prep_basin.py` rather than statewide. whitebox-tools' `ElevationAboveStream`
runs out of memory on the full Nevada DEM, so each basin's clipped DEM (with a
5 km buffer) is processed individually through the standard pipeline
(`FillDepressions → D8FlowAccumulation → ExtractStreams →
ElevationAboveStream`). This is also hydrologically appropriate: closed Great
Basin sub-basins drain internally, so within-basin "nearest stream" is the
correct HAND reference.

### 3. Set up basin directories

```bash
# List all 257 basin keys from the NWI shapefile:
python prep_basin.py --list

# Prep a single basin (clips covariates, derives HAND, generates config.toml):
python prep_basin.py 101_SierraValley

# Prep all basins at once:
python prep_basin.py --all

# Skip HAND derivation (e.g. for quick testing):
python prep_basin.py 101_SierraValley --skip-hand
```

`prep_basin.py` writes `DEM.tif`, `BpS.tif`, `WTD.tif`, and `HAND.tif` to
`basins/<basin_key>/input/`. HAND is derived from the clipped basin DEM using
whitebox-tools; tune the stream-extraction threshold with `--hand-threshold N`
(default 1000 cells ≈ 0.9 km² drainage area).

Then place each basin's ETg raster(s) and treatment shapefile into its
`basins/<basin_key>/source/` directory (the prep script creates it
empty) and review the generated `config.toml`.  The `input/` directory
is reserved for prep-generated clipped covariates; `source/` is for
user-supplied raws.

### 4. Configure per-basin parameters

Each basin gets a `config.toml` with sensible defaults. Edit as needed:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `[source] etg_tif` | -- | ETg raster filename (in `source/`) |
| `[source] treatment_shp` | -- | Treatment shapefile filename |
| `[source] boundary_shp` | -- | Optional basin boundary shapefile (custom basins) |
| `buffer_m` | `90.0` | Buffer distance (m) around treatment polygons |
| `feather_width_px` | `4` | Gaussian feathering sigma in pixels |
| `max_slope_deg` | `5.0` | Maximum slope for training pixels (degrees) |
| `backend` | `"lgbm"` | `"lgbm"` (LightGBM) or `"rf"` (RandomForest) |
| `use_wtd` | `true` | Include water-table depth covariate |
| `use_hand` | `true` | Include Height Above Nearest Drainage covariate |
| `use_soil` | `true` | Include gSSURGO soil covariates (AWC + depth to restrictive layer); each loads only if present in `input/` |
| `use_rem` | `false` | Include Relative Elevation Model (opt-in).  Only generated when prep is run with `--derive-rem`. |
| `spatial_fallback_radius_px` | `33` | Spatial window (pixels) for BpS mean fallback when terrain model fails; 0 = flat |
| `baseline_adjust` | `1.0` | Expert adjustment scalar (0.8 = reduce 20%) |

See `config.toml` comments for the full parameter list including CRS overrides
and LightGBM/RandomForest hyperparameters.

### 5. Run

```bash
# Single basin:
python etg_baseline_fill.py 101_SierraValley
python diagnostics.py 101_SierraValley
python etunit_summary.py 101_SierraValley

# All configured basins (prep + fill + diagnostics + summary):
python run_all.py

# Dry run -- show what would be processed:
python run_all.py --dry-run

# Check readiness of all basins:
python run_all.py --list
```

### 6. (Optional) Download WTD data programmatically

If you don't already have a water table depth raster, the `fetch_wtd.py` script
can download the Ma et al. (2025) 30-m CONUS product from HydroFrame:

```bash
# One-time setup: create a free account at https://hydrogen.princeton.edu/
# then register your API PIN (~48-hour expiry):
python fetch_wtd.py --register

# Download for Sierra Valley (default bounding box):
python fetch_wtd.py

# Or specify a custom bounding box (lat_min, lon_min, lat_max, lon_max):
python fetch_wtd.py --bbox 39.55 -120.30 39.95 -119.90 -o wtd_SierraValley.tif
```


### 7. Custom study areas (non-NWI)

For basins outside the Nevada NWI framework (e.g. Sierra Valley CA, or
basins in other states), use `prep_custom_basin.py` instead of the
statewide/per-basin NWI pipeline:

```bash
python prep_custom_basin.py SierraValley \
    --boundary  /path/to/sierra_valley_boundary.shp \
    --dem       /path/to/CONUS_DEM.tif \
    --bps       /path/to/LF2020_BPS_CONUS.tif \
    --wtd       /path/to/wtd_conus.tif
```

This clips all covariates to the boundary (writing them to `input/`),
derives HAND from the clipped DEM, copies the boundary shapefile into
`source/boundary.shp`, and generates a `config.toml` with a `[source]`
section that references it.  Drop your raw ETg raster and treatment
shapefile into `source/` alongside the boundary, then run the fill.

The boundary shapefile replaces the NWI dependency for the training
mask: `etg_baseline_fill.py` uses the `boundary_shp` config field to
constrain training pixels to within the study area, so the NWI
shapefile is not needed.

If your boundary file uses a geographic CRS (e.g. EPSG:4326), the
script auto-detects the appropriate UTM zone from the centroid. If it
uses a projected CRS, that CRS is used directly.

The resulting basin directory is identical in structure to an NWI basin.
All downstream scripts (`etg_baseline_fill.py`, `diagnostics.py`,
`etunit_summary.py`) work exactly the same way.


## Outputs

All output is written to `basins/<basin_key>/output/`. File names are prefixed
with the basin key (e.g., `101_SierraValley_ETg_final.tif`).

### Rasters (GeoTIFF, DEFLATE-compressed, float32)

| File | Description |
|------|-------------|
| `{key}_ETg_baseline_pred.tif` | Model-predicted baseline ETg for all pixels |
| `{key}_ETg_final.tif` | Final ETg raster with treatment zones filled |
| `{key}_ETg_pct_change.tif` | Per-pixel percent change (raw to final) |
| `treatment_zone.tif` | Binary mask: 1 = treatment polygon (buffered), 0 = outside |
| `feather_weight.tif` | Gaussian blend weight (1 = baseline, 0 = raw ETg) |
| `DEM_matched.tif` | DEM reprojected to ETg grid |
| `BpS_matched.tif` | BpS reprojected to ETg grid |
| `slope_matched.tif` | Slope in degrees derived from matched DEM |
| `WTD_matched.tif` | Water-table depth reprojected to ETg grid (if `use_wtd = true`) |
| `HAND_matched.tif` | Height Above Nearest Drainage reprojected to ETg grid (if `use_hand = true`) |
| `REM_matched.tif` | Relative Elevation Model reprojected to ETg grid (if `use_rem = true` and `REM.tif` exists) |
| `AWC_matched.tif` | gSSURGO Available Water Capacity reprojected to ETg grid (if present and `use_soil = true`) |
| `SoilDepth_matched.tif` | gSSURGO depth to restrictive layer reprojected to ETg grid (if present and `use_soil = true`) |

### Tables (CSV)

| File | Description |
|------|-------------|
| `{key}_polygon_summary.csv` | Per-polygon statistics: pixel count, treatment type, mean input/baseline/final ETg |
| `{key}_ETUNIT_SUMMARY.csv` | ET-unit-level summary: area (ac), volume (ac-ft), rate (ft/yr) with uncertainty |
| `cross_basin_summary.csv` | (Project root) Cross-basin comparison of model metrics after batch runs |

### Metadata and logs

| File | Description |
|------|-------------|
| `{key}_run_metadata.txt` | Configuration, training stats, feature importance, BpS class means (with names), results |
| `{key}_run.log` | Timestamped log of the full run for debugging |

### BpS symbology (written by prep scripts)

| File | Description |
|------|-------------|
| `input/BpS.clr` | ESRI colour file: maps BpS codes to RGB + class name (works in QGIS and ArcGIS) |
| `input/BpS.qml` | QGIS layer style file: auto-loads when opening BpS.tif in QGIS |
| `statewide/bps_lookup.json` | Cached BpS class lookup (code, R, G, B, name) extracted from LANDFIRE source |
| `{key}_SKIPPED.txt` | Written if basin had too few training pixels (< 50) |

### Diagnostic plots (PNG)

| File | Shows |
|------|-------|
| `{key}_diag_histogram.png` | ETg distributions: outside vs treatment zones (before and after) |
| `{key}_diag_scatter.png` | Predicted baseline vs. original ETg for treatment-zone pixels |
| `{key}_diag_bps_boxplots.png` | ETg by BpS class: outside vs treatment |
| `{key}_diag_map_panels.png` | Three-panel map: original, baseline, final |
| `{key}_diag_difference_map.png` | Change map in treatment zones (red = reduced, blue = increased) |
| `{key}_diag_treatment_map.png` | Treatment-zone map (outside / treatment) |
| `{key}_diag_feather_map.png` | Feather blend weight and edge detail |
| `{key}_diag_pct_change_map.png` | Per-pixel percent change map |


## How the workflow operates

This section walks through `etg_baseline_fill.py` in plain language.

### Step 1: Read the ETg raster

The input ETg raster defines the "template grid" -- its coordinate reference system
(CRS), spatial extent, resolution, and pixel dimensions become the reference that
all other inputs are aligned to. The raster is read as a 2-D array of floating-point
values in ft/yr.

### Step 2: Rasterize the treatment shapefile

The treatment shapefile is read and, if necessary, reprojected to match the ETg
CRS. Any polygon with `scale_fctr > 0` or `rplc_rt > 0` is classified as a
treatment polygon. All treatment polygons are buffered outward (default 90 m) and
rasterized into a single binary treatment-zone mask.

All pixels inside the treatment zone are excluded from training, since those
values reflect irrigation and cannot be trusted. The original ETg values in
these pixels are overwritten by the modeled baseline in the final output, with
Gaussian feathering applied in a band *outside* the treatment boundary to
smooth the transition into the surrounding (un-replaced) ETg landscape.

### Step 3: Align covariates to the ETg grid

The DEM and BpS rasters are reprojected and resampled to exactly match the ETg
grid. The DEM uses bilinear interpolation (continuous surface), while BpS uses
nearest-neighbor (categorical data). If configured, the water table depth
(WTD) and Height Above Nearest Drainage (HAND) rasters are also aligned. CRS
overrides in `config.toml` handle rasters with malformed coordinate systems.
Either covariate can be disabled per-basin by setting `use_wtd = false` or
`use_hand = false` in `config.toml [model]`.

### Step 4: Compute slope and basin boundary mask

Slope in degrees is computed from the aligned DEM using Horn's 3x3
finite-difference method. The basin boundary (either `boundary_shp` from the
config, or the matching polygon from the NWI shapefile for NWI-formatted
basin keys) is rasterized to constrain training data to within-basin pixels
only. For custom basins without a configured boundary, all valid pixels are
used.

### Step 5: Train the two-stage model

Training data consists of all pixels that are: (a) outside treatment zones,
(b) within the basin boundary, (c) on slopes below the configured threshold,
and (d) have valid ETg, DEM, slope, BpS, WTD (if enabled), HAND (if
enabled), and soil covariates (if enabled and present) values with ETg > 0. Up to 500,000 pixels are randomly sampled if
the training set is larger.

If fewer than 50 valid training pixels remain, the basin is gracefully skipped
with a marker file rather than crashing the batch run.

**Stage 1 -- BpS class means:** The mean ETg is computed for each BpS vegetation
class from the training pixels. This lookup table captures the dominant
ecological-setting signal.

**Stage 2 -- Terrain residual model:** The residual `(ETg - BpS_class_mean)` is
modeled as a function of elevation, slope, and (optionally) water table depth,
HAND, and gSSURGO soil covariates (AWC, depth to restrictive layer) using
LightGBM or RandomForest. For LightGBM, early stopping is used
to prevent over-training: 20% of the training data is held out as a validation
split, and tree-building stops when the validation loss fails to improve for 20
consecutive rounds. The model is then refitted on the full training set with
the optimal tree count. A 3-fold cross-validation (CV) reports the residual
R-squared. Feature importances are logged and saved to the run metadata.

**Automatic fallback:** If the 3-fold CV R-squared is negative -- meaning the
residual model's predictions are worse than simply using the BpS class mean
alone -- the residual model is skipped and the baseline becomes the per-BpS
mean only. This prevents over-trained terrain models from injecting noise into
the output for basins where the terrain covariates carry little signal.

### Step 6: Predict baseline ETg

For every pixel with valid covariates, the predicted baseline is
`BpS_class_mean + terrain_residual_prediction`, clamped to a minimum of zero
(or just `BpS_class_mean` if the residual model was skipped due to negative
CV R-squared). BpS classes that appear only inside treatment zones fall back
to the global training mean. The number of pixels clipped from negative values
is logged.

### Step 7: Fill treatment zones, apply expert adjustment, and feather edges

Before filling, an expert adjustment factor is applied to the modeled baseline.
This lets a hydrologist tune the predicted baseline up or down based on
professional judgment or on-the-ground information.

The adjustment is a multiplicative scalar: `adjusted_baseline = baseline * factor`.
A basin-wide default is set in `config.toml` (`baseline_adjust`, default 1.0 = no
change). Individual polygons can override the basin default via an `adj_fctr`
column in the treatment shapefile -- if the column exists and a polygon's value
is > 0, it takes precedence over the basin-wide default for that polygon's pixels.

All treatment-zone pixels then receive the adjusted baseline prediction.

Gaussian feathering creates a smooth transition *outside* the treatment
boundary. A distance transform computes how far each outside pixel is from the
nearest treatment pixel, and a Gaussian weight function blends the adjusted
baseline with the surrounding raw ETg:

```
weight = exp(-(distance / sigma)^2)
final  = weight * adjusted_baseline + (1 - weight) * raw_ETg
```

At the boundary, `weight = 1` (fully adjusted baseline); far outside,
`weight -> 0` (fully raw ETg). Inside the treatment zone, pixels retain the full
adjusted baseline value. The transition width is controlled by `feather_width_px`
(default 4 pixels ~ 120 m).

A per-pixel percent change raster and figure are also produced comparing the raw
ETg to the final product.

### Step 8: Summary statistics

A per-polygon summary CSV is written with pixel counts, treatment status, and mean
ETg values (input, baseline, final) for every polygon. Run metadata (configuration,
training stats, feature importances, results) is saved alongside.

The run log reports two volume-change numbers against the original input raster:
a basin-wide `Total ETg volume change vs original input` (over all pixels valid
in both rasters), and a `Treatment-zone ETg volume change` restricted to
treatment pixels. Together these quantify how much of the ETg signal was
replaced by the modeled baseline and how much of the basin-wide budget it
represents.


## Known considerations

### Valley-bottom training bias

Valley floors tend to have higher natural ETg than hillslopes within the same BpS
class, because of shallow water tables. If most valley-bottom pixels are irrigated
(and therefore excluded from training), the model may underestimate baseline ETg in
those areas. Inspect the scatter and histogram diagnostics for signs of this.

### BpS class coverage

If a BpS class appears only inside treatment zones and never outside, the model has
no direct training data for it and falls back to the global mean. The diagnostic
box-plots help identify this situation.

### Buffer distance

Irrigation effects (lateral wetting, spray drift) extend beyond strict field
boundaries. The `buffer_m` parameter excludes these edge pixels from training.
Increase it for areas with flood irrigation or wide influence zones.

### Basin boundary mask

Training data is constrained to pixels within the NWI basin boundary. This prevents
out-of-basin pixels (from covariate rasters that extend beyond the ETg extent) from
influencing the model. If the NWI shapefile is not found, a warning is logged and
all valid pixels are used.

### Small basins

Basins with fewer than 50 valid training pixels (after excluding treatment zones,
steep slopes, and out-of-basin areas) are automatically skipped. A `_SKIPPED.txt`
marker file is written to the output directory, and the cross-basin summary CSV
flags these basins for review.


## Future directions

- Statewide gSSURGO clip script to auto-populate `AWC.tif` / `SoilDepth.tif`
  per basin (currently user-supplied)
- Add topographic wetness index (TWI) and distance-to-stream
- Ensemble multiple models and use prediction intervals for uncertainty
- Spatial block cross-validation to test for autocorrelation leakage
- Extend to multi-year time series for temporal coverage
- Parallel basin processing for faster batch runs


## References

Beamer, J.P., Huntington, J.L., Morton, C.G., and Pohll, G.M., 2013,
Estimation of annual groundwater evapotranspiration from phreatophyte vegetation
in the Great Basin using Landsat and flux tower measurements: Journal of the
American Water Resources Association, v. 49, no. 3, p. 518-533,
[doi:10.1111/jawr.12058](https://doi.org/10.1111/jawr.12058).

Minor, B.A., 2019, Estimating annual groundwater evapotranspiration from
hydrographic areas in the Great Basin using remote sensing and evapotranspiration
data measured by flux tower systems: University of Nevada, Reno, unpublished
master's thesis.

Huntington, J.L., Bromley, M., and others, 2022, Groundwater discharge from
phreatophyte vegetation, Humboldt River Basin, Nevada: Desert Research Institute,
Publication No. 41288,
[project page](https://www.dri.edu/project/humboldt-etg/) |
[report (PDF)](https://s3-us-west-2.amazonaws.com/webfiles.dri.edu/Labs/Huntington/41288%20-%20Humboldt_ET_Final_Report.pdf).

Huntington, J.L. and Bromley, M., 2023, Remote sensing of evapotranspiration at
the NERT site and surrounding properties: prepared for Nevada Division of
Environmental Protection,
[report (PDF)](https://nertjoomla3.azurewebsites.net/index.php/project-documents/access-project-documents/file/NDEP%20Communications/2023/2023-07%20Remote%20Sensing%20of%20Evapotranspiration%20at%20the%20NERT%20Site%20and%20Surrounding%20Properties_OCR.pdf).

Ma, L., Condon, L.E., Behrens, D., and Maxwell, R.M., 2025, High resolution
water table depth estimates across the contiguous United States: Water Resources
Research, v. 61, no. 2, e2024WR038658,
[doi:10.1029/2024WR038658](https://doi.org/10.1029/2024WR038658).
