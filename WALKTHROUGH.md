# ETg Baseline Fill — Walkthrough

This guide walks through the full workflow for a new basin, from environment
setup through the initial run, reviewing output, and re-running with an
expert adjustment. The example uses Pine Valley, NV, but the process is
the same for any of the 257 NWI investigation basins.


## 1. Set up the environment

Create the conda environment from the repository's environment file and
activate it:

    conda env create -f environment.yml
    conda activate etg_fill


## 2. Run statewide covariate prep (one time only)

Before processing any basin you need Nevada-extent subsets of the
covariate rasters. This step clips CONUS-scale rasters to the dissolved
NWI boundary with a 10 km buffer.  `--dem` is optional:

    python prep_statewide.py \
        --bps /path/to/BpS_CONUS.tif \
        --wtd /path/to/WTD_CONUS.tif
        # optional:
        # --dem /path/to/DEM_CONUS.tif

This produces two (or three, if `--dem` was supplied) files in `statewide/`:

    statewide/
        BpS_statewide.tif
        WTD_statewide.tif
        DEM_statewide.tif   (only if --dem was supplied)

You only run this once. Every per-basin prep clips from these statewide
subsets rather than going back to the CONUS files. (HAND is derived
per-basin from the clipped DEM during step 4, not at the statewide level.)

**DEM handling.** If `DEM_statewide.tif` is not built, `prep_basin.py` and
`prep_custom_basin.py` fall back to downloading a per-basin Copernicus
GLO-30 (COP30) DEM tile directly from OpenTopography's Global DEM API.
This avoids memory issues on Nevada-scale py3dep requests and keeps each
basin's DEM independent.  A project-default API key is embedded in
`opentopo.py`; override it by setting the `OPENTOPOGRAPHY_API_KEY`
environment variable (free registration at
https://portal.opentopography.org).


## 3. Find your basin key

The NWI shapefile contains 257 hydrographic areas. Each has a basin key
that combines the HA number and name (e.g. `053_PineValley`).
List all available keys:

    python prep_basin.py --list

Find the key that matches your basin and note the exact string.


## 4. Prep the basin

Run the per-basin prep, substituting your basin key:

    python prep_basin.py 053_PineValley

This creates the basin directory structure, clips covariates from the
statewide subsets, and derives HAND from the basin's clipped DEM:

    basins/053_PineValley/
        source/              (empty -- you drop your raws here)
        input/
            DEM.tif          (prep-generated)
            BpS.tif
            WTD.tif
            HAND.tif
        output/
        config.toml

The `source/` directory holds user-supplied raws (ETg raster, treatment
shapefile, optional boundary).  `input/` is reserved for prep-generated
clipped covariates -- don't mix raws into it.

HAND derivation uses whitebox-tools (FillDepressions, D8 flow
accumulation, stream extraction, ElevationAboveStream) and typically
takes 10-30 seconds per basin. If HAND derivation fails for a basin in
a batch run, the basin continues without HAND rather than aborting.

Optional flags for HAND control:

    python prep_basin.py 053_PineValley --skip-hand
    python prep_basin.py 053_PineValley --hand-threshold 500


## 5. Place your data files

Copy your ETg raster and treatment shapefile (including all sidecar files:
`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`) into the `source/` folder:

    basins/053_PineValley/
        source/
            your_ETg_raster.tif  (you provide this)
            your_treatment.shp   (you provide this)
            your_treatment.shx
            your_treatment.dbf
            your_treatment.prj
            your_treatment.cpg
        input/
            DEM.tif              (from prep)
            BpS.tif              (from prep)
            WTD.tif              (from prep)
            HAND.tif             (from prep -- derived per-basin)

Keep `input/` for the prep-generated covariates only; `source/` is for
anything you supply.  If you have optional gSSURGO soil covariates
(`AWC.tif` and/or `SoilDepth.tif`), drop those into `input/` alongside
the other prep outputs -- they're treated as covariates, not raws.


## 6. Review and edit config.toml

Open `basins/053_PineValley/config.toml` in a text editor. The
prep script tries to auto-detect your filenames, but verify the `[source]`
section points to the correct files (paths resolve against `source/`):

    [source]
    etg_tif       = "your_ETg_raster.tif"
    treatment_shp = "your_treatment.shp"
    # boundary_shp  = "boundary.shp"      # optional, for non-NWI areas

The `[inputs]` section points at the prep-generated covariates (relative
to `input/`) and rarely needs editing.  The default parameters are
reasonable starting points. Key settings you may want to review:

    [treatment]
    buffer_m         = 90.0        # buffer around treatment polygons (meters)
    feather_width_px = 4           # Gaussian blend width outside boundary (pixels)

    [adjustment]
    baseline_adjust  = 1.0         # expert adjustment knob (1.0 = no change)

    [model]
    use_wtd          = true        # set false if WTD product is unreliable here
    use_hand         = true        # set false if HAND derivation was unreliable
    use_soil         = true        # drops gSSURGO AWC + SoilDepth if false
    backend          = "lgbm"      # "lgbm" or "rf"
    max_slope_deg    = 5.0         # exclude steep pixels from training

Leave `baseline_adjust = 1.0` for your first run.


## 7. Run the fill

    python etg_baseline_fill.py 053_PineValley

The script logs each step to the console and writes a detailed log file
to `output/`. A typical run on a medium-sized basin takes 30-120 seconds
depending on the number of pixels.

When running LightGBM with at least 200 training pixels, the model uses
early stopping: a 20% validation hold-out monitors loss and stops
adding trees once performance plateaus (20 rounds with no improvement).
The model is then refitted on the full training set using the optimal
tree count, so no data is wasted. The log reports how many trees were
used versus the configured maximum.

If 3-fold cross-validation R-squared is negative (meaning the terrain
residual model is making predictions worse, not better), the residual
is automatically zeroed out and the baseline falls back to spatially
weighted per-BpS class means.  Instead of a single flat rate per
vegetation class, each treatment pixel draws from nearby training
pixels of the same class using a Gaussian window (~1 km default),
producing gradients within each class.  The window radius is
configurable via `spatial_fallback_radius_px` in config.toml (set to 0
to revert to flat basin-wide class averages).  A warning is logged and
the run metadata records the fallback method used.


## 8. Run diagnostics and summary

    python diagnostics.py 053_PineValley
    python etunit_summary.py 053_PineValley

The fill-script log also prints two headline numbers before finishing:
`Total ETg volume change vs original input` (basin-wide, over all pixels
valid in both rasters) and `Treatment-zone ETg volume change` (restricted
to treatment pixels).  These quantify how much of the Landsat ETg signal
was replaced by the modeled baseline — useful when the outputs feed into
a basin water budget or a Net-ET decomposition (`ETa_applied = ETa − ETg`).

All output lands in `basins/053_PineValley/output/`. Key files
to review:

- `*_ETg_final.tif` — the corrected ETg raster
- `*_ETg_pct_change.tif` — per-pixel percent change from raw to final
- `*_polygon_summary.csv` — per-polygon statistics (pixel count, mean
  input/baseline/final ETg, adjustment factor)
- `*_diag_*.png` — diagnostic plots (histograms, scatter, BpS boxplots,
  maps, feather weights)
- `*_ETUNIT_SUMMARY.csv` — ET-unit-level area, volume, and rate with
  Low/High bounds
- `*_run_metadata.txt` — full record of configuration, training stats,
  feature importances, and results

Open the final raster and diagnostic plots in QGIS or your preferred
viewer.


## 9. Review and identify issues

After the first run, open the polygon summary in your editor or QGIS:

    basins/053_PineValley/output/053_PineValley_polygon_summary.csv

The columns to focus on are:

| Column | Meaning |
|--------|---------|
| `polygon_id` | Identifier from the treatment shapefile |
| `n_pixels` | Pixel count inside the polygon (30 m pixels, so 1,089 px ~ 1 km^2) |
| `treatment` | `replaced` (treated and modeled) or `none` (left at raw ETg) |
| `adj_factor` | Adjustment that was applied (blank if untreated, 1.0 if no override) |
| `mean_input_ETg` | Raw Landsat-derived ETg, including irrigation signal |
| `mean_baseline_ETg` | Model-predicted natural baseline (what replaces the irrigated value) |
| `mean_final_ETg` | What lands in the final raster after feathering at edges |

Compare `mean_baseline_ETg` against field knowledge or independent estimates
for the vegetation class and water-table setting of each polygon. The
diagnostic boxplots (`*_diag_bps_boxplots.png`) and scatter
(`*_diag_scatter.png`) help you see the model's behaviour by BpS class
and flag systematic over- or under-prediction.

Common reasons a polygon's baseline may need an expert override:

- Valley-bottom training bias: most native phreatophyte / meadow pixels
  in the basin sit inside irrigated polygons, so the model trained on
  hillslope analogs and under-predicts native ETg in the valley floor.
- A BpS class with very few non-irrigated training pixels in the basin.
- Local water-table information not captured in the WTD product
  (e.g., a known perched aquifer).
- Recent observed change that the long-term covariates don't reflect.

If the whole basin reads systematically low or high, use the basin-wide
knob in `config.toml`. If only a few polygons need correction, use the
per-polygon column in the shapefile. The two approaches combine cleanly
(see "Combining both approaches" below).

Before re-running, copy the existing `output/` directory to
`output/before_adjustment/` or similar so you can diff the two runs.
The re-run overwrites in place.


### Option A: Adjust a single polygon via the shapefile

Use this when one or a few polygons need correction but the rest of the
basin looks reasonable.

Open the treatment shapefile (the one named in `[source] treatment_shp`
in `config.toml`) in QGIS or ArcGIS Pro. Start an edit session.

Add a new field if it does not already exist. The default column name
is `adj_fctr`; you can rename it via `attr_adjust` in
`config.toml [adjustment]` if you prefer.

    Field name: adj_fctr
    Type:       Double / float
    Width:      10
    Precision:  4

For each polygon you want to adjust, enter the multiplicative factor
that converts the modeled baseline to your target rate:

    adj_fctr = target_baseline / modeled_baseline

**Worked example (Pine Valley, polygon 22).** The first run produced:

    polygon_id  n_pixels  mean_input_ETg  mean_baseline_ETg
    22          812       1.654           0.624

Suppose your field judgment for that part of the valley puts the
native baseline closer to **0.85 ft/yr** based on meadow phreatophyte
analogs and water-table depth. Then:

    adj_fctr = 0.85 / 0.624 = 1.362

Round to a sensible precision and enter `1.36` (or `1.37`) on polygon
22. Leave every other polygon's `adj_fctr` as NULL or 0. Save the
edits, exit the edit session, and re-run:

    python etg_baseline_fill.py 053_PineValley

**Important rule about the column.** Only values **greater than 0**
are treated as overrides. NULL, 0, and negative numbers all fall
through to the basin-wide default. If you specifically want to force
a polygon back to 1.0 when the basin default is non-1.0, you must
enter `1.0` explicitly; a blank cell will inherit the basin default.

The override is one factor per polygon. The value gets burned into
every pixel inside the polygon. There is no sub-polygon variation.


### Option B: Adjust the entire basin via config.toml

Use this when the model systematically under- or over-predicts across
the whole basin and the offset is roughly uniform.

Open `basins/053_PineValley/config.toml`. If there is no
`[adjustment]` section, add one. The minimal block:

    [adjustment]
    baseline_adjust = 1.15
    # attr_adjust   = "adj_fctr"   # optional, only needed if you renamed the column

`baseline_adjust = 1.15` multiplies the modeled baseline by 1.15 for
every treated polygon in the basin (a 15% increase). The default is
`1.0` (no change). Values less than 1.0 reduce the baseline; values
greater than 1.0 raise it. Re-run:

    python etg_baseline_fill.py 053_PineValley


### Combining both approaches

The two paths stack cleanly. The fill script first rasterizes a basin
default layer from `baseline_adjust`, then burns the per-polygon
`adj_fctr` raster on top wherever the column value is greater than 0.
Per-polygon values fully replace the basin default for that polygon's
pixels (they are not multiplied).

Example: `baseline_adjust = 1.10` in `config.toml`, and polygon 22
has `adj_fctr = 1.36` in the shapefile. The result is:

- Polygon 22: factor = 1.36 (override wins)
- Every other treated polygon: factor = 1.10 (basin default)

If you want polygon 22 to receive 1.36 *on top of* the 1.10 basin
nudge, enter `1.10 * 1.36 = 1.496` in the shapefile directly. The
script does not compose them automatically.


## 10. Verify the adjustment

Three places to check, in order of speed.

**Run log.** When adjustment is active you will see lines like:

    2b · Rasterizing per-polygon adjustment factors (column 'adj_fctr') …
        per-polygon overrides applied to 812 pixels
        expert adjustment ACTIVE — basin default: 1.0
        adjustment factors in treatment zone — min: 1.000  max: 1.360  mean: 1.028

The pixel count should match the size of the polygons you flagged
(in the worked example, polygon 22 had 812 pixels, and that is what
the override pixel count should report). If the override count is 0,
the column name does not match `attr_adjust` or every value was 0 /
NULL / negative.

**Polygon summary CSV.** The `adj_factor` column shows what each
polygon actually received. For polygon 22, after the adjustment:

    polygon_id  n_pixels  treatment  adj_factor  mean_input  mean_baseline  mean_final
    22          812       replaced   1.36        1.654       0.624          ~0.849

The `mean_baseline_ETg` value does not change (it is still the raw
model prediction); the adjustment is applied downstream when the
baseline is burned into the treatment pixels, so the effect shows up
in `mean_final_ETg`.

**Run metadata.** `output/053_PineValley_run_metadata.txt` records
the basin-wide `baseline_adjust` value and whether per-polygon
overrides were active. This is the file to point at in a project
record of why the basin's output looks the way it does.

For a visual confirmation, open `*_ETg_pct_change.tif` in QGIS and
symbolize on a divergent ramp (e.g., blue for increases, red for
decreases). The polygons you adjusted should pop out cleanly against
the rest of the basin.


### Document your reasoning

A multiplicative override changes the output without leaving any
record of *why*. For project archives and review, write a short note
alongside the basin output directory describing each override:

    basins/053_PineValley/output/ADJUSTMENT_NOTES.md

    Polygon 22: model baseline 0.62 ft/yr; field judgment 0.85 ft/yr
    based on meadow phreatophyte analogs and 1.5 m WTD; adj_fctr = 1.36.
    Date: 2026-04-21. Reviewer: M. Bromley.

This file is not read by any script, but it makes future audits much
easier.


## Running multiple basins

To process several basins in batch, use the orchestrator:

    # Run all basins that have a config.toml:
    python run_all.py

    # Preview what would run without executing:
    python run_all.py --dry-run

    # Skip diagnostics to speed up a batch:
    python run_all.py --skip-diag

The orchestrator runs prep (if needed), fill, diagnostics, and summary
for each basin in sequence. Basins with too few training pixels are
gracefully skipped with a marker file. After the batch completes, a
`cross_basin_summary.csv` is written to the project root for QC across
all basins.


## Custom study areas (outside NWI)

The NWI-based workflow (steps 2-10 above) assumes your basin exists in
the NWI investigation shapefile and that statewide covariates have been
prepared. For study areas outside Nevada, outside the NWI framework, or
in other regions entirely, `prep_custom_basin.py` replaces steps 2-4
in a single command. You do not need the NWI shapefile or
`prep_statewide.py`.


### What you need

Before running, gather these files:

    1. A boundary polygon for the study area -- shapefile, GeoJSON, or
       GeoPackage.  This defines the area that training pixels are drawn
       from.  It does not need to be in the same CRS as the rasters.

    2. (Optional) A DEM raster that covers (at least) the study area plus
       a few km of buffer.  CONUS-wide sources like USGS 3DEP work, as
       does any local DEM in GeoTIFF format.  If `--dem` is omitted,
       prep_custom_basin.py downloads a COP30 tile from OpenTopography.

    3. A LANDFIRE BpS raster that covers the study area.  The CONUS-wide
       LF2020_BPS raster works, or a regional extract.

    4. (Optional) A water-table depth raster.  If omitted, the model
       trains without WTD and config.toml sets use_wtd = false.

    5. Your ETg raster and treatment shapefile (same as for NWI basins).


### Prep the basin

Run the custom prep script with your boundary and covariate sources
(`--dem` is optional; if omitted, a COP30 tile is pulled from
OpenTopography):

    python prep_custom_basin.py SierraValley \
        --boundary  "E:\data\sierra_valley_boundary.shp" \
        --bps       "E:\data\LF2020_BPS_CONUS.tif" \
        --wtd       "E:\data\wtd_conus.tif"
        # optional: --dem "E:\data\3DEP_CONUS_30m.tif"

This does the following in one pass:

    - Reads the boundary polygon and determines the output CRS.  If
      the boundary is in a projected CRS (e.g. UTM), that CRS is used
      for all basin rasters.  If it is in lat/lon, the script picks the
      correct UTM zone automatically from the study-area centroid.

    - Clips DEM, BpS, and WTD from their source rasters to the boundary
      (with a 5 km buffer), reprojecting into the target CRS.

    - Derives HAND from the clipped DEM using the same whitebox-tools
      pipeline as NWI basins (FillDepressions, D8, ExtractStreams,
      ElevationAboveStream).

    - Copies the boundary file into source/boundary.shp so the fill
      script can use it for the training mask.

    - Writes BpS.clr and BpS.qml for QGIS symbology (if the BpS
      class lookup has been cached from a prior prep_statewide run,
      or if the source raster carries a raster attribute table).

    - Generates config.toml with boundary_shp = "boundary.shp" in the
      [source] section and sensible defaults elsewhere.

The output directory looks the same as an NWI basin:

    basins/SierraValley/
        source/
            boundary.shp  (+ .shx, .dbf, .prj)   (from prep)
            (drop your ETg + treatment shp here)
        input/
            DEM.tif
            BpS.tif
            BpS.clr
            BpS.qml
            WTD.tif
            HAND.tif
        output/
        config.toml

Optional flags:

    --skip-hand                 Skip HAND derivation
    --hand-threshold 500        Sparser stream network for HAND
    --buffer-m 10000            Wider covariate clip buffer (meters)


### Place your data and run

From here the workflow is identical to NWI basins.  Copy your ETg
raster and treatment shapefile into source/ (alongside the boundary
shapefile the prep script already placed there), review config.toml,
and run:

    python etg_baseline_fill.py SierraValley
    python diagnostics.py SierraValley
    python etunit_summary.py SierraValley

The fill script finds `boundary_shp = "boundary.shp"` in config.toml
and uses it for the training mask -- it never looks for the NWI
shapefile.  All output rasters, metadata, diagnostics, and summaries
work exactly the same way.


### Key differences from NWI basins

    - No statewide prep step.  Covariates are clipped directly from
      their source rasters rather than from pre-built statewide subsets.

    - CRS is auto-detected.  NWI basins all use EPSG:32611 (UTM 11N).
      Custom basins use whatever CRS the boundary polygon carries, or
      an auto-selected UTM zone if the boundary is geographic.

    - Training mask uses boundary.shp.  NWI basins use the NWI
      investigation polygon.

    - run_all.py does not discover custom basins automatically.  You
      run them individually.  (A batch run across custom basins would
      require listing their keys explicitly.)


## BpS symbology in QGIS

When you prep a basin (NWI or custom), the prep scripts write `.clr`
and `.qml` sidecar files alongside `BpS.tif`. These contain the
LANDFIRE class names and colours, so when you open `BpS.tif` in QGIS
it renders with a colour-coded legend showing vegetation class names
rather than raw integer codes. This makes it much easier to cross-
reference BpS classes during basin review.

The run metadata and log also now report per-class mean ETg with full
LANDFIRE class names so you can see exactly what rate each vegetation
type received.
