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

Suppose you examine the polygon summary and see that polygon `PV_042`
has a modeled baseline of 0.15 ft/yr, but field data or professional
judgment suggests the natural rate should be closer to 0.20 ft/yr. That
is about 33% higher than the model predicted.

You have two options for adjusting the results.


### Option A: Adjust a single polygon via the shapefile

This is the right approach when one or a few polygons need correction
but the rest of the basin looks reasonable.

Open the treatment shapefile in QGIS or ArcGIS Pro. Add a new field
called `adj_fctr` (type: double). For the polygon you want to adjust,
enter the multiplicative factor. In this example, to push 0.15 up to
roughly 0.20:

    adj_fctr = 1.33

Leave all other polygons as 0 or NULL. They will use the basin-wide
default of 1.0 (no change).

Save the shapefile, then re-run:

    python etg_baseline_fill.py 053_PineValley


### Option B: Adjust the entire basin via config.toml

This is the right approach when the model systematically under- or
over-predicts across the whole basin.

Edit `config.toml`:

    [adjustment]
    baseline_adjust = 1.15

This multiplies the modeled baseline by 1.15 for every treatment
polygon in the basin (a 15% increase). Re-run:

    python etg_baseline_fill.py 053_PineValley


### Combining both approaches

You can set a basin-wide adjustment in `config.toml` and still override
individual polygons via the `adj_fctr` column. Where a polygon has
`adj_fctr > 0` in the shapefile, that value is used instead of the
basin-wide default.

For example, with `baseline_adjust = 1.1` in the config and
`adj_fctr = 1.33` on polygon `PV_042`, most polygons get a 10% boost
while `PV_042` gets 33%.


## 10. Verify the adjustment

After re-running, check the log output. It will confirm the adjustment
was applied:

    expert adjustment ACTIVE — basin default: 1.15
    per-polygon overrides applied to 342 pixels
    adjustment factors in treatment zone — min: 1.150  max: 1.330  mean: 1.158

The polygon summary CSV includes an `adj_factor` column so you can
verify each polygon received the intended value. The run metadata file
also records whether adjustments were active and whether per-polygon
overrides were used.


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
