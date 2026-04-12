# ETg Baseline Fill — Walkthrough

This guide walks through the full workflow for a new basin, from environment
setup through the initial run, reviewing output, and re-running with an
expert adjustment. The example uses Railroad Valley, NV, but the process is
the same for any of the 257 NWI investigation basins.


## 1. Set up the environment

Create the conda environment from the repository's environment file and
activate it:

    conda env create -f environment.yml
    conda activate etg_fill


## 2. Run statewide covariate prep (one time only)

Before processing any basin you need Nevada-extent subsets of the three
covariate rasters. This step clips CONUS-scale rasters to the dissolved
NWI boundary with a 10 km buffer:

    python prep_statewide.py \
        --bps /path/to/BpS_CONUS.tif \
        --wtd /path/to/WTD_CONUS.tif \
        --dem /path/to/DEM.tif

This produces three files in `statewide/`:

    statewide/
        DEM_statewide.tif
        BpS_statewide.tif
        WTD_statewide.tif

You only run this once. Every per-basin prep clips from these statewide
subsets rather than going back to the CONUS files. (HAND is derived
per-basin from the clipped DEM during step 4, not at the statewide level.)


## 3. Find your basin key

The NWI shapefile contains 257 hydrographic areas. Each has a basin key
that combines the HA number and name (e.g. `173A_RailroadValleyNorth`).
List all available keys:

    python prep_basin.py --list

Find the key that matches your basin and note the exact string.


## 4. Prep the basin

Run the per-basin prep, substituting your basin key:

    python prep_basin.py 173A_RailroadValleyNorth

This creates the basin directory structure, clips covariates from the
statewide subsets, and derives HAND from the basin's clipped DEM:

    basins/173A_RailroadValleyNorth/
        input/
            DEM.tif
            BpS.tif
            WTD.tif
            HAND.tif
        output/
        config.toml

HAND derivation uses whitebox-tools (FillDepressions, D8 flow
accumulation, stream extraction, ElevationAboveStream) and typically
takes 10-30 seconds per basin. If HAND derivation fails for a basin in
a batch run, the basin continues without HAND rather than aborting.

Optional flags for HAND control:

    python prep_basin.py 173A_RailroadValleyNorth --skip-hand
    python prep_basin.py 173A_RailroadValleyNorth --hand-threshold 500


## 5. Place your data files

Copy your ETg raster and treatment shapefile (including all sidecar files:
`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`) into the `input/` folder:

    basins/173A_RailroadValleyNorth/input/
        DEM.tif              (from prep)
        BpS.tif              (from prep)
        WTD.tif              (from prep)
        HAND.tif             (from prep -- derived per-basin)
        your_ETg_raster.tif  (you provide this)
        your_treatment.shp   (you provide this)
        your_treatment.shx
        your_treatment.dbf
        your_treatment.prj
        your_treatment.cpg


## 6. Review and edit config.toml

Open `basins/173A_RailroadValleyNorth/config.toml` in a text editor. The
prep script tries to auto-detect your filenames, but verify the `[inputs]`
section points to the correct files:

    [inputs]
    etg_tif       = "your_ETg_raster.tif"
    treatment_shp = "your_treatment.shp"

The default parameters are reasonable starting points. Key settings you
may want to review:

    [treatment]
    buffer_m         = 90.0        # buffer around treatment polygons (metres)
    feather_width_px = 4           # Gaussian blend width outside boundary (pixels)

    [adjustment]
    baseline_adjust  = 1.0         # expert adjustment knob (1.0 = no change)

    [model]
    use_wtd          = true        # set false if WTD product is unreliable here
    use_hand         = true        # set false if HAND derivation was unreliable
    backend          = "lgbm"      # "lgbm" or "rf"
    max_slope_deg    = 5.0         # exclude steep pixels from training

Leave `baseline_adjust = 1.0` for your first run.


## 7. Run the fill

    python etg_baseline_fill.py 173A_RailroadValleyNorth

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

    python diagnostics.py 173A_RailroadValleyNorth
    python etunit_summary.py 173A_RailroadValleyNorth

All output lands in `basins/173A_RailroadValleyNorth/output/`. Key files
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

Suppose you examine the polygon summary and see that polygon `RR_042`
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

    python etg_baseline_fill.py 173A_RailroadValleyNorth


### Option B: Adjust the entire basin via config.toml

This is the right approach when the model systematically under- or
over-predicts across the whole basin.

Edit `config.toml`:

    [adjustment]
    baseline_adjust = 1.15

This multiplies the modeled baseline by 1.15 for every treatment
polygon in the basin (a 15% increase). Re-run:

    python etg_baseline_fill.py 173A_RailroadValleyNorth


### Combining both approaches

You can set a basin-wide adjustment in `config.toml` and still override
individual polygons via the `adj_fctr` column. Where a polygon has
`adj_fctr > 0` in the shapefile, that value is used instead of the
basin-wide default.

For example, with `baseline_adjust = 1.1` in the config and
`adj_fctr = 1.33` on polygon `RR_042`, most polygons get a 10% boost
while `RR_042` gets 33%.


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
the NWI investigation shapefile. For basins outside Nevada or outside
the NWI framework entirely, use `prep_custom_basin.py`:

    python prep_custom_basin.py SierraValley \
        --boundary  path/to/sierra_valley_boundary.shp \
        --dem       path/to/CONUS_DEM.tif \
        --bps       path/to/LF2020_BPS_CONUS.tif \
        --wtd       path/to/wtd_conus.tif

This replaces steps 2-4. It clips all covariates to your boundary,
derives HAND, copies the boundary into `input/boundary.shp`, and
generates a `config.toml` with `boundary_shp = "boundary.shp"` so the
fill script uses your boundary for the training mask instead of looking
for the NWI shapefile.

You do *not* need `prep_statewide.py` or the NWI shapefile for custom
basins. You just need:

- A boundary shapefile (or GeoJSON, or GeoPackage)
- A DEM raster covering (at least) your study area
- A BpS raster covering your study area
- Optionally, a WTD raster

If the boundary is in a geographic CRS (lat/lon), the script auto-picks
the appropriate UTM zone. If it's already in a projected CRS, that CRS
is used directly for all basin rasters.

From step 5 onward (place your ETg and treatment files, review
`config.toml`, run the fill), the workflow is identical to an NWI basin.


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
