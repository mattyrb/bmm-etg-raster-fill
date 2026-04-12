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
is automatically zeroed out and the baseline falls back to per-BpS
class means only. A warning is logged and the run metadata records
`use_residual_model = no`.


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
