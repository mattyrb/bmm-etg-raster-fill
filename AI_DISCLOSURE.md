# AI-Assisted Development Disclosure

## Overview

This codebase was developed collaboratively between DRI research staff and Claude,
an AI assistant made by Anthropic. This document describes the nature and extent
of AI involvement, in keeping with principles of scientific transparency and
reproducibility.

This tool extends the Beamer-Minor Model (BMM) ETg estimation workflow (Beamer et
al., 2013; Minor, 2019; Huntington et al., 2022) by replacing manually assigned
replacement rates with a data-driven spatial prediction. The scientific
foundation -- Landsat-based ETg estimation, ET unit delineation, and the concept of
separating irrigation-enhanced ET from natural groundwater discharge -- was
established in prior peer-reviewed work by DRI staff (see README.md, References).

## Roles

**Human (DRI research staff):**
- Defined the scientific problem and research objectives
- Specified the modeling approach (two-stage BpS + terrain residual)
- Selected input datasets and determined their appropriate use
- Made all decisions about treatment-zone classification and the decision to
  simplify from dual scale/replace to full replacement for all treatment polygons
- Chose to exclude NDVI to avoid irrigation signal leakage
- Decided to constrain training data to within-basin pixels only
- Decided feathering should occur outside the treatment boundary
- Validated outputs against domain knowledge and prior results
- Identified and corrected issues with input data (BpS raster CRS, training
  coverage, WTD projection)
- Directed all iterations and revisions to the workflow
- Reviewed and approved all code before use
- Designed the statewide scaling architecture (per-basin configs, NWI basin
  boundaries, directory structure)
- Specified the expert adjustment knob design (basin-wide default + per-polygon
  override via shapefile column) so hydrologists can tune results
- Decided to add HAND (Height Above Nearest Drainage) as a terrain covariate
  and directed the move from statewide to per-basin derivation after identifying
  the memory limitation
- Requested early stopping on LightGBM to prevent over-training on small basins
- Identified and reported each stage of the HAND derivation debugging cycle
  (memory failure, nodata barriers, algorithm speed, parameter naming)

**AI (Claude, Anthropic):**
- Implemented the Python workflow based on human-specified requirements
- Wrote raster I/O, reprojection, and rasterization routines
- Implemented the two-stage model training and prediction pipeline
- Implemented Gaussian edge feathering (inside, then revised to outside per
  human direction)
- Built the multi-basin architecture: statewide prep, per-basin prep, TOML
  config system, batch orchestrator
- Implemented the expert adjustment knob: per-pixel adjustment raster from
  basin-wide default + per-polygon shapefile overrides, applied before feathering
- Implemented basin boundary mask for training data
- Added per-basin logging, graceful skip, and cross-basin summary
- Integrated HAND covariate: whitebox-tools pipeline with elevation-wall fill
  for clipped DEMs, per-basin derivation, dynamic feature stacking
- Implemented LightGBM early stopping with validation hold-out and refit
- Implemented automatic BpS-mean-only fallback when CV R-squared is negative
- Debugged whitebox-tools integration across five iterations (memory, nodata
  barriers, algorithm selection, parameter naming, file-existence checks)
- Wrote diagnostic plotting and summary statistics code
- Debugged CRS handling issues, array broadcasting errors, and matplotlib API
  changes
- Drafted documentation and this disclosure

## Development process

The code was developed iteratively over multiple conversation sessions. The human
researcher described each requirement in domain-specific terms (e.g., "I only want
training pixels selected from valid pixels within the basin"), and the AI translated
those requirements into working Python code. The researcher tested each version on
real data, reported results and issues, and directed further changes.

Key scientific and architectural judgments made by the human researcher include:

- The decision to use BpS vegetation classes rather than NDVI as the primary
  ecological covariate
- The choice to NaN all treatment-zone pixels rather than attempt back-calculation
  from the modified raster
- The decision to simplify from a dual scale/replace treatment system to full
  baseline replacement for all treatment polygons
- The decision that feathering should blend outside the treatment boundary
  rather than inside it
- The requirement to constrain training data to within-basin pixels using the NWI
  investigation boundary
- The design of the statewide scaling approach: NWI-based basin boundaries
  extending beyond the state line, one-time statewide covariate prep, per-basin
  TOML configs that are auto-generated but editable
- The selection of buffer distances, feathering widths, slope thresholds, and
  model parameters
- The identification of a BpS raster quality issue (missing vegetation classes
  inside treatment zones) and its correction

## Reproducibility

All code is provided in source form. Per-basin `config.toml` files document every
tunable parameter for each basin. The `environment.yml` file specifies dependency
requirements. Run metadata files record the exact configuration, training
statistics, feature importances, and results for each basin. Per-basin log files
provide a timestamped audit trail. No proprietary AI APIs are called at runtime --
the AI was used only during development.

## Model and version

- AI model: Claude (Anthropic), Opus class
- Development period: 2026
- The AI has no access to the runtime environment and does not influence results
  after the code is written

## Contact

For questions about the scientific methodology, contact the DRI research team.
For questions about the code implementation, issues and pull requests are welcome
on the project repository.
