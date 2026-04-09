#!/usr/bin/env python3
"""
etunit_summary.py
=================
Produce an ET-unit-level summary CSV from the modeled ETg_final raster,
grouped by the ``ET_unit`` attribute in the treatment shapefile.

Replicates the column structure of the original BMM ETg summary CSV:

    ET Unit, Area (ac), ETg Volume (acft), ETg Volume Low (acft),
    ETg Volume High (acft), ETg Rate (ft), ETg Rate Low (ft), ETg Rate High (ft)

Uncertainty bounds (Low / High) are derived from the pixel-level distribution
within each ET unit: mean +/- 1 standard deviation, floored at 0.  This
replaces the hand-assigned confidence bounds from the legacy workflow with a
data-driven uncertainty envelope.

Usage
-----
    cd <project folder>
    python etg_fill/etunit_summary.py

All paths are read from ``etg_fill/config.py``.

License: MIT (see LICENSE)
"""

import csv
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np

# ── Make sure we can import config regardless of cwd ─────────────────────────
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
cfg = None  # set by main()

try:
    import rasterio
    from rasterio.features import rasterize
    import geopandas as gpd
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nActivate the etg_fill conda env first.")


# ── Display-name mapping ────────────────────────────────────────────────────
# The original summary CSV used presentation labels; map from shapefile values.
ET_UNIT_DISPLAY = {
    "Cropland":               "Irrigated Cropland",
    "Meadow":                 "Meadow",
    "Other":                  "Other",
    "Pasture":                "Pasture",
    "Phreatophyte":           "Phreatophyte Shrubland",
    "Phreatophyte Shrubland": "Phreatophyte Shrubland",
    "Riparian":               "Riparian",
    "Wetland":                "Wetland",
}

# Row ordering to match original summary
ET_UNIT_ORDER = [
    "Irrigated Cropland",
    "Meadow",
    "Other",
    "Pasture",
    "Phreatophyte Shrubland",
    "Riparian",
    "Wetland",
]


def _log(msg: str):
    print(msg, flush=True)


def _load_cfg(study_area: str):
    """Load basin_config (TOML) or legacy config depending on study area."""
    global cfg
    basins_dir = _here.parent / "basins"
    toml_path = basins_dir / study_area / "config.toml"
    if toml_path.exists():
        import basin_config as _cfg
        _cfg.load_basin(study_area)
        cfg = _cfg
    else:
        import config as _cfg
        _cfg.load_study_area(study_area)
        cfg = _cfg


def main(study_area: str | None = None):
    # ── Resolve study area ──────────────────────────────────────────────────
    if study_area is None:
        if len(sys.argv) > 1:
            study_area = sys.argv[1]
        else:
            import basin_config as _bc
            import config as _legacy
            all_areas = sorted(set(_bc.available_areas()) |
                               set(_legacy.available_areas()))
            sys.exit(
                f"Usage: python etg_fill/etunit_summary.py <study_area>\n"
                f"  Available: {', '.join(all_areas)}"
            )
    _load_cfg(study_area)
    print(f"Study area: {cfg.STUDY_AREA_NAME}")

    out_dir = cfg.OUT_DIR
    sa = cfg.STUDY_AREA_NAME
    etg_final_path = out_dir / f"{sa}_ETg_final.tif"

    if not etg_final_path.exists():
        sys.exit(f"{etg_final_path.name} not found in {out_dir}. "
                 f"Run: python etg_fill/etg_baseline_fill.py {sa}")

    # ── 1. Read the final ETg raster ─────────────────────────────────────────
    _log("Reading ETg_final raster …")
    with rasterio.open(etg_final_path) as src:
        etg = src.read(1).astype(np.float32)
        transform = src.transform
        grid_shape = etg.shape
        crs = src.crs
        pixel_w = abs(transform.a)   # pixel width  in CRS units (metres)
        pixel_h = abs(transform.e)   # pixel height in CRS units (metres)

    # Area of one pixel in acres (CRS should be in metres for UTM / Albers)
    m2_per_acre = 4046.8564224
    pixel_area_m2 = pixel_w * pixel_h
    pixel_area_ac = pixel_area_m2 / m2_per_acre
    _log(f"  Pixel size: {pixel_w:.1f} × {pixel_h:.1f} m  →  {pixel_area_ac:.6f} ac")

    # ── 2. Read the treatment shapefile ──────────────────────────────────────
    _log("Reading treatment shapefile …")
    gdf = gpd.read_file(cfg.TREATMENT_SHP)
    if not gdf.crs.equals(crs):
        gdf = gdf.to_crs(crs)

    if "ET_unit" not in gdf.columns:
        sys.exit("Shapefile is missing 'ET_unit' column.")

    # Normalize casing so "meadow" and "Meadow" merge into one group.
    # Title-case first, then apply the display-name mapping.
    gdf["ET_unit"] = gdf["ET_unit"].str.strip().str.title()

    # ── 3. Rasterize each ET unit ────────────────────────────────────────────
    _log("Rasterizing ET units …")
    unique_units = sorted(gdf["ET_unit"].dropna().unique())
    _log(f"  Found {len(unique_units)} ET units: {unique_units}")

    results = OrderedDict()

    for unit in unique_units:
        subset = gdf[gdf["ET_unit"] == unit]
        geoms = [(g, 1) for g in subset.geometry if g is not None]
        if not geoms:
            continue

        mask = rasterize(
            geoms,
            out_shape=grid_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        px = (mask == 1) & np.isfinite(etg)
        vals = etg[px]
        n_px = int(px.sum())

        if n_px == 0:
            _log(f"  {unit}: 0 valid pixels — skipping")
            continue

        area_ac     = n_px * pixel_area_ac
        mean_rate   = float(np.mean(vals))
        std_rate    = float(np.std(vals))
        volume_acft = float(np.sum(vals)) * pixel_area_ac   # sum(rate_ft) × area_per_pixel

        # Uncertainty: mean ± 1 SD, bounded ≥ 0
        rate_low = max(0.0, mean_rate - std_rate)
        rate_high = mean_rate + std_rate
        vol_low  = max(0.0, rate_low * area_ac)
        vol_high  = rate_high * area_ac

        display = ET_UNIT_DISPLAY.get(unit, unit)
        results[display] = {
            "ET Unit":               display,
            "Area (ac)":             area_ac,
            "ETg Volume (acft)":     volume_acft,
            "ETg Volume Low (acft)": vol_low,
            "ETg Volume High (acft)": vol_high,
            "ETg Rate (ft)":         mean_rate,
            "ETg Rate Low (ft)":     rate_low,
            "ETg Rate High (ft)":     rate_high,
        }
        _log(f"  {display:30s}  pixels={n_px:>8,}  area={area_ac:>10.2f} ac  "
             f"rate={mean_rate:.4f} ft  vol={volume_acft:>10.2f} acft")

    # ── 4. Build ordered output with totals row ──────────────────────────────
    rows = []
    for label in ET_UNIT_ORDER:
        if label in results:
            rows.append(results[label])

    # Any units not in the predefined order (shouldn't happen, but safety net)
    for label, rec in results.items():
        if label not in ET_UNIT_ORDER:
            rows.append(rec)

    # Totals row
    total_area = sum(r["Area (ac)"]             for r in rows)
    total_vol  = sum(r["ETg Volume (acft)"]     for r in rows)
    total_lci  = sum(r["ETg Volume Low (acft)"] for r in rows)
    total_uci  = sum(r["ETg Volume High (acft)"] for r in rows)
    rows.append({
        "ET Unit":               "",
        "Area (ac)":             total_area,
        "ETg Volume (acft)":     total_vol,
        "ETg Volume Low (acft)": total_lci,
        "ETg Volume High (acft)": total_uci,
        "ETg Rate (ft)":         "",
        "ETg Rate Low (ft)":     "",
        "ETg Rate High (ft)":     "",
    })

    # ── 5. Write CSV ─────────────────────────────────────────────────────────
    fieldnames = [
        "ET Unit", "Area (ac)",
        "ETg Volume (acft)", "ETg Volume Low (acft)", "ETg Volume High (acft)",
        "ETg Rate (ft)", "ETg Rate Low (ft)", "ETg Rate High (ft)",
    ]

    csv_path = out_dir / f"{cfg.STUDY_AREA_NAME}_ETUNIT_SUMMARY.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _log(f"\nWrote {csv_path}")
    _log(f"  {len(rows) - 1} ET units + 1 totals row")

    # ── 6. Print comparison table ────────────────────────────────────────────
    _log("\n── Modeled ET Unit Summary ──────────────────────────────────────")
    _log(f"  {'ET Unit':30s} {'Area (ac)':>12s} {'Vol (acft)':>12s} "
         f"{'Rate (ft)':>10s}  {'Low':>8s}  {'High':>8s}")
    _log("  " + "─" * 86)
    for r in rows:
        lbl = r["ET Unit"] if r["ET Unit"] else "TOTAL"
        rate_str = f"{r['ETg Rate (ft)']:.4f}" if isinstance(r["ETg Rate (ft)"], float) else ""
        lci_str  = f"{r['ETg Rate Low (ft)']:.4f}" if isinstance(r["ETg Rate Low (ft)"], float) else ""
        uci_str  = f"{r['ETg Rate High (ft)']:.4f}" if isinstance(r["ETg Rate High (ft)"], float) else ""
        _log(f"  {lbl:30s} {r['Area (ac)']:>12.2f} {r['ETg Volume (acft)']:>12.2f} "
             f"{rate_str:>10s}  {lci_str:>8s}  {uci_str:>8s}")


if __name__ == "__main__":
    main()
