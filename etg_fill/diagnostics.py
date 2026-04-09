#!/usr/bin/env python3
"""
diagnostics.py
==============
Post-run evaluation and visual sanity checks for the ETg baseline-fill workflow.

Generates diagnostic plots and console distribution summaries:

  1. Histogram — ETg distributions (outside vs treatment, before and after).
  2. Scatter — predicted baseline vs. original ETg for treatment-zone pixels.
  3. BpS box-plots — ETg by vegetation class (outside vs treatment).
  4. Map panels — side-by-side original, baseline prediction, and final ETg.
  5. Difference map — change in treatment zones (red = reduced, blue = increased).
  6. Treatment-zone map — color-coded zones (outside / treatment).
  7. Feather weight map — blend weights and edge detail (if feathering enabled).
  8. Console summary — distribution statistics (n, mean, median, p10, p90).

Usage:
    cd <project folder>
    python etg_fill/diagnostics.py

License: MIT (see LICENSE)
"""

import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
cfg = None  # set by main()


def _read(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd = src.nodata
    if nd is not None:
        arr[arr == np.float32(nd)] = np.nan
    return arr


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
                f"Usage: python etg_fill/diagnostics.py <study_area>\n"
                f"  Available: {', '.join(all_areas)}"
            )
    _load_cfg(study_area)
    print(f"Study area: {cfg.STUDY_AREA_NAME}")

    out = cfg.OUT_DIR

    # ── Load rasters ─────────────────────────────────────────────────────────
    print("Loading rasters …")

    # Prefer the raw (unmodified) ETg raster so diagnostics compare the true
    # Landsat-derived ETg — including the irrigation-inflated signal in
    # treatment zones — against the modeled baseline and final product.
    # Fall back to the grid-matched copy in the output dir, then to the
    # burned raster if neither raw source is available.
    raw_matched = out / "ETg_raw_matched.tif"
    if raw_matched.exists():
        print("  Using raw ETg (grid-matched) as original")
        etg_orig = _read(raw_matched)
    elif cfg.ETG_RAW_TIF is not None and Path(cfg.ETG_RAW_TIF).exists():
        print("  Using raw ETg source raster as original")
        etg_orig = _read(cfg.ETG_RAW_TIF)
    else:
        print("  Raw ETg not available — using burned raster as original")
        etg_orig = _read(cfg.ETG_TIF)

    sa = cfg.STUDY_AREA_NAME
    etg_baseline = _read(out / f"{sa}_ETg_baseline_pred.tif")
    etg_final    = _read(out / f"{sa}_ETg_final.tif")
    treatment    = _read(out / "treatment_zone.tif")
    bps          = _read(out / "BpS_matched.tif").astype(np.int32)

    is_treat   = treatment == 1
    is_outside = (~is_treat) & np.isfinite(etg_orig) & (etg_orig > 0)

    # Flattened subsets
    orig_out   = etg_orig[is_outside]
    orig_treat = etg_orig[is_treat & np.isfinite(etg_orig)]
    final_treat = etg_final[is_treat & np.isfinite(etg_final)]
    base_treat  = etg_baseline[is_treat & np.isfinite(etg_baseline)]

    # ── 1. Histogram comparison ──────────────────────────────────────────────
    print("Plotting histograms …")
    fig, ax = plt.subplots(figsize=(9, 5))
    p99 = np.nanpercentile(etg_orig[np.isfinite(etg_orig)], 99)
    bins = np.linspace(0, p99, 80)
    ax.hist(orig_out,    bins=bins, alpha=0.35, density=True, label="Outside treatment")
    ax.hist(orig_treat,  bins=bins, alpha=0.35, density=True, label="Treatment zones – original")
    ax.hist(final_treat, bins=bins, alpha=0.35, density=True, label="Treatment zones – filled")
    ax.set_xlabel("ETg (ft/yr)")
    ax.set_ylabel("Density")
    ax.set_title("ETg distributions: outside vs treatment zones")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{sa}_diag_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  → {out / 'diag_histogram.png'}")

    # ── 2. Scatter: baseline vs original (treatment zones) ───────────────────
    print("Plotting scatter …")
    n_samp = min(len(orig_treat), 20_000)
    if n_samp > 0:
        rng = np.random.default_rng(42)
        si = rng.choice(len(orig_treat), size=n_samp, replace=False)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(base_treat[si], orig_treat[si], s=1, alpha=0.15, c="steelblue")
        lims = [0, np.nanpercentile(np.concatenate([orig_treat, base_treat]), 99)]
        ax.plot(lims, lims, "k--", lw=0.8, label="1:1")
        ax.set_xlabel("Predicted baseline ETg (ft/yr)")
        ax.set_ylabel("Original (raw) ETg (ft/yr)")
        ax.set_title("Treatment-zone pixels: baseline vs original")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{sa}_diag_scatter.png", dpi=150)
        plt.close(fig)
        print(f"  → {out / 'diag_scatter.png'}")

    # ── 3. Per-BpS box-plots ─────────────────────────────────────────────────
    print("Plotting BpS box-plots …")
    bps_treat_arr = bps[is_treat & (bps > 0)]
    if len(bps_treat_arr) > 0:
        uniq, counts = np.unique(bps_treat_arr, return_counts=True)
        top10 = uniq[np.argsort(-counts)[:10]]

        data_out, data_treat, labels = [], [], []
        for b in top10:
            m_out = is_outside & (bps == b)
            m_tr  = is_treat & (bps == b) & np.isfinite(etg_final)
            if m_out.sum() > 10:
                data_out.append(etg_orig[m_out])
                data_treat.append(etg_final[m_tr] if m_tr.sum() > 0
                                  else np.array([np.nan]))
                labels.append(str(b))

        if labels:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            axes[0].boxplot(data_out, tick_labels=labels, showfliers=False)
            axes[0].set_title("Outside-treatment ETg")
            axes[0].set_xlabel("BpS code"); axes[0].set_ylabel("ETg (ft/yr)")
            axes[1].boxplot(data_treat, tick_labels=labels, showfliers=False)
            axes[1].set_title("Treatment-zone ETg (after fill)")
            axes[1].set_xlabel("BpS code")
            fig.tight_layout()
            fig.savefig(out / f"{sa}_diag_bps_boxplots.png", dpi=150)
            plt.close(fig)
            print(f"  → {out / 'diag_bps_boxplots.png'}")

    # ── 4. Map panels ────────────────────────────────────────────────────────
    print("Plotting map panels …")
    vmin, vmax = 0, np.nanpercentile(etg_orig[np.isfinite(etg_orig)], 98)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    titles = ["Original ETg", "Baseline prediction", "Final ETg"]
    arrays = [etg_orig, etg_baseline, etg_final]
    for ax, arr, title in zip(axes, arrays, titles):
        im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap="YlGnBu")
        ax.set_title(title, fontsize=14, pad=10)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])  # room at top for titles, bottom for colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal",
                 fraction=0.04, pad=0.04, shrink=0.5,
                 label="ETg (ft/yr)")
    fig.savefig(out / f"{sa}_diag_map_panels.png", dpi=150)
    plt.close(fig)
    print(f"  → {out / 'diag_map_panels.png'}")

    # ── 5. Difference map ────────────────────────────────────────────────────
    print("Plotting difference map …")
    diff = etg_orig - etg_final
    diff[treatment == 0] = np.nan

    fig, ax = plt.subplots(figsize=(8, 7))
    fin = diff[np.isfinite(diff)]
    vabs = np.nanpercentile(np.abs(fin), 98) if len(fin) > 0 else 1
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    ax.set_title("ETg change in treatment zones (original − final)\n"
                 "Red = reduced  |  Blue = increased")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.6, label="ΔETg (ft/yr)")
    fig.tight_layout()
    fig.savefig(out / f"{sa}_diag_difference_map.png", dpi=150)
    plt.close(fig)
    print(f"  → {out / 'diag_difference_map.png'}")

    # ── 6. Treatment-zone map ────────────────────────────────────────────────
    print("Plotting treatment-zone map …")
    tmap = np.full_like(etg_orig, np.nan)
    tmap[is_outside] = 0   # outside
    tmap[is_treat]   = 1   # treatment

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#d9d9d9", "#2166ac"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(tmap, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Outside", "Treatment"])
    ax.set_title("Treatment zones")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / f"{sa}_diag_treatment_map.png", dpi=150)
    plt.close(fig)
    print(f"  → {out / 'diag_treatment_map.png'}")

    # ── 7. Feather weight map (if feathering was used) ───────────────────────
    feather_path = out / "feather_weight.tif"
    if feather_path.exists():
        print("Plotting feather weight map …")
        fweight = _read(feather_path)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: feather weight
        im0 = axes[0].imshow(fweight, vmin=0, vmax=1, cmap="magma")
        axes[0].set_title("Feather blend weight\n(1 = 100% baseline, 0 = 100% raw ETg)")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], shrink=0.6, label="Weight")

        # Right: zoomed ETg comparison – original vs final for a treatment area
        # Show the full difference map with feathered edges visible
        diff_f = etg_orig - etg_final
        diff_f[treatment == 0] = np.nan
        fin = diff_f[np.isfinite(diff_f)]
        vabs = np.nanpercentile(np.abs(fin), 98) if len(fin) > 0 else 1
        im1 = axes[1].imshow(diff_f, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        axes[1].set_title("ΔETg with feathered edges\n(compare to diag_difference_map)")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], shrink=0.6, label="ΔETg (ft/yr)")

        fig.tight_layout()
        fig.savefig(out / f"{sa}_diag_feather_map.png", dpi=150)
        plt.close(fig)
        print(f"  → {out / 'diag_feather_map.png'}")

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n── Distribution summary ──────────────────────────────")
    for label, arr in [
        ("Outside treatment", orig_out),
        ("Treatment zones – original", orig_treat),
        ("Treatment zones – baseline pred", base_treat),
        ("Treatment zones – final", final_treat),
    ]:
        a = arr[np.isfinite(arr)] if len(arr) > 0 else arr
        if len(a) > 0:
            print(f"  {label:35s}  n={len(a):>8,}  "
                  f"mean={np.mean(a):.3f}  med={np.median(a):.3f}  "
                  f"p10={np.percentile(a,10):.3f}  p90={np.percentile(a,90):.3f}")

    print("\nDone – diagnostic plots saved to:", out.resolve())


if __name__ == "__main__":
    main()
