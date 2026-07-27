#!/usr/bin/env python3
"""
plot_vaf_agreement.py

Quantifies agreement between original (full-depth) and percent-subsampled
Clair3 VAF estimates for the HIGH-tier variants tracked in
variant_vaf_comparison.csv (produced by plot_variant_counts_comparison.py).

Only rows where the original variant was actually recovered in the
subsampled call set (recovered == True) are used -- agreement metrics
are only meaningful for matched pairs.

Produces:
  vaf_agreement_stats.csv   -- Pearson r, Spearman rho, MAE, bias, limits
                                of agreement, overall and per subsampling level
  vaf_agreement_figure.png / .pdf  -- 4-panel figure:
      A. Scatter: original VAF vs. subsampled VAF (colored by level)
      B. Bland-Altman: mean(orig,sub) vs. (sub - orig), with bias + 95% LoA
      C. Mean absolute error vs. subsampling level
      D. Pearson r & Spearman rho vs. subsampling level

Usage:
  python3 plot_vaf_agreement.py --input variant_vaf_comparison.csv --out-dir .
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GRAY = "#7F7F7F"

try:
    plt.rcParams["font.family"] = "Liberation Sans"
except Exception:
    pass
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["font.size"] = 10.5

LEVEL_TICKS = [0.5, 1, 2, 5, 10, 20, 40]
LEVEL_TICK_LABELS = ["0.5", "1", "2", "5", "10", "20", "40"]


def set_level_xaxis(ax):
    ax.set_xscale("log")
    ax.set_xticks(LEVEL_TICKS)
    ax.set_xticklabels(LEVEL_TICK_LABELS)
    ax.minorticks_off()


def panel_label(ax, letter):
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")


def compute_stats(sub_df):
    """Pearson r, Spearman rho, MAE, bias, SD of diff, 95% limits of agreement."""
    orig = sub_df["orig_vaf"]
    sub = sub_df["sub_vaf"]
    diff = sub - orig
    n = len(sub_df)
    if n < 2:
        return {
            "n": n, "pearson_r": np.nan, "spearman_rho": np.nan,
            "mae": np.nan, "bias": np.nan, "sd_diff": np.nan,
            "loa_lower": np.nan, "loa_upper": np.nan,
        }
    pearson_r = orig.corr(sub, method="pearson")
    # Spearman = Pearson correlation of the ranks; computed manually (rather than
    # via pandas' method="spearman", which shells out to scipy) so this script has
    # no hard scipy dependency.
    spearman_rho = orig.rank().corr(sub.rank(), method="pearson")
    mae = diff.abs().mean()
    bias = diff.mean()
    sd_diff = diff.std(ddof=1)
    return {
        "n": n,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "mae": mae,
        "bias": bias,
        "sd_diff": sd_diff,
        "loa_lower": bias - 1.96 * sd_diff,
        "loa_upper": bias + 1.96 * sd_diff,
    }


def make_figure(rec, per_level_stats, overall_stats, out_prefix):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    levels = sorted(rec["level_numeric"].dropna().unique())
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(levels), vmax=max(levels))

    # --- Panel A: scatter, orig VAF vs sub VAF, colored by level ---
    for lvl in levels:
        sub = rec[rec["level_numeric"] == lvl]
        axA.scatter(sub["orig_vaf"], sub["sub_vaf"], color=cmap(norm(lvl)),
                    s=24, alpha=0.75, edgecolor="black", linewidth=0.3, label=f"{lvl:g}%")
    lims = [0, 100]
    axA.plot(lims, lims, color="gray", linestyle="--", linewidth=1.0, label="y = x")
    axA.set_xlim(lims)
    axA.set_ylim(lims)
    axA.set_xlabel("Original VAF (%)")
    axA.set_ylabel("Subsampled VAF (%)")
    axA.set_title("Original vs. subsampled VAF", fontsize=10.5)
    txt = (f"Pearson r = {overall_stats['pearson_r']:.3f}\n"
           f"Spearman ρ = {overall_stats['spearman_rho']:.3f}\n"
           f"n = {overall_stats['n']}")
    axA.text(0.03, 0.97, txt, transform=axA.transAxes, va="top", ha="left",
              fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor=GRAY))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axA, fraction=0.046, pad=0.04)
    cbar.set_label("Subsampling level (%)", fontsize=8)
    panel_label(axA, "A")

    # --- Panel B: Bland-Altman ---
    mean_vaf = (rec["orig_vaf"] + rec["sub_vaf"]) / 2
    diff_vaf = rec["sub_vaf"] - rec["orig_vaf"]
    for lvl in levels:
        mask = rec["level_numeric"] == lvl
        axB.scatter(mean_vaf[mask], diff_vaf[mask], color=cmap(norm(lvl)),
                    s=24, alpha=0.75, edgecolor="black", linewidth=0.3)
    axB.axhline(overall_stats["bias"], color=VERMILLION, linestyle="-", linewidth=1.3,
                label=f"Bias = {overall_stats['bias']:.2f} pp")
    axB.axhline(overall_stats["loa_upper"], color=VERMILLION, linestyle="--", linewidth=1.0,
                label=f"95% LoA = [{overall_stats['loa_lower']:.1f}, {overall_stats['loa_upper']:.1f}]")
    axB.axhline(overall_stats["loa_lower"], color=VERMILLION, linestyle="--", linewidth=1.0)
    axB.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    axB.set_xlabel("Mean of original & subsampled VAF (%)")
    axB.set_ylabel("Subsampled − Original VAF (pp)")
    axB.set_title("Bland–Altman agreement", fontsize=10.5)
    axB.legend(fontsize=8, loc="best")
    panel_label(axB, "B")

    # --- Panel C: MAE vs level ---
    axC.plot(per_level_stats["level_numeric"], per_level_stats["mae"], "o-", color=VERMILLION,
             linewidth=2, markersize=7, markeredgecolor="black", markeredgewidth=0.5)
    set_level_xaxis(axC)
    axC.set_ylim(0, None)
    axC.grid(alpha=0.25, linewidth=0.6)
    axC.set_xlabel("Subsampling level (% of reads)")
    axC.set_ylabel("Mean absolute error (pp)")
    axC.set_title("VAF mean absolute error vs. level", fontsize=10.5)
    panel_label(axC, "C")

    # --- Panel D: Pearson r & Spearman rho vs level ---
    axD.plot(per_level_stats["level_numeric"], per_level_stats["pearson_r"], "o-", color=BLUE,
              linewidth=2, markersize=7, markeredgecolor="black", markeredgewidth=0.5, label="Pearson r")
    axD.plot(per_level_stats["level_numeric"], per_level_stats["spearman_rho"], "s-", color=ORANGE,
              linewidth=2, markersize=7, markeredgecolor="black", markeredgewidth=0.5, label="Spearman ρ")
    set_level_xaxis(axD)
    axD.grid(alpha=0.25, linewidth=0.6)
    # Zoom to the observed range rather than 0-1, so the trend across levels is legible.
    r_min = min(per_level_stats["pearson_r"].min(), per_level_stats["spearman_rho"].min())
    axD.set_ylim(max(0, r_min - 0.08), 1.01)
    axD.set_xlabel("Subsampling level (% of reads)")
    axD.set_ylabel("Correlation with original VAF")
    axD.set_title("Correlation vs. level", fontsize=10.5)
    axD.legend(fontsize=8, loc="lower right")
    panel_label(axD, "D")

    fig.suptitle("Agreement between original and subsampled Clair3 VAF estimates\n(recovered variants only)",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{out_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Figure saved: {out_prefix}.png / {out_prefix}.pdf")


def main():
    HERE = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(HERE / "variant_vaf_comparison.csv"))
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--out-prefix", default="vaf_agreement_figure")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    rec = df[df["recovered"] == True].copy()
    print(f"Loaded {len(df)} rows ({len(rec)} recovered / matched variant-level pairs)")

    if rec.empty:
        print("No recovered variant pairs found -- nothing to compare.")
        return

    overall_stats = compute_stats(rec)

    per_level_rows = []
    for lvl, sub in rec.groupby("level_numeric"):
        s = compute_stats(sub)
        s["level_numeric"] = lvl
        per_level_rows.append(s)
    per_level_stats = pd.DataFrame(per_level_rows).sort_values("level_numeric").reset_index(drop=True)

    stats_out = pd.concat([
        pd.DataFrame([{"level_numeric": "overall", **overall_stats}]),
        per_level_stats,
    ], ignore_index=True)
    stats_csv = out_dir / "vaf_agreement_stats.csv"
    stats_out.to_csv(stats_csv, index=False)
    print(f"✓ Stats saved: {stats_csv}")

    print("\nOverall agreement:")
    print(f"  n = {overall_stats['n']}")
    print(f"  Pearson r = {overall_stats['pearson_r']:.4f}")
    print(f"  Spearman rho = {overall_stats['spearman_rho']:.4f}")
    print(f"  MAE = {overall_stats['mae']:.3f} pp")
    print(f"  Bias (mean signed diff) = {overall_stats['bias']:.3f} pp")
    print(f"  95% limits of agreement = [{overall_stats['loa_lower']:.2f}, {overall_stats['loa_upper']:.2f}] pp")

    print("\nPer-level agreement:")
    print(per_level_stats[["level_numeric", "n", "pearson_r", "spearman_rho", "mae", "bias"]].to_string(index=False))

    make_figure(rec, per_level_stats, overall_stats, str(out_dir / args.out_prefix))


if __name__ == "__main__":
    main()
