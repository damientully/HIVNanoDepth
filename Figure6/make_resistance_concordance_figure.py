#!/usr/bin/env python3
"""
make_resistance_concordance_figure.py

Publication figure summarizing drug-resistance-calling concordance between
subsampled and original full-depth data, using the outputs of
compare_resistance_subsampling.py:
  pooled_concordance_by_level.csv, sdrm_concordance_by_level.csv,
  discordant_calls.csv, plus the original ONTreads_sequenceSummaries.csv
  (for per-sample read depth).

Panels:
  A. Overall exact & binary (susceptible vs. reduced) drug-level agreement vs. level
  B. No-call rate vs. level (fraction of drug calls where subsampling failed
     to produce any resistance-level call at all)
  C. SDRM recall by class (PI/NRTI/NNRTI) vs. level

Usage:
  python3 make_resistance_concordance_figure.py \
      --pooled pooled_concordance_by_level.csv \
      --sdrm sdrm_concordance_by_level.csv \
      --out-dir . --out-prefix resistance_concordance_figure
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

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
CLASS_COLORS = {"PI SDRMs": VERMILLION, "NRTI SDRMs": BLUE, "NNRTI SDRMs": GREEN}


def set_level_xaxis(ax):
    ax.set_xscale("log")
    ax.set_xticks(LEVEL_TICKS)
    ax.set_xticklabels(LEVEL_TICK_LABELS)
    ax.minorticks_off()


def panel_label(ax, letter):
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")


def main():
    HERE = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooled", default=str(HERE / "pooled_concordance_by_level.csv"))
    ap.add_argument("--sdrm", default=str(HERE / "sdrm_concordance_by_level.csv"))
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--out-prefix", default="resistance_concordance_figure")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled = pd.read_csv(args.pooled)
    sdrm = pd.read_csv(args.sdrm)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    axA, axB, axC = axes[0], axes[1], axes[2]

    # === Panel A: overall agreement vs level ===
    axA.plot(pooled["level_numeric"], pooled["exact_pct"], "o-", color=BLUE, linewidth=2, markersize=7,
              markeredgecolor="black", markeredgewidth=0.5, label="Exact 5-level agreement")
    axA.plot(pooled["level_numeric"], pooled["binary_pct"], "s--", color=ORANGE, linewidth=1.8, markersize=6,
              markeredgecolor="black", markeredgewidth=0.5, label="Binary (susceptible vs. reduced) agreement")
    set_level_xaxis(axA)
    axA.set_ylim(95, 100.5)
    axA.set_xlabel("Subsampling level (% of reads)")
    axA.set_ylabel("Agreement with original (%)")
    axA.set_title(f"Drug-level agreement vs. subsampling level\n(n = {int(pooled['n_drug_calls'].iloc[0])} drug calls per level, 68 samples × 22 drugs)",
                   fontsize=10.5)
    axA.legend(fontsize=8.5, loc="lower right")
    axA.grid(alpha=0.25, linewidth=0.6)
    panel_label(axA, "A")

    # === Panel B: no-call rate vs level ===
    axB.bar([str(l) for l in pooled["level_numeric"]], pooled["no_call_pct"], color=GRAY,
            edgecolor="black", linewidth=0.6, alpha=0.85)
    axB.set_xlabel("Subsampling level (% of reads)")
    axB.set_ylabel("Drug calls with no result (%)")
    axB.set_title("No-call rate vs. level\n(subsampling failed to produce any resistance-level call)", fontsize=10.5)
    axB.grid(axis="y", alpha=0.25, linewidth=0.6)
    panel_label(axB, "B")

    # === Panel C: SDRM recall by class vs level ===
    for cat in ["PI SDRMs", "NRTI SDRMs", "NNRTI SDRMs"]:
        sub = sdrm[sdrm["category"] == cat].sort_values("level_numeric")
        axC.plot(sub["level_numeric"], sub["recall"] * 100, "o-", color=CLASS_COLORS[cat],
                  linewidth=2, markersize=6, markeredgecolor="black", markeredgewidth=0.4, label=cat)
    set_level_xaxis(axC)
    axC.set_ylim(0, 105)
    axC.set_xlabel("Subsampling level (% of reads)")
    axC.set_ylabel("SDRM recall vs. original (%)")
    axC.set_title("Surveillance drug-resistance mutation (SDRM) recall", fontsize=10.5)
    axC.legend(fontsize=8.5, loc="lower right")
    axC.grid(alpha=0.25, linewidth=0.6)
    panel_label(axC, "C")

    fig.suptitle("Drug-resistance-calling concordance: percent-subsampled vs. original full-depth data",
                 fontsize=13.5, fontweight="bold", y=1.04)
    fig.tight_layout()

    png_path = out_dir / f"{args.out_prefix}.png"
    pdf_path = out_dir / f"{args.out_prefix}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Figure saved: {png_path} / {pdf_path}")


if __name__ == "__main__":
    main()
