"""
make_subsampling_fidelity_figure.py

Publication figure: does percent-based read subsampling preserve the
within-host diversity profile of the original (full-depth) sequencing?

Panel A: genome-wide Shannon entropy across the cohort, at each subsampling
         percentage (median + IQR across 68 samples) -- shows the raw
         diversity metric itself is stable regardless of subsampling level.
Panel B: per-site Shannon entropy profile correlation with the original BAM
         (median + IQR), i.e. does subsampling recover the SAME diversity
         landscape, not just a similar average.
Panel C: recall of the original's minor variants (>=5% minor allele
         frequency) after subsampling (median + IQR) -- the most clinically
         relevant readout, since it asks whether real minority variants are
         still detected.

barcode61 and barcode20 are overlaid individually in B/C since they behave
differently from the rest of the cohort (established via prior analysis):
barcode61 has a persistent recall ceiling (~60%) that subsampling depth
alone doesn't fix; barcode20 converges slowly but does reach high fidelity
by the top subsampling levels.

Panel D: iteration-to-iteration coefficient of variation (CV) of the mean
         Shannon entropy estimate from repeated in-silico rarefaction, vs.
         read count -- how noisy a single subsample of N reads is, purely
         from resampling variance. Well-resolved from 5-1000 reads (dense
         candidate sizes); not shown beyond 1000 reads because per-sample
         read counts diverge above that point (each sample's own max differs),
         so the curve is not comparably resolved across the cohort there.

Inputs (from bam_diversity_metrics.py --skip-depth --rarefaction-source percent):
    external_subsampling_diversity.csv
    vs_original_diversity_comparison.csv
    rarefaction_curve.csv
"""
import csv
from collections import defaultdict
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 0.8
# --- portable paths: resolve I/O relative to this script's own folder ---
# Override with env vars DATA_DIR / OUT_DIR if your data live elsewhere.
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('DATA_DIR', HERE))
OUT_DIR = Path(os.environ.get('OUT_DIR', HERE))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Okabe-Ito colorblind-safe palette (consistent with the rest of the manuscript figures)
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GREY = "#555555"

EXT_CSV = str(DATA_DIR / "external_subsampling_diversity.csv")
VS_CSV = str(DATA_DIR / "vs_original_diversity_comparison.csv")
RAREFACTION_CSV = str(DATA_DIR / "rarefaction_curve.csv")

HIGHLIGHT_SAMPLES = {
    "barcode61": (VERMILLION, "barcode61 (recall ceiling ~60% regardless of depth)"),
    "barcode20": (ORANGE, "barcode20 (slow convergence, high fidelity by 20-40%)"),
}


def load_by_level(path, value_col):
    rows = list(csv.DictReader(open(path)))
    by_level = defaultdict(list)
    for r in rows:
        try:
            val = float(r[value_col])
        except ValueError:
            continue
        by_level[float(r["level_numeric"])].append(val)
    return by_level


def load_sample_series(path, value_col, sample):
    rows = list(csv.DictReader(open(path)))
    pts = []
    for r in rows:
        if r["sample"] != sample:
            continue
        try:
            val = float(r[value_col])
        except ValueError:
            continue
        pts.append((float(r["level_numeric"]), val))
    return sorted(pts)


def median_iqr_series(by_level):
    levels = sorted(by_level.keys())
    med = [st.median(by_level[l]) for l in levels]
    p25 = [sorted(by_level[l])[len(by_level[l]) // 4] for l in levels]
    p75 = [sorted(by_level[l])[3 * len(by_level[l]) // 4] for l in levels]
    return levels, med, p25, p75


def plot_median_iqr(ax, path, value_col, color, ylabel, ylim=None):
    by_level = load_by_level(path, value_col)
    levels, med, p25, p75 = median_iqr_series(by_level)
    ax.fill_between(levels, p25, p75, color=color, alpha=0.18, linewidth=0, label="Cohort IQR (25th-75th pctile)")
    ax.plot(levels, med, color=color, marker="o", markersize=4, linewidth=1.6, label="Cohort median")
    ax.set_xscale("log")
    ax.set_xlabel("Subsampling level (%)")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return levels


def compute_cv_curve(path, max_n=1000, min_samples=5):
    rows = list(csv.DictReader(open(path)))
    by_sample_n = defaultdict(lambda: defaultdict(list))
    for r in rows:
        n = int(r["n_subsampled"])
        if n > max_n:
            continue
        try:
            val = float(r["mean_shannon"])
        except ValueError:
            continue
        by_sample_n[r["sample"]][n].append(val)

    cv_at_n = defaultdict(list)
    for sample, by_n in by_sample_n.items():
        for n, vals in by_n.items():
            if len(vals) > 1 and st.mean(vals) > 0:
                cv_at_n[n].append(st.stdev(vals) / st.mean(vals))

    ns = sorted(n for n, vals in cv_at_n.items() if len(vals) >= min_samples)
    mean_cv = [st.mean(cv_at_n[n]) * 100 for n in ns]
    return ns, mean_cv


fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), dpi=600)
axA, axB, axC, axD = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

# --- Panel A: raw diversity metric stability ---
plot_median_iqr(axA, EXT_CSV, "mean_shannon", BLUE, "Mean Shannon entropy (bits)", ylim=(0, 0.16))
axA.set_title("A. Diversity estimate vs. subsampling level", fontsize=10.5, fontweight="bold", loc="left")

# --- Panel B: entropy-profile correlation with original ---
plot_median_iqr(axB, VS_CSV, "r_shannon_profile", BLUE, "Correlation with original\n(per-site Shannon entropy, r)", ylim=(0.4, 1.02))
for sample, (color, label) in HIGHLIGHT_SAMPLES.items():
    pts = load_sample_series(VS_CSV, "r_shannon_profile", sample)
    if pts:
        xs, ys = zip(*pts)
        axB.plot(xs, ys, color=color, marker="D", markersize=4, linewidth=1.3, linestyle="--", label=sample)
axB.set_title("B. Diversity-profile fidelity vs. original", fontsize=10.5, fontweight="bold", loc="left")

# --- Panel C: minor-variant recall ---
plot_median_iqr(axC, VS_CSV, "variable_site_recall_5pct", BLUE, "Recall of original's minor variants\n(>=5% minor allele frequency)", ylim=(0.2, 1.05))
for sample, (color, label) in HIGHLIGHT_SAMPLES.items():
    pts = load_sample_series(VS_CSV, "variable_site_recall_5pct", sample)
    if pts:
        xs, ys = zip(*pts)
        axC.plot(xs, ys, color=color, marker="D", markersize=4, linewidth=1.3, linestyle="--", label=sample)
axC.set_title("C. Minor-variant recall vs. original", fontsize=10.5, fontweight="bold", loc="left")

# --- Panel D: rarefaction CV vs. read count (well-resolved range only, 5-1000 reads) ---
ns, mean_cv = compute_cv_curve(RAREFACTION_CSV, max_n=1000)
axD.plot(ns, mean_cv, color=BLUE, marker="o", markersize=3.5, linewidth=1.6, label="Cohort mean CV")
for thresh, style_color in [(10, GREEN), (5, ORANGE)]:
    # find first n where mean CV drops to/below threshold
    hit = next((n for n, cv in zip(ns, mean_cv) if cv <= thresh), None)
    if hit:
        axD.axvline(hit, color=style_color, linestyle=":", linewidth=1.1)
        axD.annotate(f"{hit} reads", xy=(hit, thresh), xytext=(hit * 1.15, thresh + 8),
                     fontsize=8, color=style_color)
axD.axhline(10, color=GREEN, linestyle=":", linewidth=1.1, alpha=0.6)
axD.axhline(5, color=ORANGE, linestyle=":", linewidth=1.1, alpha=0.6)
axD.set_xscale("log")
axD.set_xlabel("Reads subsampled (in silico)")
axD.set_ylabel("Iteration-to-iteration CV of\nmean Shannon entropy (%)")
axD.set_title("D. Rarefaction noise vs. read count", fontsize=10.5, fontweight="bold", loc="left")
for spine in ("top", "right"):
    axD.spines[spine].set_visible(False)

# Shared legend below all panels (drawn from panels B/C, which contain the full label set)
handles, labels = axC.get_legend_handles_labels()
seen = set()
final_handles, final_labels = [], []
label_lookup = {"barcode61": HIGHLIGHT_SAMPLES["barcode61"][1], "barcode20": HIGHLIGHT_SAMPLES["barcode20"][1]}
for h, l in zip(handles, labels):
    disp = label_lookup.get(l, l)
    if disp not in seen:
        seen.add(disp)
        final_handles.append(h)
        final_labels.append(disp)

fig.legend(final_handles, final_labels, loc="lower center", ncol=2, frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.04))

fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT_DIR / "subsampling_fidelity_figure.png", dpi=600, facecolor="white", bbox_inches="tight")
fig.savefig(OUT_DIR / "subsampling_fidelity_figure.pdf", facecolor="white", bbox_inches="tight")
print("Saved subsampling_fidelity_figure.png / .pdf")
