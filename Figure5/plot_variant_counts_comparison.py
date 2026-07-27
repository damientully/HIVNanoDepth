#!/usr/bin/env python3
"""
plot_variant_counts_comparison.py

Compares Clair3 variant calls from percent-subsampled BAMs
(clair3_percent_analysis/<sample>/<pct_label>/) against the original,
full-depth Clair3 calls (clair3_original_analysis/<sample>/) to answer two
questions:
  1. Is a given original HIGH-VAF variant (>=15%, Clair3's confident,
     clinically-relevant tier) recovered at all under subsampling?
  2. When recovered, is it called at approximately the same VAF, or does
     the estimated frequency drift as read depth drops?

Scope of this analysis (per user request):
  - Only samples whose full-depth ("original") Clair3 output contains at
    least one HIGH-tier variant are included -- everything else (samples
    with zero, or only MEDIUM/LOW/ARTIFACT-tier, original variants) is
    dropped entirely, not just NaN-filled.
  - By default only VCF records with FILTER == "PASS" are read at all
    (pass --include-non-pass to disable).
  - The subsampled side is searched for a *positional* match to each
    original HIGH-tier variant (chrom:pos:ref:alt) at whatever VAF it
    happens to be called at in the subsampled data -- this is what lets
    us measure VAF drift, not just presence/absence.

Outputs:
  variant_counts_comparison.csv  -- one row per sample x level: counts +
                                     recall/precision of original HIGH variants
  variant_vaf_comparison.csv     -- one row per sample x level x original
                                     HIGH variant: orig_vaf, sub_vaf, delta_vaf
  variant_counts_comparison.png / .pdf  -- 4-panel summary figure

Usage:
  python3 plot_variant_counts_comparison.py \
      --percent-dir clair3_percent_analysis \
      --original-dir clair3_original_analysis \
      --out-dir .
"""

import argparse
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# House style (consistent with the other subsampling-fidelity figures)
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

TIER_RANK = {"ARTIFACT": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


# ---------------------------------------------------------------------------
# VCF parsing
# ---------------------------------------------------------------------------
def _open_vcf(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_vcf_with_vaf(vcf_file, pass_only=True):
    """Parse a Clair3 VCF and extract variants keyed by chrom:pos:ref:alt,
    with VAF computed from the AD (allele depth) FORMAT field
    (VAF = ALT_depth / (REF_depth + ALT_depth) * 100).

    If pass_only is True (default), records whose FILTER column is not
    "PASS" are dropped. Returns (variants_dict, n_dropped_non_pass).
    """
    variants = {}
    n_dropped = 0
    try:
        with _open_vcf(vcf_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 10:
                    continue

                chrom, pos, _id, ref, alt = fields[0], int(fields[1]), fields[2], fields[3], fields[4]
                qual = float(fields[5]) if fields[5] != "." else 0.0
                filter_val = fields[6]

                if pass_only and filter_val != "PASS":
                    n_dropped += 1
                    continue

                format_parts = fields[8].split(":")
                vaf = 0.0
                if "AD" in format_parts:
                    ad_index = format_parts.index("AD")
                    gt_fields = fields[9].split(":")
                    if len(gt_fields) > ad_index:
                        try:
                            depths = gt_fields[ad_index].split(",")
                            if len(depths) >= 2:
                                ref_depth, alt_depth = int(depths[0]), int(depths[1])
                                total = ref_depth + alt_depth
                                if total > 0:
                                    vaf = (alt_depth / total) * 100
                        except (ValueError, IndexError):
                            vaf = 0.0

                key = f"{chrom}:{pos}:{ref}:{alt}"
                variants[key] = {
                    "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
                    "qual": qual, "vaf": vaf, "filter": filter_val,
                }
    except FileNotFoundError:
        print(f"  Warning: VCF not found: {vcf_file}")
    return variants, n_dropped


def categorize_by_vaf(variants):
    categories = {"HIGH": set(), "MEDIUM": set(), "LOW": set(), "ARTIFACT": set()}
    for key, v in variants.items():
        vaf = v["vaf"]
        if vaf >= 15:
            categories["HIGH"].add(key)
        elif vaf >= 5:
            categories["MEDIUM"].add(key)
        elif vaf >= 1:
            categories["LOW"].add(key)
        else:
            categories["ARTIFACT"].add(key)
    return categories


def filter_to_min_tier(variants, min_tier):
    """Keep only variants whose VAF tier is >= min_tier (HIGH is the top)."""
    cats = categorize_by_vaf(variants)
    min_rank = TIER_RANK[min_tier]
    keep = set()
    for tier, rank in TIER_RANK.items():
        if rank >= min_rank:
            keep |= cats[tier]
    return {k: v for k, v in variants.items() if k in keep}


def find_vcf(sample_dir: Path):
    """Prefer merge_output.vcf.gz, fall back to a pre-extracted variants.vcf."""
    vcf_gz = sample_dir / "merge_output.vcf.gz"
    if vcf_gz.exists():
        return vcf_gz
    vcf = sample_dir / "variants.vcf"
    if vcf.exists():
        return vcf
    vcf2 = sample_dir / "merge_output.vcf"
    if vcf2.exists():
        return vcf2
    return None


def level_to_numeric(label):
    """'0p5pct' -> 0.5, '1pct' -> 1.0, '40pct' -> 40.0"""
    m = re.match(r"^(\d+(?:p\d+)?)pct$", label)
    if not m:
        return float("nan")
    return float(m.group(1).replace("p", "."))


# ---------------------------------------------------------------------------
# Discovery + comparison
# ---------------------------------------------------------------------------
def discover_original(original_dir: Path, pass_only=True, min_tier="HIGH"):
    """sample -> {key: variant_dict}, restricted to samples that have at
    least one variant at or above min_tier; everything below is dropped."""
    out = {}
    total_dropped_nonpass = 0
    n_excluded_no_tier = 0
    n_missing_vcf = 0

    if not original_dir.exists():
        print(f"Error: original Clair3 dir not found: {original_dir}")
        return out

    for sample_dir in sorted(original_dir.glob("barcode*")):
        if not sample_dir.is_dir():
            continue
        vcf = find_vcf(sample_dir)
        if vcf is None:
            print(f"  ⊘ {sample_dir.name} (original): no VCF found")
            n_missing_vcf += 1
            continue

        variants, n_dropped = parse_vcf_with_vaf(vcf, pass_only=pass_only)
        total_dropped_nonpass += n_dropped

        variants = filter_to_min_tier(variants, min_tier)
        if not variants:
            n_excluded_no_tier += 1
            continue

        out[sample_dir.name] = variants

    if pass_only:
        print(f"  Dropped {total_dropped_nonpass} non-PASS records from original calls")
    print(f"  Excluded {n_excluded_no_tier} samples with no {min_tier}-tier variant in the original")
    if n_missing_vcf:
        print(f"  Excluded {n_missing_vcf} samples with no original VCF at all")
    print(f"  Retained {len(out)} samples with a qualifying original {min_tier}-tier variant")
    return out


def compare_all(percent_dir: Path, original_variants, pass_only=True):
    """Returns (summary_df, variant_df).
    summary_df: one row per sample x level (counts/recall/precision).
    variant_df: one row per sample x level x original qualifying variant
                (orig_vaf, sub_vaf, delta_vaf) -- the frequency-fidelity table.
    """
    rows = []
    variant_rows = []
    total_dropped = 0

    if not percent_dir.exists():
        print(f"Error: percent Clair3 dir not found: {percent_dir}")
        return pd.DataFrame(rows), pd.DataFrame(variant_rows)

    for sample_dir in sorted(percent_dir.glob("barcode*")):
        if not sample_dir.is_dir():
            continue
        sample = sample_dir.name

        orig_variants = original_variants.get(sample)
        if orig_variants is None:
            # Sample excluded upstream (no qualifying original variant, or no original VCF).
            continue
        orig_keys = set(orig_variants.keys())

        for level_dir in sorted(sample_dir.glob("*")):
            if not level_dir.is_dir():
                continue
            level_label = level_dir.name
            level_numeric = level_to_numeric(level_label)

            vcf = find_vcf(level_dir)
            if vcf is None:
                print(f"  ⊘ {sample}/{level_label}: no VCF found")
                continue

            # Sub side is intentionally NOT tier-restricted: we need to find the
            # original HIGH variant wherever it landed in the subsampled call set
            # (even if its apparent VAF dropped into a lower tier) in order to
            # measure frequency drift, not just presence/absence.
            sub_variants, n_dropped = parse_vcf_with_vaf(vcf, pass_only=pass_only)
            total_dropped += n_dropped
            sub_keys = set(sub_variants.keys())
            sub_high_keys = categorize_by_vaf(sub_variants)["HIGH"]

            recovered_keys = orig_keys & sub_keys

            rows.append({
                "sample": sample,
                "level": level_label,
                "level_numeric": level_numeric,
                "n_variants_orig": len(orig_keys),
                "n_recovered": len(recovered_keys),
                "recall": len(recovered_keys) / len(orig_keys) if orig_keys else np.nan,
                "n_sub_HIGH_calls": len(sub_high_keys),
                "precision_HIGH": (len(orig_keys & sub_high_keys) / len(sub_high_keys)) if sub_high_keys else np.nan,
            })

            for key, ov in orig_variants.items():
                sv = sub_variants.get(key)
                sub_vaf = sv["vaf"] if sv else np.nan
                delta = (sub_vaf - ov["vaf"]) if sv else np.nan
                variant_rows.append({
                    "sample": sample,
                    "level": level_label,
                    "level_numeric": level_numeric,
                    "chrom": ov["chrom"], "pos": ov["pos"], "ref": ov["ref"], "alt": ov["alt"],
                    "orig_vaf": ov["vaf"],
                    "sub_vaf": sub_vaf,
                    "recovered": sv is not None,
                    "delta_vaf": delta,
                    "abs_delta_vaf": abs(delta) if sv else np.nan,
                })

    if pass_only:
        print(f"  Dropped {total_dropped} non-PASS records from subsampled calls")

    return pd.DataFrame(rows), pd.DataFrame(variant_rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def median_iqr_by_level(df, col):
    g = df.groupby("level_numeric")[col].agg(
        median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)
    ).reset_index().sort_values("level_numeric")
    return g


def plot_band(ax, g, color, label):
    ax.plot(g["level_numeric"], g["median"], "o-", color=color, label=label, linewidth=1.6, markersize=4)
    ax.fill_between(g["level_numeric"], g["q25"], g["q75"], color=color, alpha=0.2, linewidth=0)


def make_figure(summary_df, variant_df, original_variants, min_tier, out_prefix):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for ax in (axA, axB, axC):
        ax.set_xscale("log")
        ax.grid(alpha=0.25, linewidth=0.6)

    n_samples = len(original_variants)

    # --- Panel A: recovered variant count vs level, vs original reference ---
    g = median_iqr_by_level(summary_df, "n_recovered")
    plot_band(axA, g, BLUE, "Recovered (median, IQR)")
    orig_counts = [len(v) for v in original_variants.values()]
    orig_median = float(np.median(orig_counts)) if orig_counts else float("nan")
    axA.axhline(orig_median, color=VERMILLION, linestyle="--", linewidth=1.3,
                label=f"Original median ({min_tier}, n={orig_median:.0f})")
    axA.set_xlabel("Subsampling level (% of reads)")
    axA.set_ylabel(f"{min_tier}-tier variants recovered")
    axA.set_title(f"A. Recovered {min_tier}-tier variants vs. level (n={n_samples} samples)")
    axA.legend(fontsize=8, loc="best")

    # --- Panel B: recall / precision vs level ---
    gr = median_iqr_by_level(summary_df, "recall")
    gp = median_iqr_by_level(summary_df, "precision_HIGH")
    plot_band(axB, gr, BLUE, f"Recall of original {min_tier} variants")
    plot_band(axB, gp, ORANGE, f"Precision of subsampled {min_tier} calls")
    axB.set_ylim(0, 1.05)
    axB.set_xlabel("Subsampling level (% of reads)")
    axB.set_ylabel("Fraction")
    axB.set_title("B. Recall & precision vs. original")
    axB.legend(fontsize=8, loc="best")

    # --- Panel C: VAF drift (|delta VAF|) for recovered variants vs level ---
    recovered_df = variant_df[variant_df["recovered"]]
    if not recovered_df.empty:
        gd = median_iqr_by_level(recovered_df, "abs_delta_vaf")
        plot_band(axC, gd, VERMILLION, "|ΔVAF| (median, IQR)")
    axC.set_xlabel("Subsampling level (% of reads)")
    axC.set_ylabel("|Subsampled VAF − Original VAF| (pp)")
    axC.set_title("C. Frequency drift of recovered variants vs. level")
    axC.legend(fontsize=8, loc="best")

    # --- Panel D: subsampled VAF vs original VAF scatter, colored by level ---
    if not recovered_df.empty:
        levels = sorted(recovered_df["level_numeric"].dropna().unique())
        cmap = plt.get_cmap("viridis")
        norm = Normalize(vmin=min(levels), vmax=max(levels))
        for lvl in levels:
            sub = recovered_df[recovered_df["level_numeric"] == lvl]
            axD.scatter(sub["orig_vaf"], sub["sub_vaf"], color=cmap(norm(lvl)),
                        s=22, alpha=0.75, edgecolor="black", linewidth=0.3, label=f"{lvl:g}%")
        lims = [0, 100]
        axD.plot(lims, lims, color="gray", linestyle="--", linewidth=1.0, label="y = x (perfect agreement)")
        axD.set_xlim(lims)
        axD.set_ylim(lims)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axD, fraction=0.046, pad=0.04)
        cbar.set_label("Subsampling level (%)", fontsize=8)
    axD.set_xlabel("Original VAF (%)")
    axD.set_ylabel("Subsampled VAF (%)")
    axD.set_title("D. Subsampled vs. original VAF (recovered variants)")

    fig.suptitle(f"Clair3 {min_tier}-tier variants: subsampled vs. original full-depth calls",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Figure saved: {out_prefix}.png / {out_prefix}.pdf")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--percent-dir", default="clair3_percent_analysis")
    ap.add_argument("--original-dir", default="clair3_original_analysis")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--out-prefix", default="variant_counts_comparison")
    ap.add_argument("--include-non-pass", action="store_true",
                     help="Include non-PASS VCF records (default: PASS-only calls are compared).")
    ap.add_argument("--min-tier", default="HIGH", choices=["HIGH", "MEDIUM", "LOW", "ARTIFACT"],
                     help="Only original samples/variants at or above this VAF tier are kept "
                          "(default: HIGH, i.e. >=15%% VAF; everything below is dropped).")
    args = ap.parse_args()
    pass_only = not args.include_non_pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VARIANT COMPARISON: percent-subsampled vs. original Clair3 calls")
    print(f"Filter mode: {'PASS-only' if pass_only else 'all records (including non-PASS)'}")
    print(f"Minimum original VAF tier: {args.min_tier}")
    print("=" * 80)

    print("\nLoading original (full-depth) Clair3 calls...")
    original_variants = discover_original(Path(args.original_dir), pass_only=pass_only, min_tier=args.min_tier)

    if not original_variants:
        print(f"\nNo samples with a {args.min_tier}-tier original variant found -- nothing to compare.")
        return

    print("\nLoading percent-subsampled Clair3 calls and comparing...")
    summary_df, variant_df = compare_all(Path(args.percent_dir), original_variants, pass_only=pass_only)

    if summary_df.empty:
        print("No data found -- check --percent-dir path, or that sample names match the original set.")
        return

    summary_csv = out_dir / f"{args.out_prefix}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Summary saved: {summary_csv} ({len(summary_df)} rows)")

    variant_csv = out_dir / "variant_vaf_comparison.csv"
    variant_df.to_csv(variant_csv, index=False)
    print(f"✓ Per-variant VAF comparison saved: {variant_csv} ({len(variant_df)} rows)")

    print(f"\nMedian recall / precision / |ΔVAF| by subsampling level ({args.min_tier}-tier only):")
    recovered_df = variant_df[variant_df["recovered"]]
    summary = summary_df.groupby("level_numeric").agg(
        median_recall=("recall", "median"),
        median_precision=("precision_HIGH", "median"),
    ).sort_index()
    if not recovered_df.empty:
        drift = recovered_df.groupby("level_numeric")["abs_delta_vaf"].median()
        summary["median_abs_delta_vaf"] = drift
    print(summary.to_string())

    make_figure(summary_df, variant_df, original_variants, args.min_tier, str(out_dir / args.out_prefix))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
