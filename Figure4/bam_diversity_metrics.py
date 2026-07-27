#!/usr/bin/env python3
"""
bam_diversity_metrics.py

Computes intra-host viral diversity metrics directly from BAM pileups,
independent of the R/longreadvqs workflow -- useful as a cross-check, and
adds a rarefaction analysis (repeated in-silico resampling of reads from the
highest-depth BAM) to test whether diversity estimates have plateaued.

Per reference site, from the column of read bases:
  - Shannon entropy (bits):            H = -sum(p_i * log2(p_i))
  - Normalized Shannon entropy:        H / log2(depth)   (corrects for the
                                        fact that raw H is mechanically
                                        higher with more reads, even with no
                                        real diversity change -- see below)
  - Nucleotide diversity (Nei, 1979):  pi = (n/(n-1)) * (1 - sum(p_i^2))
  - Minor allele frequency:            2nd-largest base frequency at the site

Genome-wide summaries average these across sites with adequate coverage
(--min-site-depth), and count "variable sites" at several minor-allele-
frequency thresholds (1%, 2%, 5%, 20%) as a minority-variant-detection
sensitivity readout.

Two outputs, written to --out-dir:
  1. external_subsampling_diversity.csv
       One row per existing BAM (every depth/percent level, every sample).
  2. rarefaction_curve.csv
       For each sample's highest-depth BAM, repeated in-silico subsampling
       at a range of read counts (multiple bootstrap iterations each),
       recomputing the same summary metrics -- shows the read count at
       which estimates stop changing.

Requires: pysam, numpy
Optional: matplotlib (rarefaction plots; skipped gracefully if unavailable)

Usage:
    python3 bam_diversity_metrics.py
    python3 bam_diversity_metrics.py --min-site-depth 5 --rare-iters 30
"""

import argparse
import csv
import os
import re
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: this script requires numpy.\nInstall with: pip install numpy")

try:
    import pysam
except ImportError:
    sys.exit(
        "ERROR: this script requires pysam.\n"
        "Install it with:  pip install pysam\n"
        "or:                conda install -c bioconda pysam"
    )

BASES = ["A", "C", "G", "T"]
AF_THRESHOLDS = (0.01, 0.02, 0.05, 0.20)


# -----------------------------------------------------------------------------
# BAM / reference I/O  (mirrors bam_to_aligned_fasta.py's alignment logic)
# -----------------------------------------------------------------------------

def read_single_fasta(path):
    seq_chunks = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_chunks:
                    break
                continue
            seq_chunks.append(line)
    return "".join(seq_chunks)


def get_aligned_matrix(bam_path, ref_len, min_mapq=0):
    """Return an (n_reads x ref_len) numpy char array of reference-padded reads."""
    rows = []
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam:
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality is not None and read.mapping_quality < min_mapq:
                continue
            seq = read.query_sequence
            if seq is None:
                continue

            row = ["-"] * ref_len
            for qpos, rpos in read.get_aligned_pairs(matches_only=False):
                if rpos is None or rpos >= ref_len:
                    continue
                row[rpos] = "-" if qpos is None else seq[qpos].upper()
            rows.append(row)

    if not rows:
        return np.empty((0, ref_len), dtype="<U1")
    return np.array(rows, dtype="<U1")


def discover_samples(bam_root):
    if not os.path.isdir(bam_root):
        return []
    return sorted(
        d for d in os.listdir(bam_root)
        if os.path.isdir(os.path.join(bam_root, d)) and d.startswith("barcode")
    )


# -----------------------------------------------------------------------------
# Core diversity math (pure numpy -- no BAM dependency, easy to unit test)
# -----------------------------------------------------------------------------

def compute_site_arrays(matrix):
    """Per-site depth, Shannon entropy, normalized entropy, nucleotide diversity."""
    n_reads, ref_len = matrix.shape
    counts = np.zeros((4, ref_len), dtype=float)
    for i, b in enumerate(BASES):
        counts[i] = (matrix == b).sum(axis=0)
    depth = counts.sum(axis=0)

    safe_depth = np.where(depth == 0, 1, depth)
    p = counts / safe_depth  # shape (4, ref_len)

    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
        shannon = -(p * logp).sum(axis=0)

        # Max achievable entropy given this site's depth: capped by whichever binds first --
        # the sample size (can't observe more than `depth` distinct values) or the 4-letter
        # nucleotide alphabet (can never exceed log2(4)=2 regardless of depth).
        capped_depth = np.where(depth > 1, np.minimum(depth, 4), 2)
        max_h = np.log2(capped_depth)
        norm_shannon = np.where(depth > 1, shannon / max_h, np.nan)

        sum_p2 = (p ** 2).sum(axis=0)
        denom = np.where(depth > 1, depth - 1, 1)
        pi = np.where(depth > 1, (depth / denom) * (1 - sum_p2), np.nan)

    return {"depth": depth, "shannon": shannon, "norm_shannon": norm_shannon, "pi": pi, "p": p}


def summarize(matrix, min_site_depth=3, af_thresholds=AF_THRESHOLDS):
    """Genome-wide summary metrics for one read x reference-position matrix."""
    n_reads, ref_len = matrix.shape
    if n_reads == 0:
        out = {
            "n_reads": 0, "ref_len": ref_len, "n_sites_used": 0,
            "mean_depth": float("nan"), "mean_shannon": float("nan"),
            "mean_norm_shannon": float("nan"), "mean_pi": float("nan"),
        }
        out.update({f"n_variable_sites_{int(t*100)}pct": 0 for t in af_thresholds})
        return out

    stats = compute_site_arrays(matrix)
    depth = stats["depth"]
    mask = depth >= min_site_depth
    n_sites_used = int(mask.sum())

    def safe_mean(arr):
        vals = arr[mask]
        vals = vals[~np.isnan(vals)]
        return float(np.mean(vals)) if len(vals) else float("nan")

    mean_shannon = safe_mean(stats["shannon"])
    mean_norm_shannon = safe_mean(stats["norm_shannon"])
    mean_pi = safe_mean(stats["pi"])
    mean_depth = safe_mean(depth.astype(float))

    p_sorted = -np.sort(-stats["p"], axis=0)  # descending per-site over the 4 bases
    minor_af = p_sorted[1]

    out = {
        "n_reads": n_reads,
        "ref_len": ref_len,
        "n_sites_used": n_sites_used,
        "mean_depth": mean_depth,
        "mean_shannon": mean_shannon,
        "mean_norm_shannon": mean_norm_shannon,
        "mean_pi": mean_pi,
    }
    for thr in af_thresholds:
        out[f"n_variable_sites_{int(thr*100)}pct"] = int(((minor_af >= thr) & mask).sum())
    return out


def get_or_load_original(sample, ref_len, orig_bam_dir, orig_bam_suffix, orig_cache, min_mapq):
    """Load + cache per-site stats for a sample's original (pre-subsampling) BAM."""
    if sample in orig_cache:
        return orig_cache[sample]

    orig_path = os.path.join(orig_bam_dir, f"{sample}{orig_bam_suffix}")
    if not os.path.isfile(orig_path):
        print(f"  No original BAM found for {sample} at {orig_path} -- skipping vs-original comparison")
        orig_cache[sample] = None
        return None

    try:
        matrix = get_aligned_matrix(orig_path, ref_len, min_mapq=min_mapq)
    except Exception as e:
        print(f"  ERROR reading original BAM for {sample}: {e}")
        orig_cache[sample] = None
        return None

    if matrix.shape[0] == 0:
        print(f"  Original BAM for {sample} has 0 usable reads -- skipping")
        orig_cache[sample] = None
        return None

    stats = compute_site_arrays(matrix)
    stats["n_reads"] = matrix.shape[0]
    print(f"  Loaded original BAM for {sample}: n_reads={matrix.shape[0]} ({orig_path})")
    orig_cache[sample] = stats
    return stats


def _safe_corr(a, b, mask):
    a = a[mask]
    b = b[mask]
    valid = ~(np.isnan(a) | np.isnan(b))
    a = a[valid]
    b = b[valid]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare_to_original(sub_matrix, orig_stats, min_site_depth, af_thresholds=AF_THRESHOLDS):
    """Compare one subsampled BAM's per-site diversity profile against the original BAM's."""
    sub_stats = compute_site_arrays(sub_matrix)
    joint_mask = (sub_stats["depth"] >= min_site_depth) & (orig_stats["depth"] >= min_site_depth)
    n_sites_compared = int(joint_mask.sum())

    r_shannon = _safe_corr(sub_stats["shannon"], orig_stats["shannon"], joint_mask)
    r_pi = _safe_corr(sub_stats["pi"], orig_stats["pi"], joint_mask)

    sub_minor_af = (-np.sort(-sub_stats["p"], axis=0))[1]
    orig_minor_af = (-np.sort(-orig_stats["p"], axis=0))[1]
    r_minor_af = _safe_corr(sub_minor_af, orig_minor_af, joint_mask)

    if n_sites_compared > 0:
        mean_shannon_sub = float(np.nanmean(sub_stats["shannon"][joint_mask]))
        mean_shannon_orig = float(np.nanmean(orig_stats["shannon"][joint_mask]))
        mean_pi_sub = float(np.nanmean(sub_stats["pi"][joint_mask]))
        mean_pi_orig = float(np.nanmean(orig_stats["pi"][joint_mask]))
        mean_abs_diff_minor_af = float(np.mean(np.abs(sub_minor_af[joint_mask] - orig_minor_af[joint_mask])))
    else:
        mean_shannon_sub = mean_shannon_orig = float("nan")
        mean_pi_sub = mean_pi_orig = float("nan")
        mean_abs_diff_minor_af = float("nan")

    out = {
        "n_sites_compared": n_sites_compared,
        "r_shannon_profile": r_shannon,
        "r_pi_profile": r_pi,
        "r_minor_af_spectrum": r_minor_af,
        "mean_shannon_sub": mean_shannon_sub,
        "mean_shannon_orig": mean_shannon_orig,
        "delta_mean_shannon": (mean_shannon_sub - mean_shannon_orig) if n_sites_compared > 0 else float("nan"),
        "mean_pi_sub": mean_pi_sub,
        "mean_pi_orig": mean_pi_orig,
        "delta_mean_pi": (mean_pi_sub - mean_pi_orig) if n_sites_compared > 0 else float("nan"),
        "mean_abs_diff_minor_af": mean_abs_diff_minor_af,
    }

    # Minor-variant "recall/precision" relative to the original at each AF threshold:
    # recall = fraction of the original's variable sites still flagged variable after subsampling
    # precision = fraction of the subsample's flagged variable sites that were truly variable in the original
    for thr in af_thresholds:
        ov = (orig_minor_af >= thr) & joint_mask
        sv = (sub_minor_af >= thr) & joint_mask
        n_ov = int(ov.sum())
        n_sv = int(sv.sum())
        n_tp = int((ov & sv).sum())
        thr_pct = int(thr * 100)
        out[f"n_variable_sites_orig_{thr_pct}pct"] = n_ov
        out[f"n_variable_sites_sub_{thr_pct}pct"] = n_sv
        out[f"variable_site_recall_{thr_pct}pct"] = (n_tp / n_ov) if n_ov > 0 else float("nan")
        out[f"variable_site_precision_{thr_pct}pct"] = (n_tp / n_sv) if n_sv > 0 else float("nan")

    return out


def rarefaction_curve(matrix, sizes, n_iter, min_site_depth, af_thresholds, rng):
    n_reads = matrix.shape[0]
    rows = []
    for size in sizes:
        if size > n_reads or size < 1:
            continue
        for it in range(n_iter):
            if size == n_reads:
                sub = matrix
            else:
                idx = rng.choice(n_reads, size=size, replace=False)
                sub = matrix[idx, :]
            s = summarize(sub, min_site_depth=min_site_depth, af_thresholds=af_thresholds)
            s["n_subsampled"] = size
            s["iteration"] = it
            rows.append(s)
    return rows


def default_rarefaction_sizes(n_reads):
    candidates = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    sizes = sorted(set([c for c in candidates if c <= n_reads] + [n_reads]))
    return sizes


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def label_to_numeric(label, mode):
    if mode == "percent":
        lbl = re.sub(r"pct$", "", label).replace("p", ".", 1)
    else:
        lbl = re.sub(r"x$", "", label)
    try:
        return float(lbl)
    except ValueError:
        return float("nan")


def process_tree(bam_root, mode, label_regex, ref_dir, min_mapq, min_site_depth, csv_writer,
                  cache_for_rarefaction=None,
                  orig_bam_dir=None, orig_bam_suffix=None, orig_cache=None, comparison_rows=None):
    samples = discover_samples(bam_root)
    if not samples:
        print(f"No sample folders found under {bam_root}/")
        return

    for sample in samples:
        ref_path = os.path.join(ref_dir, sample, f"{sample}.contigs.fwd.fasta")
        if not os.path.isfile(ref_path):
            print(f"Missing reference for {sample} ({ref_path}) -- skipping")
            continue
        ref_len = len(read_single_fasta(ref_path))
        if ref_len == 0:
            print(f"Empty/unreadable reference for {sample} -- skipping")
            continue

        sample_dir = os.path.join(bam_root, sample)
        bams = sorted(
            f for f in os.listdir(sample_dir)
            if label_regex.search(f) and f.endswith(".bam")
        )
        if not bams:
            print(f"No matching BAMs for {sample} in {sample_dir}/")
            continue

        print(f"[{mode}] {sample}: {len(bams)} BAMs, ref length {ref_len} bp")
        best_level = None
        best_matrix = None

        for bam_fname in bams:
            label = re.sub(r"\.bam$", "", re.sub(rf"^{re.escape(sample)}_", "", bam_fname))
            level_numeric = label_to_numeric(label, mode)
            bam_path = os.path.join(sample_dir, bam_fname)
            try:
                matrix = get_aligned_matrix(bam_path, ref_len, min_mapq=min_mapq)
            except Exception as e:
                print(f"  ERROR reading {bam_fname}: {e}")
                continue

            summary = summarize(matrix, min_site_depth=min_site_depth)
            row = {"mode": mode, "sample": sample, "label": label, "level_numeric": level_numeric}
            row.update(summary)
            csv_writer.writerow(row)
            print(f"  {bam_fname}: n_reads={summary['n_reads']}, mean_shannon={summary['mean_shannon']:.3f}, "
                  f"mean_pi={summary['mean_pi']:.4f}")

            if cache_for_rarefaction is not None:
                if best_level is None or (level_numeric == level_numeric and level_numeric > best_level):
                    best_level = level_numeric
                    best_matrix = matrix

            if orig_bam_dir is not None:
                orig_stats = get_or_load_original(sample, ref_len, orig_bam_dir, orig_bam_suffix,
                                                    orig_cache, min_mapq)
                if orig_stats is not None:
                    comp = compare_to_original(matrix, orig_stats, min_site_depth)
                    comp_row = {
                        "mode": mode, "sample": sample, "label": label, "level_numeric": level_numeric,
                        "n_reads_sub": summary["n_reads"], "n_reads_orig": orig_stats["n_reads"],
                    }
                    comp_row.update(comp)
                    comparison_rows.append(comp_row)

        if cache_for_rarefaction is not None and best_matrix is not None:
            cache_for_rarefaction[sample] = (best_level, best_matrix)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth-bam-dir", default="subsampled_bams")
    p.add_argument("--pct-bam-dir", default="subsampled_bams_by_percent")
    p.add_argument("--ref-dir", default="Canu/best_contig_identity")
    p.add_argument("--out-dir", default="diversity_results")
    p.add_argument("--min-mapq", type=int, default=0)
    p.add_argument("--min-site-depth", type=int, default=3,
                    help="Minimum read depth at a site to include it in genome-wide averages")
    p.add_argument("--skip-depth", action="store_true")
    p.add_argument("--skip-pct", action="store_true")
    p.add_argument("--skip-rarefaction", action="store_true")
    p.add_argument("--rarefaction-source", choices=["depth", "percent"], default="depth",
                    help="Which tree's highest-coverage BAM per sample to use for rarefaction")
    p.add_argument("--rare-iters", type=int, default=20)
    p.add_argument("--rare-sizes", default=None,
                    help="Comma-separated list of read counts to rarefy to (default: auto)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--orig-bam-dir", default="final_bam",
                    help="Directory containing the original (pre-subsampling) BAMs")
    p.add_argument("--orig-bam-suffix", default=".sorted.bam",
                    help="Suffix appended to <sample> to form the original BAM filename, "
                         "e.g. barcode01.sorted.bam")
    p.add_argument("--skip-vs-original", action="store_true",
                    help="Skip comparison of every subsampled BAM against the original BAM")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    depth_label_re = re.compile(r"_\d+x\.bam$")
    pct_label_re = re.compile(r"_[\w]+pct\.bam$")

    summary_fields = ["mode", "sample", "label", "level_numeric", "n_reads", "ref_len",
                       "n_sites_used", "mean_depth", "mean_shannon", "mean_norm_shannon", "mean_pi",
                       "n_variable_sites_1pct", "n_variable_sites_2pct",
                       "n_variable_sites_5pct", "n_variable_sites_20pct"]

    rarefaction_cache = {}
    orig_cache = {} if not args.skip_vs_original else None
    comparison_rows = [] if not args.skip_vs_original else None
    orig_bam_dir = args.orig_bam_dir if not args.skip_vs_original else None

    out_csv_path = os.path.join(args.out_dir, "external_subsampling_diversity.csv")
    with open(out_csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()

        cache_target = rarefaction_cache if not args.skip_rarefaction else None

        if not args.skip_depth:
            print("=== Depth-based BAMs ===")
            process_tree(
                args.depth_bam_dir, "depth", depth_label_re, args.ref_dir,
                args.min_mapq, args.min_site_depth, writer,
                cache_for_rarefaction=cache_target if args.rarefaction_source == "depth" else None,
                orig_bam_dir=orig_bam_dir, orig_bam_suffix=args.orig_bam_suffix,
                orig_cache=orig_cache, comparison_rows=comparison_rows,
            )

        if not args.skip_pct:
            print("=== Percent-based BAMs ===")
            process_tree(
                args.pct_bam_dir, "percent", pct_label_re, args.ref_dir,
                args.min_mapq, args.min_site_depth, writer,
                cache_for_rarefaction=cache_target if args.rarefaction_source == "percent" else None,
                orig_bam_dir=orig_bam_dir, orig_bam_suffix=args.orig_bam_suffix,
                orig_cache=orig_cache, comparison_rows=comparison_rows,
            )

    print(f"\nWrote {out_csv_path}")

    if comparison_rows:
        comparison_fields = [
            "mode", "sample", "label", "level_numeric", "n_reads_sub", "n_reads_orig",
            "n_sites_compared", "r_shannon_profile", "r_pi_profile", "r_minor_af_spectrum",
            "mean_shannon_sub", "mean_shannon_orig", "delta_mean_shannon",
            "mean_pi_sub", "mean_pi_orig", "delta_mean_pi", "mean_abs_diff_minor_af",
        ]
        for thr in AF_THRESHOLDS:
            thr_pct = int(thr * 100)
            comparison_fields += [
                f"n_variable_sites_orig_{thr_pct}pct", f"n_variable_sites_sub_{thr_pct}pct",
                f"variable_site_recall_{thr_pct}pct", f"variable_site_precision_{thr_pct}pct",
            ]
        comp_csv_path = os.path.join(args.out_dir, "vs_original_diversity_comparison.csv")
        with open(comp_csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=comparison_fields)
            writer.writeheader()
            for row in comparison_rows:
                writer.writerow(row)
        print(f"Wrote {comp_csv_path}")
        try:
            plot_vs_original(comp_csv_path, args.out_dir)
        except ImportError:
            print("matplotlib not available -- skipping vs-original plot (CSV is still written).")
        except Exception as e:
            print(f"Could not generate vs-original plot: {e}")
    elif not args.skip_vs_original:
        print("No vs-original comparisons produced (no matching original BAMs found "
              f"in {args.orig_bam_dir}/ with suffix {args.orig_bam_suffix}).")

    if args.skip_rarefaction:
        print("Skipping rarefaction analysis (--skip-rarefaction).")
        return

    if not rarefaction_cache:
        print("No BAMs available for rarefaction (check --rarefaction-source matches a processed tree).")
        return

    print("\n=== Rarefaction analysis ===")
    rng = np.random.default_rng(args.seed)
    rare_fields = ["sample", "source_level", "n_subsampled", "iteration", "n_reads", "ref_len",
                   "n_sites_used", "mean_depth", "mean_shannon", "mean_norm_shannon", "mean_pi",
                   "n_variable_sites_1pct", "n_variable_sites_2pct",
                   "n_variable_sites_5pct", "n_variable_sites_20pct"]
    rare_csv_path = os.path.join(args.out_dir, "rarefaction_curve.csv")

    custom_sizes = None
    if args.rare_sizes:
        custom_sizes = [int(x) for x in args.rare_sizes.split(",")]

    with open(rare_csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rare_fields)
        writer.writeheader()
        for sample, (level, matrix) in rarefaction_cache.items():
            n_reads = matrix.shape[0]
            sizes = custom_sizes if custom_sizes else default_rarefaction_sizes(n_reads)
            print(f"{sample}: rarefying from n_reads={n_reads} (source level {level}) "
                  f"at sizes {sizes}, {args.rare_iters} iterations each")
            rows = rarefaction_curve(matrix, sizes, args.rare_iters, args.min_site_depth, AF_THRESHOLDS, rng)
            for row in rows:
                row["sample"] = sample
                row["source_level"] = level
                writer.writerow(row)

    print(f"\nWrote {rare_csv_path}")

    try:
        plot_rarefaction(rare_csv_path, args.out_dir)
    except ImportError:
        print("matplotlib not available -- skipping rarefaction plot (CSV is still written).")
    except Exception as e:
        print(f"Could not generate rarefaction plot: {e}")


def plot_rarefaction(csv_path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    data = defaultdict(lambda: defaultdict(list))  # sample -> n_subsampled -> [mean_shannon,...]
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sample = row["sample"]
            n = int(row["n_subsampled"])
            try:
                val = float(row["mean_shannon"])
            except ValueError:
                continue
            data[sample][n].append(val)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sample, by_n in sorted(data.items()):
        ns = sorted(by_n.keys())
        means = [np.mean(by_n[n]) for n in ns]
        stds = [np.std(by_n[n]) for n in ns]
        ax.errorbar(ns, means, yerr=stds, marker="o", markersize=3, linewidth=1, alpha=0.7, label=sample)

    ax.set_xscale("log")
    ax.set_xlabel("Reads subsampled (in silico)")
    ax.set_ylabel("Mean Shannon entropy (bits)")
    ax.set_title("Rarefaction: diversity estimate vs. read count")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "rarefaction_shannon.png")
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")


def plot_vs_original(csv_path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # rows[mode][sample] = list of (level_numeric, r_shannon_profile, variable_site_recall_5pct)
    rows = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                level = float(row["level_numeric"])
                r_shannon = float(row["r_shannon_profile"])
            except ValueError:
                continue
            try:
                recall = float(row["variable_site_recall_5pct"])
            except ValueError:
                recall = float("nan")
            rows[row["mode"]][row["sample"]].append((level, r_shannon, recall))

    modes = sorted(rows.keys())
    if not modes:
        print("No data to plot for vs-original comparison.")
        return

    fig, axes = plt.subplots(len(modes), 2, figsize=(10, 4 * len(modes)), squeeze=False)
    for i, mode in enumerate(modes):
        ax_r, ax_recall = axes[i][0], axes[i][1]
        for sample, vals in sorted(rows[mode].items()):
            vals = sorted(vals, key=lambda t: t[0])
            levels = [v[0] for v in vals]
            r_shannon = [v[1] for v in vals]
            recall = [v[2] for v in vals]
            ax_r.plot(levels, r_shannon, marker="o", markersize=3, alpha=0.7, label=sample)
            ax_recall.plot(levels, recall, marker="o", markersize=3, alpha=0.7, label=sample)

        ax_r.set_xscale("log")
        ax_r.set_ylim(-1.05, 1.05)
        ax_r.set_xlabel(f"{mode} level")
        ax_r.set_ylabel("Correlation with original\n(per-site Shannon entropy, r)")
        ax_r.set_title(f"{mode}: entropy profile vs. original")

        ax_recall.set_xscale("log")
        ax_recall.set_ylim(-0.05, 1.05)
        ax_recall.set_xlabel(f"{mode} level")
        ax_recall.set_ylabel("Recall of original's variable\nsites (5% minor-AF threshold)")
        ax_recall.set_title(f"{mode}: minor-variant recall vs. original")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "vs_original_comparison.png")
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
