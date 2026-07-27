#!/usr/bin/env python3
"""
Compute Lin's concordance correlation coefficient (CCC) between Sanger and ONT
(Canu-assembled) HIV drug-resistance levels, and write ccc_stats.json for
make_combined_ABC.py (Figure 2, Panel B).

Reads merged_resistance.csv (paired Stanford HIVdb resistance levels, one column
per drug per platform: "<drug> Level_sanger" / "<drug> Level_canu") and writes:

    ccc_stats.json  ->  {ccc, ci_lo, ci_hi, pearson}

The 95% CI is estimated by cluster bootstrap, resampling *samples* (rows) with
replacement so that the correlated drug calls within a participant are kept
together, matching the manuscript's approach.

Usage:
    python3 compute_ccc_stats.py                 # uses files in this folder
    python3 compute_ccc_stats.py --reps 3000 --seed 7
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

DRUG_COLS = ['ATV/r', 'DRV/r', 'FPV/r', 'IDV/r', 'LPV/r', 'NFV', 'SQV/r', 'TPV/r',
             'ABC', 'AZT', 'D4T', 'DDI', 'FTC', 'ISL', '3TC', 'TDF',
             'DOR', 'DPV', 'EFV', 'ETR', 'NVP', 'RPV']


def lin_ccc(x, y):
    """Lin's concordance correlation coefficient (population moments)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    vx = x.var()          # population variance (ddof=0)
    vy = y.var()
    cov = ((x - mx) * (y - my)).mean()
    return 2 * cov / (vx + vy + (mx - my) ** 2)


def build_matrix(df):
    """Return per-sample arrays of Sanger and Canu resistance levels (n_samples x n_drugs)."""
    s = np.column_stack([df[f'{d} Level_sanger'].astype(int).values for d in DRUG_COLS])
    c = np.column_stack([df[f'{d} Level_canu'].astype(int).values for d in DRUG_COLS])
    return s, c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default=str(HERE / 'merged_resistance.csv'))
    ap.add_argument('--out', default=str(HERE / 'ccc_stats.json'))
    ap.add_argument('--reps', type=int, default=3000, help='bootstrap replicates')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    s_mat, c_mat = build_matrix(df)
    n_samples = s_mat.shape[0]

    # point estimates on all drug-sample pairs
    xs, ys = s_mat.flatten(), c_mat.flatten()
    ccc = lin_ccc(xs, ys)
    pearson = float(np.corrcoef(xs, ys)[0, 1])

    # cluster bootstrap: resample samples (rows) with replacement
    rng = np.random.default_rng(args.seed)
    boot = np.empty(args.reps)
    idx = np.arange(n_samples)
    for b in range(args.reps):
        pick = rng.choice(idx, size=n_samples, replace=True)
        boot[b] = lin_ccc(s_mat[pick].flatten(), c_mat[pick].flatten())
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    stats = {
        'ccc': round(float(ccc), 3),
        'ci_lo': round(float(ci_lo), 3),
        'ci_hi': round(float(ci_hi), 3),
        'pearson': round(pearson, 3),
        'n_samples': int(n_samples),
        'n_pairs': int(xs.size),
        'bootstrap_reps': int(args.reps),
    }
    Path(args.out).write_text(json.dumps(stats, indent=2) + '\n')
    print(f"Wrote {args.out}")
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
