#!/usr/bin/env python3
"""
compare_resistance_subsampling.py

Compares Stanford HIVdb drug-resistance calls (from fastq2codfreq output,
run per subsampling level) against the original full-depth calls, across
all seven percent-subsampling levels (0.5-40%). Methodology mirrors
compare_canu_sanger.py (the earlier Canu-vs-Sanger concordance analysis in
this project), generalized to loop over multiple comparison levels instead
of a single pair.

Inputs expected in --input-dir (default: current directory):
  ONTreads_resistanceSummaries.csv, ONTreads_sequenceSummaries.csv   (original)
  <level>_resistanceSummaries.csv, <level>_sequenceSummaries.csv      (per level)
  where <level> in 0.5, 1, 2, 5, 10, 20, 40

Only PI/NRTI/NNRTI drugs are compared (INSTI is uniformly NA in this
dataset, since integrase was not sequenced by either the original or
subsampled runs).

Outputs (all in --out-dir):
  drug_concordance_by_level.csv      -- one row per level x drug
  pooled_concordance_by_level.csv    -- one row per level (sens/spec/kappa/etc.)
  mutation_concordance_by_level.csv  -- one row per level x mutation category
  sdrm_concordance_by_level.csv      -- one row per level x SDRM category
  discordant_calls.csv               -- every individual drug call that
                                         disagreed with the original, with the
                                         actual resistance levels on both sides
  subtype_concordance_by_level.csv   -- subtype match rate per level

Usage:
  python3 compare_resistance_subsampling.py --input-dir . --out-dir .
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["0.5", "1", "2", "5", "10", "20", "40"]
LEVEL_NUMERIC = {lvl: float(lvl) for lvl in LEVELS}

DRUG_CLASS = {}
for d in ["ATV/r", "DRV/r", "FPV/r", "IDV/r", "LPV/r", "NFV", "SQV/r", "TPV/r"]:
    DRUG_CLASS[d] = "PI"
for d in ["ABC", "AZT", "D4T", "DDI", "FTC", "ISL", "3TC", "TDF"]:
    DRUG_CLASS[d] = "NRTI"
for d in ["DOR", "DPV", "EFV", "ETR", "NVP", "RPV"]:
    DRUG_CLASS[d] = "NNRTI"
DRUG_COLS = list(DRUG_CLASS.keys())

MUT_COLS = ["PI Major", "PI Accessory", "NRTI", "NNRTI"]
SDRM_COLS = ["PI SDRMs", "NRTI SDRMs", "NNRTI SDRMs"]

RESISTANCE_LEVEL_NAMES = {
    "1": "Susceptible", "2": "Potential low-level", "3": "Low-level",
    "4": "Intermediate", "5": "High-level",
}


def parse_set(s):
    s = (s or "").strip()
    if s in ("None", "NA", ""):
        return set()
    return set(x.strip() for x in s.split(","))


def parse_subtype(s):
    m = re.match(r"([A-Za-z0-9_]+)", str(s))
    return m.group(1) if m else s


def load(input_dir, prefix_res, prefix_seq):
    res = pd.read_csv(input_dir / f"{prefix_res}", dtype=str, keep_default_na=False)
    seq = pd.read_csv(input_dir / f"{prefix_seq}", dtype=str, keep_default_na=False)
    res["bc"] = res["Sequence Name"].str.extract(r"(barcode\d+)")
    seq["bc"] = seq["Sequence Name"].str.extract(r"(barcode\d+)")
    return res, seq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default=".")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_res, orig_seq = load(input_dir, "ONTreads_resistanceSummaries.csv", "ONTreads_sequenceSummaries.csv")
    print(f"Original: {len(orig_res)} samples")

    drug_rows = []
    pooled_rows = []
    mut_rows = []
    sdrm_rows = []
    subtype_rows = []
    discordant_rows = []

    for lvl in LEVELS:
        res, seq = load(input_dir, f"{lvl}_resistanceSummaries.csv", f"{lvl}_sequenceSummaries.csv")
        r = orig_res.merge(res, on="bc", suffixes=("_orig", "_sub"))
        s = orig_seq.merge(seq, on="bc", suffixes=("_orig", "_sub"))
        n = len(r)
        lvl_num = LEVEL_NUMERIC[lvl]

        # ---- drug-level concordance ----
        # Note: the original (full-depth) call is never "NA" for PI/NRTI/NNRTI drugs in
        # this dataset (only INSTI, already excluded) -- but subsampled runs sometimes
        # fail to produce a level at all (insufficient depth for that gene at that level).
        # Those "no-call" cells are tracked separately and excluded from the agreement
        # stats below (which need a real call on both sides to be comparable).
        tp_all = tn_all = fp_all = fn_all = 0
        exact_all = 0
        no_call_all = 0
        for d in DRUG_COLS:
            lvl_o = pd.to_numeric(r[f"{d} Level_orig"], errors="coerce")
            lvl_su = pd.to_numeric(r[f"{d} Level_sub"], errors="coerce")
            no_call_mask = lvl_su.isna() & lvl_o.notna()
            valid_mask = lvl_o.notna() & lvl_su.notna()
            n_no_call = int(no_call_mask.sum())
            no_call_all += n_no_call

            lvl_o_v = lvl_o[valid_mask]
            lvl_su_v = lvl_su[valid_mask]
            exact = (lvl_o_v == lvl_su_v)
            bin_o = lvl_o_v >= 2
            bin_su = lvl_su_v >= 2
            bin_match = (bin_o == bin_su)
            tp = int((bin_o & bin_su).sum())
            tn = int((~bin_o & ~bin_su).sum())
            fp = int((~bin_o & bin_su).sum())
            fn = int((bin_o & ~bin_su).sum())
            tp_all += tp; tn_all += tn; fp_all += fp; fn_all += fn
            exact_all += int(exact.sum())

            drug_rows.append(dict(
                level=lvl, level_numeric=lvl_num, drug=d, drug_class=DRUG_CLASS[d], n=n,
                n_valid=int(valid_mask.sum()), n_no_call=n_no_call, no_call_pct=100 * n_no_call / n,
                exact_agree=int(exact.sum()), exact_pct=100 * exact.mean() if valid_mask.any() else np.nan,
                binary_agree=int(bin_match.sum()), binary_pct=100 * bin_match.mean() if valid_mask.any() else np.nan,
                tp=tp, tn=tn, fp=fp, fn=fn,
                n_resistant_orig=int(bin_o.sum()), n_resistant_sub=int(bin_su.sum()),
            ))

            # log every discordant call individually (both true disagreements and no-calls)
            for _, row in r[valid_mask & ~bin_match].iterrows():
                discordant_rows.append(dict(
                    level=lvl, level_numeric=lvl_num, sample=row["bc"], drug=d, drug_class=DRUG_CLASS[d],
                    orig_score=row[f"{d} Score_orig"], orig_level=row[f"{d} Level_orig"],
                    sub_score=row[f"{d} Score_sub"], sub_level=row[f"{d} Level_sub"],
                    direction="sub_gained_resistance" if float(row[f"{d} Level_sub"]) > float(row[f"{d} Level_orig"]) else "sub_lost_resistance",
                ))
            for _, row in r[no_call_mask].iterrows():
                discordant_rows.append(dict(
                    level=lvl, level_numeric=lvl_num, sample=row["bc"], drug=d, drug_class=DRUG_CLASS[d],
                    orig_score=row[f"{d} Score_orig"], orig_level=row[f"{d} Level_orig"],
                    sub_score=row[f"{d} Score_sub"], sub_level=row[f"{d} Level_sub"],
                    direction="no_call",
                ))

        total_cells = n * len(DRUG_COLS)
        n_valid_cells = total_cells - no_call_all
        po = (tp_all + tn_all) / n_valid_cells if n_valid_cells else np.nan
        pe = (((tp_all + fp_all) * (tp_all + fn_all) + (tn_all + fn_all) * (tn_all + fp_all)) / n_valid_cells ** 2
              ) if n_valid_cells else np.nan
        kappa = (po - pe) / (1 - pe) if (n_valid_cells and pe != 1) else np.nan
        sens = tp_all / (tp_all + fn_all) if (tp_all + fn_all) else np.nan
        spec = tn_all / (tn_all + fp_all) if (tn_all + fp_all) else np.nan
        ppv = tp_all / (tp_all + fp_all) if (tp_all + fp_all) else np.nan
        npv = tn_all / (tn_all + fn_all) if (tn_all + fn_all) else np.nan

        pooled_rows.append(dict(
            level=lvl, level_numeric=lvl_num, n_samples=n, n_drug_calls=total_cells,
            n_valid_calls=n_valid_cells, n_no_call=no_call_all, no_call_pct=100 * no_call_all / total_cells,
            exact_pct=100 * exact_all / n_valid_cells if n_valid_cells else np.nan,
            binary_pct=100 * po if n_valid_cells else np.nan,
            sensitivity=sens, specificity=spec, ppv=ppv, npv=npv, accuracy=po, kappa=kappa,
            tp=tp_all, tn=tn_all, fp=fp_all, fn=fn_all,
        ))

        # ---- mutation-list concordance ----
        for col in MUT_COLS:
            shared = sanger_only = canu_only = 0
            orig_total = sub_total = 0
            for _, row in r.iterrows():
                oset = parse_set(row[f"{col}_orig"])
                sset = parse_set(row[f"{col}_sub"])
                shared += len(oset & sset)
                sanger_only += len(oset - sset)
                canu_only += len(sset - oset)
                orig_total += len(oset)
                sub_total += len(sset)
            mut_rows.append(dict(
                level=lvl, level_numeric=lvl_num, category=col,
                orig_total=orig_total, sub_total=sub_total, shared=shared,
                orig_only=sanger_only, sub_only=canu_only,
                recall=shared / orig_total if orig_total else np.nan,
                precision=shared / sub_total if sub_total else np.nan,
            ))

        # ---- SDRM concordance (from sequenceSummaries) ----
        for col in SDRM_COLS:
            shared = orig_only = sub_only = 0
            orig_total = sub_total = 0
            for _, row in s.iterrows():
                oset = parse_set(row[f"{col}_orig"])
                sset = parse_set(row[f"{col}_sub"])
                shared += len(oset & sset)
                orig_only += len(oset - sset)
                sub_only += len(sset - oset)
                orig_total += len(oset)
                sub_total += len(sset)
            sdrm_rows.append(dict(
                level=lvl, level_numeric=lvl_num, category=col,
                orig_total=orig_total, sub_total=sub_total, shared=shared,
                orig_only=orig_only, sub_only=sub_only,
                recall=shared / orig_total if orig_total else np.nan,
                precision=shared / sub_total if sub_total else np.nan,
            ))

        # ---- subtype concordance ----
        s["subtype_orig"] = s["Subtype (%)_orig"].map(parse_subtype)
        s["subtype_sub"] = s["Subtype (%)_sub"].map(parse_subtype)
        match = (s["subtype_orig"] == s["subtype_sub"])
        subtype_rows.append(dict(level=lvl, level_numeric=lvl_num, n=len(s),
                                  match_pct=100 * match.mean(), n_mismatch=int((~match).sum())))

        print(f"{lvl}%: n={n}  no_call={100*no_call_all/total_cells:.2f}%  "
              f"binary_agree={100*po:.2f}%  exact_agree={100*exact_all/n_valid_cells:.2f}%  "
              f"kappa={kappa:.3f}  subtype_match={100*match.mean():.1f}%")

    drug_df = pd.DataFrame(drug_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    mut_df = pd.DataFrame(mut_rows)
    sdrm_df = pd.DataFrame(sdrm_rows)
    subtype_df = pd.DataFrame(subtype_rows)
    discordant_df = pd.DataFrame(discordant_rows)

    drug_df.to_csv(out_dir / "drug_concordance_by_level.csv", index=False)
    pooled_df.to_csv(out_dir / "pooled_concordance_by_level.csv", index=False)
    mut_df.to_csv(out_dir / "mutation_concordance_by_level.csv", index=False)
    sdrm_df.to_csv(out_dir / "sdrm_concordance_by_level.csv", index=False)
    subtype_df.to_csv(out_dir / "subtype_concordance_by_level.csv", index=False)
    discordant_df.to_csv(out_dir / "discordant_calls.csv", index=False)

    print("\n✓ Saved: drug_concordance_by_level.csv, pooled_concordance_by_level.csv,")
    print("  mutation_concordance_by_level.csv, sdrm_concordance_by_level.csv,")
    print("  subtype_concordance_by_level.csv, discordant_calls.csv")

    print("\nPooled concordance by level:")
    print(pooled_df[["level", "n_drug_calls", "no_call_pct", "exact_pct", "binary_pct",
                      "sensitivity", "specificity", "kappa"]]
          .to_string(index=False))

    print("\nSDRM recall by level (pooled across PI/NRTI/NNRTI):")
    sdrm_pooled = sdrm_df.groupby("level_numeric").agg(
        orig_total=("orig_total", "sum"), shared=("shared", "sum"), sub_total=("sub_total", "sum")
    )
    sdrm_pooled["recall"] = sdrm_pooled["shared"] / sdrm_pooled["orig_total"]
    sdrm_pooled["precision"] = sdrm_pooled["shared"] / sdrm_pooled["sub_total"]
    print(sdrm_pooled.round(3).to_string())

    print(f"\nTotal discordant individual drug calls across all levels: {len(discordant_df)}")
    if len(discordant_df):
        print(discordant_df["direction"].value_counts().to_string())


if __name__ == "__main__":
    main()
