# HIVNanoDepth

Analysis code and figure-generation scripts for:

> **Optimising Nanopore Sequencing for Reliable HIV-1 Drug Resistance Profiling**
> Daniel Bugembe Lule, Deogratius Ssemwanga, Nicholas Bbosa, Pontiano Kaleebu, Damien C. Tully.

This repository reproduces the main and supplementary figures of the manuscript. It defines
application-specific Oxford Nanopore (ONT) sequencing-depth requirements for HIV-1 drug
resistance genotyping, comparing matched Sanger and ONT sequencing of the protease and
reverse transcriptase regions in 68 Ugandan individuals receiving antiretroviral therapy.

---

## Repository layout

Each figure lives in its own folder alongside the input data it needs.

| Folder | Script | Produces | Input data (shipped) |
|--------|--------|----------|----------------------|
| `Figure1/` | `sequencing_depth_viral_load_error_figure.R` | Depth / viral load / consensus error rate (Fig 1) | `Viral_load_68.xlsx` |
| `Figure2/` | `make_combined_ABC.py` | Sanger–ONT resistance concordance (Fig 2) | `drug_concordance.csv`, `merged_resistance.csv`, `summary_stats.json`, `ccc_stats.json` |
| `Figure3/` | `make_subtype_confusion_gd.py` | HIV-1 subtype confusion matrix (Fig 3) | `subtype_4way.csv` |
| `Figure4/` | `make_subsampling_fidelity_figure.py` | Within-host diversity vs depth (Fig 4) | `external_subsampling_diversity.csv`, `vs_original_diversity_comparison.csv`, `rarefaction_curve.csv` |
| `Figure4/` | `bam_diversity_metrics.py` | *Upstream:* computes the three CSVs above from BAMs | (requires aligned BAMs, not shipped) |
| `Figure5/` | `plot_vaf_agreement.py` | Variant allele-frequency agreement vs depth (Fig 5) | *requires `variant_vaf_comparison.csv` — see note below* |
| `Figure5/` | `plot_variant_counts_comparison.py` | *Upstream:* builds `variant_vaf_comparison.csv` from VCFs | (requires per-sample VCFs, not shipped) |
| `Figure6/` | `make_resistance_concordance_figure.py` | Resistance concordance vs depth (Fig 6) | `pooled_concordance_by_level.csv`, `sdrm_concordance_by_level.csv` |
| `Figure6/` | `compare_resistance_subsampling.py` | *Upstream:* builds the by-level CSVs from Stanford HIVdb summaries | (requires Stanford summary CSVs, not shipped) |
| `SFigure2/` | `make_discordance_predictors_fig.py` | Predictors of discordance (Supp. Fig 2) | `discordance_predictors.csv` |

`docs/` contains a small landing page for GitHub Pages.

---

## Requirements

**Python** (Figures 2–6, SFigure 2): Python ≥ 3.9 with the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
```

**R** (Figure 1): R ≥ 4.1 with:

```r
install.packages(c("readxl", "tidyverse", "ggpubr", "patchwork", "ggcorrplot", "scales"))
```

---

## Reproducing the figures

Every script resolves its input and output paths **relative to its own location**, so you can
run it from anywhere without editing any paths:

```bash
# Python figures
python3 Figure2/make_combined_ABC.py
python3 Figure3/make_subtype_confusion_gd.py
python3 Figure4/make_subsampling_fidelity_figure.py
python3 Figure6/make_resistance_concordance_figure.py
python3 SFigure2/make_discordance_predictors_fig.py

# R figure — run from inside the folder
cd Figure1 && Rscript sequencing_depth_viral_load_error_figure.R
```

Each script writes a `.png` (600 dpi preview) and a vector `.pdf` (submission quality) into its
own folder. To send outputs elsewhere, set the `OUT_DIR` environment variable (Python scripts),
or use the `--out-dir` flag on the scripts that expose one:

```bash
OUT_DIR=/path/to/figures python3 Figure2/make_combined_ABC.py
python3 Figure6/make_resistance_concordance_figure.py --out-dir /path/to/figures
```

`DATA_DIR` can likewise be set if you keep the input files somewhere other than the script folder.

---

## Reproducibility notes

- **Upstream vs. plotting steps.** For Figures 4, 5 and 6 the repository ships the *intermediate*
  tables that the plotting scripts consume, so the published figures can be regenerated directly.
  The upstream scripts (`bam_diversity_metrics.py`, `plot_variant_counts_comparison.py`,
  `compare_resistance_subsampling.py`) rebuild those tables from the raw BAM/VCF/Stanford outputs
  and are provided for full transparency; they require the primary sequencing data (see Data
  availability) and expose their input directories as command-line arguments.
- **`Figure5/variant_vaf_comparison.csv`** is an upstream product of
  `plot_variant_counts_comparison.py` and is not committed (it derives from per-sample VCFs).
  Generate it first, or point `--input` at your own copy, before running `plot_vaf_agreement.py`.
- **`Figure2/ccc_stats.json`** was reconstructed from the published Lin's concordance correlation
  coefficient (0.839; 95% CI 0.733–0.928) and Pearson r (0.839). Regenerate from the raw
  resistance-level matrix if exact bootstrap confidence intervals are required.
- **Fonts.** The Python scripts request *Liberation Sans* and the R script *Arial/sans*; if these
  are unavailable, Matplotlib/ggplot fall back to the default sans-serif font without affecting
  the data.

---

## Data availability

Raw nanopore sequencing data, consensus sequences and associated metadata have been deposited in
[ENA/SRA accession XXXXX]. Please replace this accession before publication.

## Citation

If you use this code, please cite the manuscript above. A `CITATION.cff` can be added once the DOI
is issued.

## License

Released under the MIT License — see [LICENSE](LICENSE).
