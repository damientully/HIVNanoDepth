import pandas as pd, numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

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


W = str(DATA_DIR) + os.sep

# ---- shared data ----
drug_df = pd.read_csv(W+"drug_concordance.csv")
with open(W+"summary_stats.json") as f:
    stats = json.load(f)
with open(W+"ccc_stats.json") as f:
    ccc_stats = json.load(f)
res = pd.read_csv(W+"merged_resistance.csv", dtype=str, keep_default_na=False)
res = res.sort_values('bc').reset_index(drop=True)

drug_cols = ['ATV/r','DRV/r','FPV/r','IDV/r','LPV/r','NFV','SQV/r','TPV/r',
             'ABC','AZT','D4T','DDI','FTC','ISL','3TC','TDF',
             'DOR','DPV','EFV','ETR','NVP','RPV']
drug_class = {}
for d in ['ATV/r','DRV/r','FPV/r','IDV/r','LPV/r','NFV','SQV/r','TPV/r']: drug_class[d]='PI'
for d in ['ABC','AZT','D4T','DDI','FTC','ISL','3TC','TDF']: drug_class[d]='NRTI'
for d in ['DOR','DPV','EFV','ETR','NVP','RPV']: drug_class[d]='NNRTI'

# Okabe-Ito colorblind-safe palette
col_PI, col_NRTI, col_NNRTI = "#0072B2", "#E69F00", "#009E73"
class_colors = {"PI": col_PI, "NRTI": col_NRTI, "NNRTI": col_NNRTI}
col_concordant, col_discordant = "#0072B2", "#D55E00"

def style_ax(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8, length=3)

n = len(res)
level_s = np.zeros((n, len(drug_cols)), dtype=int)
level_c = np.zeros((n, len(drug_cols)), dtype=int)
for j, d in enumerate(drug_cols):
    level_s[:, j] = res[f'{d} Level_sanger'].astype(int).values
    level_c[:, j] = res[f'{d} Level_canu'].astype(int).values
match = (level_s == level_c)

# ---------------------------------------------------------------
fig = plt.figure(figsize=(14.5, 10.5), dpi=600)
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.35, 1.0], height_ratios=[1, 1],
              hspace=0.55, wspace=0.32, left=0.07, right=0.97, top=0.90, bottom=0.07)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[:, 1])

# ================= Panel A: per-drug binary agreement (bars) + exact agreement (markers) =================
order = drug_df.sort_values(['class_', 'binary_pct'], ascending=[True, False])
x = np.arange(len(order))
colors = [class_colors[c] for c in order['class_']]
axA.bar(x, order['binary_pct'], color=colors, width=0.68, edgecolor='white', linewidth=0.5, zorder=2)
axA.set_xticks(x)
axA.set_xticklabels(order['drug'], rotation=45, ha='right', fontsize=7.5)
axA.set_ylabel("Binary agreement with Sanger (%)\n(susceptible vs. reduced susceptibility)", fontsize=8.5)
axA.set_ylim(0, 112)
axA.axhline(stats['total_binary_pct'], color='grey', linestyle='--', linewidth=0.9)
axA.text(len(x) - 0.5, stats['total_binary_pct'] - 6, f"overall binary {stats['total_binary_pct']:.1f}%",
         ha='right', fontsize=7, color='#555555')
axA.set_title("A", fontsize=13, fontweight='bold', loc='left')
style_ax(axA)

handles_cls = [plt.Rectangle((0, 0), 1, 1, color=class_colors[c]) for c in ["PI", "NRTI", "NNRTI"]]
axA.legend(handles_cls, ["PI", "NRTI", "NNRTI"], fontsize=7.5, frameon=False, loc='upper left',
           bbox_to_anchor=(0.0, 1.22), ncol=3, handlelength=1.2, columnspacing=1.0)

# ================= Panel B: resistance-level scatter with Lin's CCC =================
rng = np.random.default_rng(7)
xs = level_s.flatten().astype(float)
ys = level_c.flatten().astype(float)
classes = np.array([drug_class[d] for d in drug_cols])
class_flat = np.repeat(classes[np.newaxis, :], n, axis=0).flatten()
jit = 0.12
for cls in ["PI", "NRTI", "NNRTI"]:
    m = class_flat == cls
    axB.scatter(xs[m] + rng.uniform(-jit, jit, m.sum()),
                ys[m] + rng.uniform(-jit, jit, m.sum()),
                s=18, alpha=0.55, color=class_colors[cls], edgecolors='none', label=cls)
axB.plot([0.5, 5.5], [0.5, 5.5], linestyle='--', color='grey', linewidth=1.2, zorder=1)
axB.set_xlim(0.5, 5.5); axB.set_ylim(0.5, 5.5)
axB.set_xticks([1, 2, 3, 4, 5]); axB.set_yticks([1, 2, 3, 4, 5])
axB.set_xlabel("Sanger resistance level", fontsize=9.5)
axB.set_ylabel("Canu (ONT) resistance level", fontsize=9.5)
axB.set_title("B", fontsize=13, fontweight='bold', loc='left')
style_ax(axB)
# Both the class legend and the CCC/r annotation are placed entirely outside the
# data area (above the axes) since every corner of the 1-5 x 1-5 grid contains
# real data points (including dense clusters at (1,1), (1,5), (5,1) and (5,5)).
txt = (f"Lin's CCC = {ccc_stats['ccc']:.3f} (95% CI {ccc_stats['ci_lo']:.3f}-{ccc_stats['ci_hi']:.3f})\n"
       f"Pearson r = {ccc_stats['pearson']:.3f}   n = {n*len(drug_cols):,} drug-sample pairs")
axB.text(1.0, 1.20, txt, transform=axB.transAxes, ha='right', va='bottom', fontsize=8, color='#333333')
axB.legend(fontsize=7.5, frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.20),
           ncol=3, title="Drug class", title_fontsize=7.5, columnspacing=1.0, handlelength=1.2)

# ================= Panel C: sample x drug discordance heatmap =================
order_idx = np.argsort(res['bc'].astype(int).values)  # barcode ascending
bc_labels = res['bc'].values[order_idx]
mat = match[order_idx, :].astype(int)

cmap = ListedColormap([col_discordant, col_concordant])
Xe = np.arange(len(drug_cols) + 1)
Ye = np.arange(n + 1)
axC.pcolormesh(Xe, Ye, mat, cmap=cmap, vmin=0, vmax=1, edgecolors='white', linewidth=0.35, shading='flat')
axC.invert_yaxis()
axC.set_xlim(0, len(drug_cols)); axC.set_ylim(n, 0)
axC.set_xticks(np.arange(len(drug_cols)) + 0.5)
axC.set_xticklabels(drug_cols, rotation=90, fontsize=6.5)
for tick, d in zip(axC.get_xticklabels(), drug_cols):
    tick.set_color(class_colors[drug_class[d]])
axC.set_yticks(np.arange(n) + 0.5)
axC.set_yticklabels([f"BC{int(b):02d}" for b in bc_labels], fontsize=4.6)
axC.set_ylabel("Sample", fontsize=9.5)
axC.set_xlabel("Drug", fontsize=9.5)
axC.set_title("C", fontsize=13, fontweight='bold', loc='left')
axC.tick_params(length=0)

class_bounds = []
prev = drug_class[drug_cols[0]]
for j, d in enumerate(drug_cols):
    if drug_class[d] != prev:
        class_bounds.append(j)
        prev = drug_class[d]
for b in class_bounds:
    axC.axvline(b, color='white', linewidth=2.0)

handles_bin = [plt.Rectangle((0, 0), 1, 1, color=col_concordant), plt.Rectangle((0, 0), 1, 1, color=col_discordant)]
axC.legend(handles_bin, ["Concordant", "Discordant"], fontsize=8, frameon=False,
           loc='upper center', bbox_to_anchor=(0.5, 1.035), ncol=2)
for spine in axC.spines.values():
    spine.set_visible(False)

out_png = str(OUT_DIR / "concordance_ABC_figure.png")
out_pdf = str(OUT_DIR / "concordance_ABC_figure.pdf")
fig.savefig(out_png, dpi=600, facecolor='white', bbox_inches='tight')
fig.savefig(out_pdf, facecolor='white', bbox_inches='tight')
print("saved")
