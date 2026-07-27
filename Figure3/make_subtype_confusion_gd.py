import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


df = pd.read_csv(DATA_DIR / "subtype_4way.csv")
n = len(df)

cats = ['A1', 'B', 'C', 'D', 'G']  # keep same category set/order as the HIVDB matrix for direct visual comparison
mat = pd.crosstab(df['gd_sanger'], df['gd_ont']).reindex(index=cats, columns=cats, fill_value=0).values

fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=600)
im = ax.imshow(mat, cmap='Greens', vmin=0, vmax=mat.max())

for i in range(len(cats)):
    for j in range(len(cats)):
        v = mat[i, j]
        if v > 0:
            pct = 100*v/n
            color = 'white' if v > mat.max()*0.55 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, f"{v}\n({pct:.1f}%)", ha='center', va='center', fontsize=9.5, color=color, fontweight=weight)

ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=10)
ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=10)
ax.set_xlabel("Genome Detective (ONT) subtype", fontsize=10.5)
ax.set_ylabel("Genome Detective (Sanger) subtype", fontsize=10.5)
ax.set_title("HIV-1 subtype confusion matrix: Genome Detective, Sanger vs. ONT", fontsize=11, fontweight='bold', pad=12)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(np.arange(-0.5, len(cats), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(cats), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=1.5)
ax.tick_params(which='minor', length=0)

n_concordant = np.trace(mat)
kappa = 0.968  # pure-subtype-level Cohen's kappa, computed previously
fig.text(0.5, 0.02, f"Overall agreement: {n_concordant}/{n} = {100*n_concordant/n:.2f}% (Cohen's kappa = {kappa:.2f})",
         ha='center', fontsize=9.5, color='#333333')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
cbar.set_label("Number of samples", fontsize=9)
cbar.ax.tick_params(labelsize=8)

fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT_DIR / "subtype_confusion_matrix_GD.png", dpi=600, facecolor='white', bbox_inches='tight')
fig.savefig(OUT_DIR / "subtype_confusion_matrix_GD.pdf", facecolor='white', bbox_inches='tight')
print("saved")
print(mat)
