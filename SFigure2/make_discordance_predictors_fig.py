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


W = str(DATA_DIR) + os.sep
d = pd.read_csv(W+"discordance_predictors.csv")
top5 = [73,75,76,78,81]
d['highlight'] = d['bc'].isin(top5)

col_main, col_hi = "#0072B2", "#D55E00"

def style_ax(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8, length=3)

def ols_line(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    b1 = np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)
    b0 = y.mean() - b1*x.mean()
    xg = np.linspace(x.min(), x.max(), 100)
    return xg, b0+b1*xg

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=600)

panels = [
    ("error_rate", "ONT consensus error rate (%)", axes[0], True, lambda v: v*100),
    ("depth", "ONT mean sequencing depth (x)", axes[1], False, lambda v: v),
    ("viral_load", "log10 plasma viral load (copies/mL)", axes[2], False, lambda v: np.log10(v)),
]

for col, xlabel, ax, fit_line, transform in panels:
    xv = transform(d[col].values)
    ax.scatter(xv[~d['highlight']], d['n_discord'][~d['highlight']], s=26, color=col_main,
               alpha=0.65, edgecolors='white', linewidths=0.3, label='Other samples')
    ax.scatter(xv[d['highlight']], d['n_discord'][d['highlight']], s=42, color=col_hi,
               edgecolors='black', linewidths=0.6, zorder=3, label='5 most-discordant samples')
    for i in np.where(d['highlight'])[0]:
        ax.annotate(f"BC{d['bc'][i]:02d}", (xv[i], d['n_discord'][i]), fontsize=7,
                     xytext=(4,3), textcoords='offset points')
    if fit_line:
        xg, yg = ols_line(xv, d['n_discord'].values)
        ax.plot(xg, yg, color='grey', linestyle='--', linewidth=1.2, zorder=1)
    r = np.corrcoef(xv, d['n_discord'])[0,1]
    ax.text(0.03, 0.97, f"r = {r:.2f}", transform=ax.transAxes, ha='left', va='top', fontsize=9, color='#333333')
    ax.set_xlabel(xlabel, fontsize=9)
    style_ax(ax)

axes[0].set_ylabel("Discordant drug-level calls per sample", fontsize=9.5)
axes[0].set_title("A", fontsize=12, fontweight='bold', loc='left')
axes[1].set_title("B", fontsize=12, fontweight='bold', loc='left')
axes[2].set_title("C", fontsize=12, fontweight='bold', loc='left')
axes[0].legend(fontsize=7.5, frameon=False, loc='upper right')

fig.tight_layout()
fig.savefig(OUT_DIR / "discordance_predictors_figure.png", dpi=600, facecolor='white', bbox_inches='tight')
fig.savefig(OUT_DIR / "discordance_predictors_figure.pdf", facecolor='white', bbox_inches='tight')
print("saved")
