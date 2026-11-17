"""gen_mn_figures.py — Generates Stage 4 figures: Mininet vs Simulator rankings."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FG = "results/mn/figures"
TB = "results/mn/tables"
os.makedirs(FG, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.grid": True,
                     "grid.alpha": .3, "figure.dpi": 150})

df = pd.read_csv(f"{TB}/mn_vs_sim_ranking.csv")
cfgs = [c.split()[0] for c in df["Configuration"]]
x = np.arange(len(cfgs)); w = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

# MTTR comparison
ax1.bar(x - w/2, df["Sim MTTR (s)"].astype(float), w, label="Simulator", color="#4878cf")
ax1.bar(x + w/2, df["MN MTTR (s)"].astype(float),   w, label="Mininet",   color="#e68000")
ax1.set_xticks(x); ax1.set_xticklabels(cfgs, fontsize=7)
ax1.set_ylabel("MTTR (s)"); ax1.set_title("(a) MTTR: Sim vs Mininet", fontsize=8)
ax1.legend(fontsize=7)

# Recovery Rate comparison
ax2.bar(x - w/2, df["Sim Rec (%)"].astype(float), w, label="Simulator", color="#4878cf")
ax2.bar(x + w/2, df["MN Rec (%)"].astype(float),   w, label="Mininet",   color="#e68000")
ax2.set_xticks(x); ax2.set_xticklabels(cfgs, fontsize=7)
ax2.set_ylabel("Recovered (%)"); ax2.set_title("(b) Recovery Rate: Sim vs Mininet", fontsize=8)
ax2.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f"{FG}/fig_mn_vs_sim.pdf"); plt.savefig(f"{FG}/fig_mn_vs_sim.png", dpi=200)
plt.close()
print("Saved fig_mn_vs_sim")
