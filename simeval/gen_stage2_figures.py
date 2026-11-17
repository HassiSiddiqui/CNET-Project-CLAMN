"""gen_stage2_figures.py — Stage 2 figures: v2 comparison and unseen faults. Reads only from CSV files."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FG = "results/sim/figures"
TB = "results/sim/tables"
os.makedirs(FG, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.grid": True,
                     "grid.alpha": .3, "figure.dpi": 150})

# ── Fig: FULL vs CL-ANM-v2 comparison (bar chart, 4 metrics)
df_v2 = pd.read_csv(f"{TB}/v2_comparison.csv")
metrics = ["Recovered (%)", "MTTR (s)", "TRT (s)", "Fallbacks (%)"]
labels  = ["Recovered (%)", "MTTR (s)", "TRT (s)", "Fallback (%)"]
x = np.arange(len(metrics)); w = 0.35
fig, ax = plt.subplots(figsize=(5.5, 2.8))
vals_full = [float(df_v2.loc[0, m]) for m in metrics]
vals_v2   = [float(df_v2.loc[1, m]) for m in metrics]
b1 = ax.bar(x - w/2, vals_full, w, label="CL-ANM (FULL)", color="#4878cf")
b2 = ax.bar(x + w/2, vals_v2,   w, label="CL-ANM-v2 (post-hoc)", color="#6acc65")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.legend(fontsize=7); ax.set_title("Post-hoc CL-ANM-v2 vs FULL (same 30 eval seeds)", fontsize=8)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=6)
plt.tight_layout()
plt.savefig(f"{FG}/fig_v2_comparison.pdf"); plt.savefig(f"{FG}/fig_v2_comparison.png", dpi=200)
plt.close()
print("Saved fig_v2_comparison")

# ── Fig: Unseen faults — Recovered % per configuration (grouped bars by scenario)
df_us_scn = pd.read_csv(f"{TB}/unseen_by_scenario.csv", index_col=0)
configs = ["B0", "B1", "B2", "B3", "FULL", "FULL_v2"]
cfg_labels = {"B0": "B0", "B1": "B1", "B2": "B2", "B3": "B3", "FULL": "FULL", "FULL_v2": "v2"}
colors = {"B0": "#c0c0c0", "B1": "#4878cf", "B2": "#e68000", "B3": "#d43f3a",
          "FULL": "#6acc65", "FULL_v2": "#2ca02c"}
scens = df_us_scn.index.tolist()
n_scn = len(scens); n_cfg = len(configs)
x = np.arange(n_scn); w = 0.13
fig, ax = plt.subplots(figsize=(7, 3))
for i, cfg in enumerate(configs):
    if cfg in df_us_scn.columns:
        vals = df_us_scn[cfg].values
        ax.bar(x + (i - n_cfg/2 + 0.5)*w, vals, w,
               label=cfg_labels[cfg], color=colors[cfg])
ax.set_xticks(x); ax.set_xticklabels([s.replace("U", "U").replace("_", "\n", 1) for s in scens], fontsize=7)
ax.set_ylabel("Recovered (%)"); ax.set_ylim(0, 115)
ax.legend(ncol=6, fontsize=6, loc="upper right")
ax.set_title("Unseen fault generalisation (held-out seeds 2001–2030)", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FG}/fig_unseen_faults.pdf"); plt.savefig(f"{FG}/fig_unseen_faults.png", dpi=200)
plt.close()
print("Saved fig_unseen_faults")

# ── Fig: v2 per-scenario MTTR comparison
df_sc = pd.read_csv(f"{TB}/v2_scenario_breakdown.csv")
sc_names = df_sc["Scenario"].str.replace("_", " ").tolist()
x = np.arange(len(sc_names)); w = 0.35
fig, ax = plt.subplots(figsize=(5.5, 2.8))
ax.bar(x - w/2, df_sc["FULL MTTR (s)"].astype(float), w, label="CL-ANM (FULL)", color="#4878cf")
ax.bar(x + w/2, df_sc["v2 MTTR (s)"].astype(float),   w, label="CL-ANM-v2 (post-hoc)", color="#6acc65")
ax.set_xticks(x); ax.set_xticklabels(sc_names, fontsize=6, rotation=15, ha="right")
ax.set_ylabel("MTTR (s)"); ax.legend(fontsize=7)
ax.set_title("Per-scenario MTTR: FULL vs v2", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FG}/fig_v2_mttr_scenario.pdf"); plt.savefig(f"{FG}/fig_v2_mttr_scenario.png", dpi=200)
plt.close()
print("Saved fig_v2_mttr_scenario")

print("All Stage 2 figures saved.")
