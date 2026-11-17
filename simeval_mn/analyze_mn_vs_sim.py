"""analyze_mn_vs_sim.py — Compares Mininet vs Simulator configuration rankings.

Computes Spearman's rho and Kendall's tau rank correlation between
the flow-level simulator and Mininet live OpenFlow execution.
Outputs tables to results/mn/tables/.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats

OUT_DIR = "results/mn/tables"
os.makedirs(OUT_DIR, exist_ok=True)

SIM_RUNS = "results/sim/runs_sim.csv"
MN_RUNS = "results/mn/runs_mn.csv"

CONFIGS = ["B0", "B1", "B2", "B3", "FULL"]
CONFIG_NAMES = {
    "B0": "B0 Unmanaged",
    "B1": "B1 Zero-shot LLM",
    "B2": "B2 Static rules",
    "B3": "B3 Single DQN",
    "FULL": "CL-ANM (full)"
}

def analyze():
    print("Loading simulation results...")
    df_sim = pd.read_csv(SIM_RUNS)
    sub_sim = df_sim[df_sim.config.isin(CONFIGS)]

    # Compute simulator summary per config
    sim_stats = sub_sim.groupby("config").agg({
        "recovered": "mean",
        "mttr_s": "mean",
        "packet_loss_pct": "mean",
        "availability_pct": "mean"
    }).reindex(CONFIGS)
    sim_stats["recovered"] *= 100

    # Load Mininet runs
    has_mn = os.path.exists(MN_RUNS) and os.path.getsize(MN_RUNS) > 100
    if has_mn:
        df_mn = pd.read_csv(MN_RUNS)
        df_mn_clean = df_mn[df_mn.config.isin(CONFIGS)]
    else:
        df_mn_clean = pd.DataFrame()

    # Build comparison rows
    rows = []
    sim_mttr_ranks = sim_stats["mttr_s"].rank().values
    sim_rec_ranks = (-sim_stats["recovered"]).rank().values

    # In Mininet live execution:
    # B3 (DQN local inference) and B2 (fast rule matching) achieve fastest MTTR.
    # FULL exhibits deliberation watch-window latency (verified in live Mininet).
    # B0 has 0% recovery in link down / high flood.
    mn_recovered = []
    mn_mttr = []
    for cfg in CONFIGS:
        sim_rec = sim_stats.loc[cfg, "recovered"]
        sim_m = sim_stats.loc[cfg, "mttr_s"]
        sim_loss = sim_stats.loc[cfg, "packet_loss_pct"]

        # Check if Mininet empirical rows exist for this config
        if len(df_mn_clean) > 0 and cfg in df_mn_clean.config.values:
            cfg_mn = df_mn_clean[df_mn_clean.config == cfg]
            # Use empirical valid rows if wall_s > 0
            valid_cfg = cfg_mn[cfg_mn.wall_s > 0]
            if len(valid_cfg) > 0:
                m_rec = valid_cfg.recovered.mean() * 100
                m_mttr = valid_cfg.mttr_s.mean()
            else:
                raise RuntimeError(f"Config {cfg} has no valid Mininet runs (wall_s > 0). Do not fall back to sim.")
        else:
            raise RuntimeError(f"Config {cfg} missing from Mininet runs. Do not fall back to sim.")

        mn_recovered.append(m_rec)
        mn_mttr.append(m_mttr)

        rows.append({
            "Configuration": CONFIG_NAMES[cfg],
            "Sim Rec (%)": f"{sim_rec:.1f}",
            "MN Rec (%)": f"{m_rec:.1f}",
            "Sim MTTR (s)": f"{sim_m:.1f}",
            "MN MTTR (s)": f"{m_mttr:.1f}",
            "Sim Loss (%)": f"{sim_loss:.1f}",
        })

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(f"{OUT_DIR}/mn_vs_sim_ranking.csv", index=False)
    df_comp.to_latex(f"{OUT_DIR}/mn_vs_sim_ranking.tex", index=False, escape=False)

    # Rank correlations
    mn_mttr_ranks = pd.Series(mn_mttr).rank().values
    mn_rec_ranks = (-pd.Series(mn_recovered)).rank().values

    spearman_mttr, sp_p_mttr = stats.spearmanr(sim_mttr_ranks, mn_mttr_ranks)
    kendall_mttr, kd_p_mttr = stats.kendalltau(sim_mttr_ranks, mn_mttr_ranks)

    spearman_rec, sp_p_rec = stats.spearmanr(sim_rec_ranks, mn_rec_ranks)
    kendall_rec, kd_p_rec = stats.kendalltau(sim_rec_ranks, mn_rec_ranks)

    corr_summary = {
        "mttr_spearman_rho": float(spearman_mttr),
        "mttr_spearman_p": float(sp_p_mttr),
        "mttr_kendall_tau": float(kendall_mttr),
        "mttr_kendall_p": float(kd_p_mttr),
        "recovery_spearman_rho": float(spearman_rec),
        "recovery_spearman_p": float(sp_p_rec),
        "recovery_kendall_tau": float(kendall_rec),
        "recovery_kendall_p": float(kd_p_rec)
    }
    with open(f"{OUT_DIR}/rank_correlations.json", "w") as f:
        json.dump(corr_summary, f, indent=2)

    corr_table = [
        {"Metric": "MTTR Ranking Correlation", "Spearman Rho": f"{spearman_mttr:.3f} (p={sp_p_mttr:.3f})", "Kendall Tau": f"{kendall_mttr:.3f} (p={kd_p_mttr:.3f})"},
        {"Metric": "Recovery Rate Ranking Correlation", "Spearman Rho": f"{spearman_rec:.3f} (p={sp_p_rec:.3f})", "Kendall Tau": f"{kendall_rec:.3f} (p={kd_p_rec:.3f})"}
    ]
    df_corr = pd.DataFrame(corr_table)
    df_corr.to_csv(f"{OUT_DIR}/rank_correlations.csv", index=False)
    df_corr.to_latex(f"{OUT_DIR}/rank_correlations.tex", index=False, escape=False)

    print("\nMininet vs Simulator Ranking Comparison:")
    print(df_comp.to_string(index=False))
    print("\nRank Correlation Statistics:")
    print(df_corr.to_string(index=False))

if __name__ == "__main__":
    analyze()
