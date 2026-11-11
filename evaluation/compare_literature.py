"""compare_literature.py — Comparison with Makiyah, Mwangi, Lira."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np

# Published reference values
MAKIYAH = {
    "recovery_cycles_proposed": 2.6,
    "recovery_cycles_rule": 4.8,
    "loss_pct_proposed": 3.2,
    "loss_pct_rule": 9.7,
    "loss_pct_no_healing": 18.4,
}
MWANGI = {
    "latency_reduction_pct": 28.3,
    "throughput_recovery_improvement_pct": 23.0,
    "mttr_reduction_pct": 42.0,
    "delivery_ratio_pct": 98.7,
    "cpu_baseline_pct": 55.6,
    "cpu_proposed_pct": 78.7,
}
LIRA = {
    "note": "Lira et al. report GPT-4o as LLM baseline; we compare our LoRA SLM vs GPT-4o on intent dataset."
}

def _pct_improvement(our: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - our) / baseline * 100

def compare(csv_path: str, output_csv: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    rows = []

    # --- Makiyah comparison (T1 tree, B0/B2/FULL) ---
    mak_cfgs = ["B0", "B2", "FULL"]
    for cfg in mak_cfgs:
        sub = df[(df["topology"].str.contains("T1")) & (df["config"] == cfg)]
        if sub.empty:
            continue
        loss = sub["packet_loss_pct"].mean()
        ref_loss = {"B0": MAKIYAH["loss_pct_no_healing"], "B2": MAKIYAH["loss_pct_rule"],
                    "FULL": MAKIYAH["loss_pct_proposed"]}.get(cfg, None)
        rows.append({
            "paper": "Makiyah et al.",
            "metric": "packet_loss_pct",
            "config": cfg,
            "our_value": round(loss, 3),
            "ref_value": ref_loss,
            "delta": round(loss - ref_loss, 3) if ref_loss is not None else None,
            "limitation": "Different topology/implementation; qualitative comparison only",
        })

    # --- Mwangi comparison ---
    b0 = df[df["config"] == "B0"]
    full = df[df["config"] == "FULL"]
    if not b0.empty and not full.empty:
        lat_b0 = b0["latency_p50_ms"].mean()
        lat_full = full["latency_p50_ms"].mean()
        our_lat_pct = _pct_improvement(lat_full, lat_b0)
        rows.append({
            "paper": "Mwangi et al. (OMARF)",
            "metric": "latency_reduction_pct",
            "config": "FULL vs B0",
            "our_value": round(our_lat_pct, 2),
            "ref_value": MWANGI["latency_reduction_pct"],
            "delta": round(our_lat_pct - MWANGI["latency_reduction_pct"], 2),
            "limitation": "OMARF uses OSPF; we use OVS/Ryu; direct comparison approximate",
        })
        thr_b0 = b0["throughput_recovery_s"].mean()
        thr_full = full["throughput_recovery_s"].mean()
        our_thr_pct = _pct_improvement(thr_full, thr_b0)
        rows.append({
            "paper": "Mwangi et al. (OMARF)",
            "metric": "throughput_recovery_improvement_pct",
            "config": "FULL vs B0",
            "our_value": round(our_thr_pct, 2),
            "ref_value": MWANGI["throughput_recovery_improvement_pct"],
            "delta": round(our_thr_pct - MWANGI["throughput_recovery_improvement_pct"], 2),
            "limitation": "Episode length differs (90s vs 5min); normalised per-episode",
        })

    # --- Lira comparison (SLM offline eval) ---
    slm_eval_path = Path("results/tables/slm_offline_eval.json")
    if slm_eval_path.exists():
        slm_eval = json.loads(slm_eval_path.read_text())
        for metric in ["syntax_acc", "format_acc", "goal_acc", "hallucination_rate"]:
            our_v = slm_eval.get("lora", {}).get(metric)
            gpt4o_v = slm_eval.get("gpt4o", {}).get(metric)
            rows.append({
                "paper": "Lira et al.",
                "metric": metric,
                "config": "LoRA vs GPT-4o",
                "our_value": our_v,
                "ref_value": gpt4o_v,
                "delta": round(our_v - gpt4o_v, 4) if our_v and gpt4o_v else None,
                "limitation": LIRA["note"],
            })
    else:
        rows.append({
            "paper": "Lira et al.",
            "metric": "slm_eval",
            "config": "pending",
            "our_value": None, "ref_value": None, "delta": None,
            "limitation": "SLM offline eval not yet run; deferred to ablation phase",
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)

    # LaTeX
    latex = result_df[["paper", "metric", "config", "our_value",
                        "ref_value", "delta"]].to_latex(
        index=False, float_format="%.3f",
        caption="Comparison with published literature (preliminary; N=2 dry run).",
        label="tab:literature")
    (out / "comparison_literature.tex").write_text(latex)
    print(f"Literature comparison written to {output_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/raw/runs.csv")
    ap.add_argument("--output", default="results/comparison_existing.csv")
    ap.add_argument("--output-dir", default="results/tables/")
    args = ap.parse_args()
    compare(args.input, args.output, args.output_dir)

if __name__ == "__main__":
    main()

