"""make_plots.py — All publication figures from runs.csv."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS_ORDER = ["B0", "B1", "B2", "B3", "FULL", "A1", "A2", "A3", "A4", "A5", "A6"]
COLORS = plt.cm.tab10.colors

def _save(fig, path: Path, name: str):
    p = path / name
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")

def plot_mttr_cdf(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, cfg in enumerate([c for c in CONFIGS_ORDER if c in df["config"].unique()]):
        vals = df[df["config"] == cfg]["mttr_s"].dropna().sort_values()
        if vals.empty:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=cfg, color=COLORS[i % 10])
    ax.set_xlabel("MTTR (s)")
    ax.set_ylabel("CDF")
    ax.set_title("MTTR CDF by Configuration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out, "mttr_cdf.pdf")

def plot_ablation_bars(df: pd.DataFrame, out: Path):
    metrics = ["mttr_s", "latency_p95_ms", "packet_loss_pct"]
    cfgs = [c for c in CONFIGS_ORDER if c in df["config"].unique()]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, metric in zip(axes, metrics):
        means = [df[df["config"] == c][metric].mean() for c in cfgs]
        ax.bar(range(len(cfgs)), means, color=COLORS[:len(cfgs)])
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric.replace("_", " "))
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out, "ablation_bars.pdf")

def plot_latency_vs_throughput(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    cfgs = [c for c in CONFIGS_ORDER if c in df["config"].unique()]
    for i, cfg in enumerate(cfgs):
        sub = df[df["config"] == cfg]
        ax.scatter(sub["throughput_recovery_s"].mean(),
                   sub["latency_p50_ms"].mean(),
                   label=cfg, s=80, color=COLORS[i % 10])
    ax.set_xlabel("Throughput Recovery Time (s)")
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_title("Latency vs Throughput Recovery")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out, "latency_vs_throughput.pdf")

def plot_km_curves(df: pd.DataFrame, out: Path):
    try:
        from lifelines import KaplanMeierFitter  # type: ignore
    except ImportError:
        print("  [skip] lifelines not installed")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    cfgs = [c for c in CONFIGS_ORDER if c in df["config"].unique()]
    for i, cfg in enumerate(cfgs):
        sub = df[df["config"] == cfg]
        times = sub["mttr_s"].fillna(90).values.astype(float)
        events = (sub["outcome"] == "recovered").astype(int).values
        kmf = KaplanMeierFitter()
        kmf.fit(times, event_observed=events, label=cfg)
        kmf.plot_survival_function(ax=ax, color=COLORS[i % 10])
    ax.set_title("Kaplan-Meier Recovery Curves")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P(not yet recovered)")
    ax.grid(True, alpha=0.3)
    _save(fig, out, "km_curves.pdf")

def plot_overhead(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    cfgs = [c for c in CONFIGS_ORDER if c in df["config"].unique()]
    cpu = [df[df["config"] == c]["cpu_mean"].mean() for c in cfgs]
    tokens = [df[df["config"] == c]["tokens"].mean() for c in cfgs]
    x = np.arange(len(cfgs))
    ax.bar(x - 0.2, cpu, 0.4, label="CPU mean %", color="steelblue")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, tokens, 0.4, label="Tokens/decision", color="darkorange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("CPU mean %")
    ax2.set_ylabel("Tokens/decision")
    ax.set_title("System Overhead by Configuration")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    _save(fig, out, "overhead.pdf")

def plot_hallucination_rate(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5, 3))
    cfgs = [c for c in CONFIGS_ORDER if c in df["config"].unique()]
    hr = [df[df["config"] == c]["hallucination_rate"].mean() for c in cfgs]
    ax.bar(range(len(cfgs)), hr, color=COLORS[:len(cfgs)])
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate by Configuration")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out, "hallucination_rate.pdf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/raw/runs.csv")
    ap.add_argument("--output", default="results/figures/")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    plot_mttr_cdf(df, out)
    plot_ablation_bars(df, out)
    plot_latency_vs_throughput(df, out)
    plot_km_curves(df, out)
    plot_overhead(df, out)
    plot_hallucination_rate(df, out)
    print("All plots done.")

if __name__ == "__main__":
    main()

