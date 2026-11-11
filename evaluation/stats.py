"""stats.py — Statistical evaluation: Shapiro-Wilk, ANOVA/Kruskal, Welch t, KM."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

def load_runs(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def shapiro_wilk(data: np.ndarray) -> tuple[float, float]:
    from scipy.stats import shapiro
    return shapiro(data)

def levene_test(*groups) -> tuple[float, float]:
    from scipy.stats import levene
    return levene(*groups)

def anova_tukey(groups: dict[str, np.ndarray], metric: str) -> dict:
    from scipy.stats import f_oneway
    vals = list(groups.values())
    F, p = f_oneway(*vals)
    result: dict[str, Any] = {"test": "ANOVA", "F": float(F), "p": float(p)}
    if p < 0.05:
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd  # type: ignore
            import itertools
            all_vals = np.concatenate(vals)
            labels = np.concatenate([[k] * len(v) for k, v in groups.items()])
            tukey = pairwise_tukeyhsd(all_vals, labels)
            result["tukey_summary"] = str(tukey.summary())
        except Exception:
            pass
    return result

def kruskal_dunn(groups: dict[str, np.ndarray]) -> dict:
    from scipy.stats import kruskal
    vals = list(groups.values())
    H, p = kruskal(*vals)
    result: dict[str, Any] = {"test": "Kruskal-Wallis", "H": float(H), "p": float(p)}
    if p < 0.05:
        try:
            import scikit_posthocs as sp  # type: ignore
            df = pd.concat([pd.Series(v, name=k) for k, v in groups.items()], axis=1)
            dunn = sp.posthoc_dunn(df.melt().dropna(), val_col="value",
                                   group_col="variable", p_adjust="holm")
            result["dunn_pvalues"] = dunn.to_dict()
        except Exception:
            pass
    return result

def welch_t(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import ttest_ind
    t, p = ttest_ind(a, b, equal_var=False)
    n = len(a)
    # Cohen's d
    pooled = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
    # 95% CI bootstrap
    diffs = []
    for _ in range(500):
        sa = np.random.choice(a, size=len(a), replace=True)
        sb = np.random.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(sa) - np.mean(sb))
    ci = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
    return {"t": float(t), "p": float(p), "cohens_d": float(d),
            "ci_95": [float(ci[0]), float(ci[1])]}

def kaplan_meier(times: np.ndarray, events: np.ndarray) -> dict:
    """events=1 if recovered (not censored), 0 if censored."""
    try:
        from lifelines import KaplanMeierFitter  # type: ignore
        kmf = KaplanMeierFitter()
        kmf.fit(times, event_observed=events)
        median = float(kmf.median_survival_time_)
        return {"median_recovery_s": median}
    except Exception as exc:
        return {"error": str(exc)}

def run_stats(input_csv: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_runs(input_csv)
    metrics = ["latency_p50_ms", "latency_p95_ms", "throughput_recovery_s",
               "mttr_s", "packet_loss_pct", "availability_pct"]
    configs = df["config"].unique().tolist()
    results: dict[str, Any] = {}

    for metric in metrics:
        col = df[metric].dropna()
        if len(col) < 3:
            continue
        groups = {cfg: df[df["config"] == cfg][metric].dropna().values
                  for cfg in configs}
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if len(groups) < 2:
            continue

        normal = True
        for k, v in groups.items():
            if len(v) >= 3:
                _, p = shapiro_wilk(v)
                if p < 0.05:
                    normal = False

        if normal and len(groups) > 2:
            stat_res = anova_tukey(groups, metric)
        else:
            stat_res = kruskal_dunn(groups)

        # Pairwise FULL vs others
        pairwise: dict[str, Any] = {}
        if "FULL" in groups:
            for cfg, v in groups.items():
                if cfg != "FULL":
                    pairwise[cfg] = welch_t(groups["FULL"], v)

        results[metric] = {"omnibus": stat_res, "pairwise_full": pairwise}

    # Kaplan-Meier on MTTR
    if "mttr_s" in df.columns and "outcome" in df.columns:
        km_groups: dict[str, dict] = {}
        for cfg in configs:
            sub = df[df["config"] == cfg]
            times = sub["mttr_s"].fillna(90).values.astype(float)
            events = (sub["outcome"] == "recovered").astype(int).values
            km_groups[cfg] = kaplan_meier(times, events)
        results["kaplan_meier"] = km_groups

    # Write JSON
    (out / "stats.json").write_text(json.dumps(results, indent=2))

    # LaTeX table stub
    rows = []
    for metric, res in results.items():
        omni = res.get("omnibus", {})
        rows.append({"metric": metric, "test": omni.get("test", ""), 
                     "stat": omni.get("F", omni.get("H", "")),
                     "p": omni.get("p", "")})
    pd.DataFrame(rows).to_csv(out / "stats_summary.csv", index=False)
    print(f"Stats written to {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/raw/runs.csv")
    ap.add_argument("--output", default="results/tables/")
    args = ap.parse_args()
    run_stats(args.input, args.output)

if __name__ == "__main__":
    main()

