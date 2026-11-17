"""analyze.py — statistics, tables (LaTeX) and figures from results/sim/runs_sim.csv. Nothing here is hand-edited."""
import os, json, itertools
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/sim"; TB = f"{OUT}/tables"; FG = f"{OUT}/figures"; os.makedirs(TB, exist_ok=True); os.makedirs(FG, exist_ok=True)
d = pd.read_csv(f"{OUT}/runs_sim.csv"); sw = pd.read_csv(f"{OUT}/runs_sweep.csv")
ORDER = ["B0", "B1", "B2", "B3", "FULL", "A1", "A2", "A3", "A4", "A5", "A6"]
NAME = {"B0": "B0 Unmanaged", "B1": "B1 Zero-shot LLM", "B2": "B2 Static rules", "B3": "B3 Single DQN", "FULL": "CL-ANM (full)",
        "A1": "A1 w/o verification", "A2": "A2 w/o debate", "A3": "A3 w/o RL", "A4": "A4 w/o rollback", "A5": "A5 w/o graph enc.", "A6": "A6 debate always-on"}
COL = dict(zip(ORDER, plt.cm.tab20(np.linspace(0, 1, 11))))
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150})
rng = np.random.default_rng(0)

def ci(x, n=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return x.mean(), x.mean(), x.mean()
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]; return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

def cliff(a, b):
    a, b = np.asarray(a), np.asarray(b); return ((a[:, None] > b[None]).sum() - (a[:, None] < b[None]).sum()) / (len(a) * len(b))

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj

# ---------- main table (pooled over all 12 topology x scenario cells; each cell has 30 seeds)
d["halluc_exec_rate"] = np.where(d.actions_executed > 0, d.halluc_executed / d.actions_executed.clip(lower=1), np.nan) * 100
mets = [("recovered", "Recovered (\\%)", 100), ("mttr_s", "MTTR (s)", 1), ("throughput_recovery_s", "TRT (s)", 1), ("packet_loss_pct", "Loss (\\%)", 1),
        ("availability_pct", "Avail. (\\%)", 1), ("latency_p95_ms", "p95 lat. (ms)", 1)]
rows = []; summ = {}
for c in ORDER:
    x = d[d.config == c]; r = {"Configuration": NAME[c]}
    for k, lab, sc in mets:
        m, lo, hi = ci(x[k] * sc); r[lab] = f"{m:.1f} [{lo:.1f}, {hi:.1f}]"; summ[f"{c}.{k}"] = float(m)
    rows.append(r)
pd.DataFrame(rows).to_latex(f"{TB}/main.tex", index=False, escape=False, column_format="lrrrrrr")
pd.DataFrame(rows).to_csv(f"{TB}/main.csv", index=False)

# ---------- overhead / safety table
ov = []
for c in ORDER:
    x = d[d.config == c]; g = x[x.actions_generated > 0]
    ov.append({"Configuration": NAME[c], "LLM calls": f"{x.llm_calls.mean():.1f}", "Tokens": f"{x.tokens.mean():.0f}",
               "Ctl ms/ep": f"{x.ctl_ms.mean():.0f}", "Invalid gen. (\\%)": f"{100 * g.halluc_generated.sum() / max(g.actions_generated.sum(), 1):.1f}" if len(g) else "--",
               "Invalid exec. (\\%)": f"{100 * g.halluc_executed.sum() / max(g.actions_executed.sum(), 1):.1f}" if len(g) else "--",
               "Rejected": f"{x.plans_rejected.mean():.2f}", "Rollbacks": f"{x.rollbacks.mean():.2f}", "Fallback (\\%)": f"{100 * x.fallback.mean():.0f}"})
pd.DataFrame(ov).to_latex(f"{TB}/overhead.tex", index=False, escape=False, column_format="lrrrrrrrr"); pd.DataFrame(ov).to_csv(f"{TB}/overhead.csv", index=False)

# ---------- per-scenario recovered / MTTR
for k, nm, f in (("recovered", "rec_by_scn", 100), ("mttr_s", "mttr_by_scn", 1)):
    t = d.pivot_table(index="scenario", columns="config", values=k, aggfunc="mean")[ORDER] * f
    t.round(1).to_csv(f"{TB}/{nm}.csv"); t.round(1).rename(columns=lambda c: c).to_latex(f"{TB}/{nm}.tex", column_format="l" + "r" * 11)

# ---------- statistics
st = {}
H, p = stats.kruskal(*[d[d.config == c].mttr_s for c in ORDER]); st["kruskal_mttr_pooled"] = dict(H=float(H), p=float(p))
H, p = stats.kruskal(*[d[d.config == c].throughput_recovery_s for c in ORDER]); st["kruskal_trt_pooled"] = dict(H=float(H), p=float(p))
for scn in sorted(d.scenario.unique()):
    s = d[d.scenario == scn]; H, p = stats.kruskal(*[s[s.config == c].mttr_s for c in ORDER]); st[f"kruskal_mttr_{scn}"] = dict(H=float(H), p=float(p))
pw = []
for metric in ("mttr_s", "packet_loss_pct", "availability_pct"):
    ps, tmp = [], []
    for c in ORDER:
        if c == "FULL": continue
        a, b = d[d.config == "FULL"][metric].values, d[d.config == c][metric].values
        u = stats.mannwhitneyu(a, b, alternative="two-sided"); w = stats.ttest_ind(a, b, equal_var=False)
        ps.append(u.pvalue); tmp.append(dict(metric=metric, vs=c, mean_full=a.mean(), mean_other=b.mean(), welch_p=w.pvalue, mwu_p=u.pvalue, cliffs_delta=cliff(a, b)))
    adj = holm(ps)
    for r, ap in zip(tmp, adj): r["mwu_p_holm"] = ap; pw.append(r)
pw = pd.DataFrame(pw); pw.to_csv(f"{TB}/pairwise_vs_full.csv", index=False)
# per-scenario FULL vs B1/B2/B3 (recovered rate, Fisher exact) 
fs = []
for scn in sorted(d.scenario.unique()):
    for c in ("B1", "B2", "B3"):
        a = d[(d.scenario == scn) & (d.config == "FULL")].recovered; b = d[(d.scenario == scn) & (d.config == c)].recovered
        fs.append(dict(scenario=scn, vs=c, full=a.mean(), other=b.mean(), fisher_p=stats.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1],
                       mttr_full=d[(d.scenario == scn) & (d.config == "FULL")].mttr_s.mean(), mttr_other=d[(d.scenario == scn) & (d.config == c)].mttr_s.mean()))
pd.DataFrame(fs).to_csv(f"{TB}/per_scenario_vs_baselines.csv", index=False)
json.dump(dict(stats=st, summary=summ), open(f"{OUT}/stats.json", "w"), indent=1)

# ---------- Figures
def save(n): plt.tight_layout(); plt.savefig(f"{FG}/{n}.pdf"); plt.savefig(f"{FG}/{n}.png", dpi=200); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
for c in ORDER:
    x = np.sort(d[d.config == c].mttr_s.values); ax[0].step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COL[c], label=c, lw=1.2)
ax[0].set(xlabel="MTTR (s); 70 = not recovered", ylabel="CDF"); ax[0].legend(ncol=2, fontsize=6)
def km(x, ev, T=70):
    t = np.arange(0, T + 1); s = []; S = 1.0
    for u in t:
        n = (x >= u).sum(); e = ((x == u) & (ev == 1)).sum(); S *= 1 - e / n if n else 1; s.append(S)
    return t, np.array(s)
for c in ORDER:
    x = d[d.config == c]; t, s = km(x.throughput_recovery_s.values, x.recovered.values); ax[1].step(t, s, where="post", color=COL[c], lw=1.2, label=c)
ax[1].set(xlabel="Time since fault onset (s)", ylabel="P(not yet recovered)"); save("fig_cdf_km")
fig, ax = plt.subplots(figsize=(3.4, 2.8))
for c in ORDER:
    x = d[d.config == c]; m, lo, hi = ci(x.packet_loss_pct); m2, l2, h2 = ci(x.latency_p95_ms)
    ax.errorbar(m, m2, xerr=[[m - lo], [hi - m]], yerr=[[m2 - l2], [h2 - m2]], fmt="o", color=COL[c], ms=4, lw=.8, label=c)
ax.set(xlabel="Mean packet loss after onset (%)", ylabel="p95 latency (ms)"); ax.legend(fontsize=5, ncol=2); save("fig_lat_loss")
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
for a_, (k, lab) in zip(ax, [("mttr_s", "$\\Delta$ MTTR vs full (s)"), ("packet_loss_pct", "$\\Delta$ loss vs full (pp)"), ("availability_pct", "$\\Delta$ availability vs full (pp)")]):
    base = d[d.config == "FULL"][k].mean(); ys = []
    for c in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        x = d[d.config == c][k].values; f = d[d.config == "FULL"][k].values
        dm = x.mean() - f.mean(); bs = [rng.choice(x, len(x)).mean() - rng.choice(f, len(f)).mean() for _ in range(1000)]
        ys.append((dm, dm - np.percentile(bs, 2.5), np.percentile(bs, 97.5) - dm))
    a_.barh(range(6), [y[0] for y in ys], xerr=[[y[1] for y in ys], [y[2] for y in ys]], color=[COL[c] for c in ["A1", "A2", "A3", "A4", "A5", "A6"]], capsize=2)
    a_.axvline(0, color="k", lw=.6); a_.set_yticks(range(6)); a_.set_yticklabels(["A1", "A2", "A3", "A4", "A5", "A6"] if a_ is ax[0] else []); a_.set_xlabel(lab)
save("fig_ablation")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
t = d.pivot_table(index="scenario", columns="config", values="recovered", aggfunc="mean")[["B0", "B1", "B2", "B3", "FULL"]] * 100
t.index = [i.split("_")[0] for i in t.index]; t.plot.bar(ax=ax[0], width=.8, color=[COL[c] for c in t.columns], legend=False, rot=0); ax[0].set_ylabel("Recovered episodes (%)")
t = d.pivot_table(index="scenario", columns="config", values="mttr_s", aggfunc="mean")[["B0", "B1", "B2", "B3", "FULL"]]; t.index = [i.split("_")[0] for i in t.index]
t.plot.bar(ax=ax[1], width=.8, color=[COL[c] for c in t.columns], rot=0); ax[1].set_ylabel("Mean MTTR (s), 70 = censored"); ax[1].legend(fontsize=6); save("fig_scenarios")
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4)); cf = ["B1", "FULL", "A1", "A4"]
for a_, (k, lab) in zip(ax, [("mttr_s", "MTTR (s)"), ("recovered", "Recovered"), ("halluc_executed", "Executed invalid actions / episode")]):
    for c in cf:
        g = sw[sw.config == c].groupby("p_hal")[k].mean(); a_.plot(g.index, g.values, "o-", color=COL[c], label=NAME[c], ms=3)
    a_.set_xlabel("assumed invalid-action rate $p_{hal}$"); a_.set_ylabel(lab)
ax[0].legend(fontsize=5); save("fig_sensitivity")
fig, ax = plt.subplots(figsize=(3.4, 2.4)); o = pd.read_csv(f"{TB}/overhead.csv")
ax.bar(range(11), [d[d.config == c].llm_calls.mean() for c in ORDER], color=[COL[c] for c in ORDER]); ax.set_xticks(range(11)); ax.set_xticklabels(ORDER, rotation=45); ax.set_ylabel("LLM calls / episode"); save("fig_overhead")
print(json.dumps(st, indent=0)[:900]); print(pw[pw.metric == "mttr_s"].round(4).to_string())
