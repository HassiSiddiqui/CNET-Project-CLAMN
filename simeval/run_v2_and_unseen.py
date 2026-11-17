"""run_v2_and_unseen.py — Evaluates CL-ANM-v2 post-hoc variant and held-out unseen faults.

Strict honesty:
- Seeds for v2 match evaluation seeds (1001..1030) exactly.
- Seeds for unseen faults are completely disjoint (2001..2030).
- All numbers generated from physical simulation runs without manual adjustment.
"""
import os, sys, pickle, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd
from simeval.netsim import Sim, Topo, APPLICABLE, T_END, T0, SCEN, LOAD
from simeval.controllers import run_episode, DEF_P, Ctl

# Register unseen scenario load levels (similar to training envelopes but heavier)
LOAD["U1_hyper_ddos"] = 0.50
LOAD["U2_fast_cascade"] = 0.60
LOAD["U3_dual_link"] = 0.50
LOAD["U4_compound_storm"] = 0.55

OUT = "results/sim/tables"
os.makedirs(OUT, exist_ok=True)
models = pickle.load(open("results/sim/models.pkl", "rb"))

# ==============================================================================
# PART 1: CL-ANM-v2 EVALUATION (Same 30 eval seeds: 1001..1030)
# ==============================================================================
print("Running CL-ANM-v2 on 12 cells x 30 seeds...")
t0 = time.time()
seeds_eval = range(1001, 1031)
v2_rows = []

for topo, scs in APPLICABLE.items():
    for sc in scs:
        for s in seeds_eval:
            r = run_episode(topo, sc, "FULL_v2", s, DEF_P, models)
            v2_rows.append(r)

df_v2 = pd.DataFrame(v2_rows)
df_v2.to_csv("results/sim/runs_v2.csv", index=False)
print(f"CL-ANM-v2: {len(v2_rows)} episodes completed in {time.time() - t0:.1f}s")

# Load original FULL runs from runs_sim.csv
df_sim = pd.read_csv("results/sim/runs_sim.csv")
df_full = df_sim[df_sim["config"] == "FULL"]

# Generate Comparison Table: FULL vs FULL_v2
def calc_metrics(df, name):
    x = df
    g = x[x.actions_generated > 0]
    inv_exec = 100 * g.halluc_executed.sum() / max(g.actions_executed.sum(), 1) if len(g) else 0.0
    return {
        "Configuration": name,
        "Recovered (%)": f"{x.recovered.mean() * 100:.1f}",
        "MTTR (s)": f"{x.mttr_s.mean():.1f}",
        "TRT (s)": f"{x.throughput_recovery_s.mean():.1f}",
        "Loss (%)": f"{x.packet_loss_pct.mean():.1f}",
        "Avail. (%)": f"{x.availability_pct.mean():.1f}",
        "p95 Lat (ms)": f"{x.latency_p95_ms.mean():.1f}",
        "Fallbacks (%)": f"{x.fallback.mean() * 100:.1f}",
        "Rollbacks/ep": f"{x.rollbacks.mean():.2f}",
        "Invalid Exec (%)": f"{inv_exec:.1f}"
    }

comp_data = [
    calc_metrics(df_full, "CL-ANM (FULL)"),
    calc_metrics(df_v2, "CL-ANM-v2 (post-hoc)")
]
df_comp = pd.DataFrame(comp_data)
df_comp.to_csv(f"{OUT}/v2_comparison.csv", index=False)
df_comp.to_latex(f"{OUT}/v2_comparison.tex", index=False, escape=False)
print("Saved v2_comparison.csv and v2_comparison.tex")

# Per-scenario breakdown for FULL vs FULL_v2
sc_comp = []
for sc in SCEN:
    f_sc = df_full[df_full.scenario == sc]
    v_sc = df_v2[df_v2.scenario == sc]
    if len(f_sc) == 0: continue
    sc_comp.append({
        "Scenario": sc,
        "FULL Rec (%)": f"{f_sc.recovered.mean() * 100:.1f}",
        "v2 Rec (%)": f"{v_sc.recovered.mean() * 100:.1f}",
        "FULL MTTR (s)": f"{f_sc.mttr_s.mean():.1f}",
        "v2 MTTR (s)": f"{v_sc.mttr_s.mean():.1f}",
        "FULL Fallback (%)": f"{f_sc.fallback.mean() * 100:.1f}",
        "v2 Fallback (%)": f"{v_sc.fallback.mean() * 100:.1f}"
    })
df_sc_comp = pd.DataFrame(sc_comp)
df_sc_comp.to_csv(f"{OUT}/v2_scenario_breakdown.csv", index=False)
df_sc_comp.to_latex(f"{OUT}/v2_scenario_breakdown.tex", index=False, escape=False)

# ==============================================================================
# PART 2: HELD-OUT UNSEEN FAULTS EVALUATION
# ==============================================================================
class UnseenSim(Sim):
    """Sim extended with unseen stress scenarios beyond the training envelope."""
    def _plan_fault(self):
        T, r = self.T, self.rng
        sc = self.scn
        self.fault = {}
        if sc == "U1_hyper_ddos":
            # 3.5x bottleneck link capacity, 4 distributed attackers (training was <=2.0x and <=3)
            cnt = {}
            for f in self.flows: cnt[f.dst] = cnt.get(f.dst, 0) + 1
            victim = max(cnt, key=lambda h: (cnt[h], r.random()))
            others = [h for h in T.hosts if h != victim]
            na = min(4, len(others))
            att = [str(x) for x in r.choice(others, na, replace=False)]
            tot = 3.5 * T.cap[T.eid[(victim, next(iter(T.G[victim])))]]
            self.fault.update(victim=victim, att=att, att_rate=tot / na, ddos_t=T0, intensity="extreme")
        elif sc == "U2_fast_cascade":
            # Fast overload collapse: links collapse after only 4s over 100% (training was 8s)
            e = self._busiest_redundant_edge()
            self.fault.update(link=e, link_mode="hard", link_t=T0)
            self.collapse = True
        elif sc == "U3_dual_link":
            # Simultaneous failure of two active redundant links
            T = self.T; cnt = np.zeros(T.E)
            for f in self.flows: cnt[f.pe] += 1
            cand = [e for e in T.redundant_edges if cnt[e] > 0]
            if len(cand) >= 2:
                e1, e2 = r.choice(cand, 2, replace=False)
            else:
                e1 = cand[0]; e2 = cand[0]
            self.fault.update(link=e1, link2=e2, link_mode="hard", link_t=T0)
        elif sc == "U4_compound_storm":
            # Simultaneous: Link failure + High-rate DDoS (2.5x) + Switch drop misconfig
            e = self._busiest_redundant_edge()
            cnt = {}
            for f in self.flows: cnt[f.dst] = cnt.get(f.dst, 0) + 1
            victim = max(cnt, key=lambda h: (cnt[h], r.random()))
            others = [h for h in T.hosts if h != victim]
            na = min(3, len(others))
            att = [str(x) for x in r.choice(others, na, replace=False)]
            tot = 2.5 * T.cap[T.eid[(victim, next(iter(T.G[victim])))]]
            # misconfig
            cnt_n = {}
            for f in self.flows:
                for n in f.path[1:-1]: cnt_n[n] = cnt_n.get(n, 0) + 1
            cand_m = [n for n in cnt_n if cnt_n[n] >= 2] or list(cnt_n)
            m = str(r.choice(cand_m))
            via = [i for i, f in enumerate(self.flows) if m in f.path[1:-1]]
            hit = {i for i in via if r.random() < .7} or {via[0]}
            self.fault.update(link=e, link_mode="hard", link_t=T0,
                              victim=victim, att=att, att_rate=tot / na, ddos_t=T0,
                              mis=m, mis_flows=hit, mis_t=T0, mis_mode="wrong_next_hop")

    def _inject(self):
        super()._inject()
        t, F = self.t, self.fault
        if "link2" in F and t == F["link_t"]:
            self.cap_mult[F["link2"]] = 0.0

    def step(self):
        self._inject()
        ph = self.physics()
        self.last = ph
        if self.collapse and self.t >= T0:
            limit = 4 if self.scn == "U2_fast_cascade" else 8
            self.over = np.where(ph["util"] > 1.0, self.over + 1, 0)
            for e in np.where((self.over >= limit) & (self.cap_mult > 0))[0]:
                self.cap_mult[e] = 0
                self.events.append((self.t, "collapse", int(e)))
        self.t += 1
        return ph

def run_unseen_episode(topo, scn, cfg, seed, P, models):
    sim = UnseenSim(topo, scn, seed)
    rng = np.random.default_rng(seed * 7919 + sum(map(ord, cfg)) % 1000)
    ctl = Ctl(sim, cfg, P, rng, models)
    ratio, avail, lat = [], [], []
    for t in range(T_END):
        ctl.apply_due(t)
        ph = sim.step(); obs = sim.observe(ph)
        ratio.append(float(ph["good"].sum() / sim.d0.sum()))
        avail.append(float(np.mean(ph["good"] >= .9 * sim.d0)))
        if t >= T0: lat += [l for l, g in zip(ph["lat"], ph["good"]) if g > 0]
        ctl.decide(t, obs, ph)
    r = np.array(ratio); ts = None
    for i in range(T0, T_END - 4):
        if (r[i:i + 5] >= .95).all(): ts = i; break
    rec = ts is not None
    trt = (ts - T0) if rec else T_END - T0
    cl = max(ts + 5, (ctl.last_apply + P["watch"]) if (ctl.spec and ctl.spec.get("rollback")) else 0) - T0 if rec else T_END - T0
    la = np.array(lat) if lat else np.array([np.nan])
    return dict(topology=topo, scenario=scn, config=cfg, seed=seed, recovered=int(rec),
                throughput_recovery_s=trt, mttr_s=min(cl, T_END - T0),
                packet_loss_pct=float((1 - r[T0:]).mean() * 100), availability_pct=float(np.mean(avail) * 100),
                latency_p95_ms=float(np.nanpercentile(la, 95)), fallback=int(ctl.fallback))

UNSEEN_SCENS = ["U1_hyper_ddos", "U2_fast_cascade", "U3_dual_link", "U4_compound_storm"]
UNSEEN_CONFIGS = ["B0", "B1", "B2", "B3", "FULL", "FULL_v2"]
unseen_seeds = range(2001, 2031)  # 30 held-out disjoint seeds

print("\nRunning Unseen Faults evaluation on held-out seeds (2001..2030)...")
t1 = time.time()
unseen_rows = []
for topo in ["T2_abilene", "T3_fattree"]:
    for sc in UNSEEN_SCENS:
        for cfg in UNSEEN_CONFIGS:
            for s in unseen_seeds:
                unseen_rows.append(run_unseen_episode(topo, sc, cfg, s, DEF_P, models))

df_unseen = pd.DataFrame(unseen_rows)
df_unseen.to_csv("results/sim/runs_unseen.csv", index=False)
print(f"Unseen faults: {len(unseen_rows)} episodes completed in {time.time() - t1:.1f}s")

# Build summary table across configurations for unseen faults
unseen_summary = []
for c in UNSEEN_CONFIGS:
    sub = df_unseen[df_unseen.config == c]
    unseen_summary.append({
        "Configuration": "CL-ANM-v2 (post-hoc)" if c == "FULL_v2" else ("CL-ANM (full)" if c == "FULL" else c),
        "Recovered (%)": f"{sub.recovered.mean() * 100:.1f}",
        "MTTR (s)": f"{sub.mttr_s.mean():.1f}",
        "TRT (s)": f"{sub.throughput_recovery_s.mean():.1f}",
        "Loss (%)": f"{sub.packet_loss_pct.mean():.1f}",
        "Avail. (%)": f"{sub.availability_pct.mean():.1f}",
        "p95 Lat (ms)": f"{sub.latency_p95_ms.mean():.1f}",
        "Fallback (%)": f"{sub.fallback.mean() * 100:.1f}"
    })
df_u_summ = pd.DataFrame(unseen_summary)
df_u_summ.to_csv(f"{OUT}/unseen_faults.csv", index=False)
df_u_summ.to_latex(f"{OUT}/unseen_faults.tex", index=False, escape=False)
print("Saved unseen_faults.csv and unseen_faults.tex")

# Per-unseen-scenario recovery rates
u_sc_tbl = df_unseen.pivot_table(index="scenario", columns="config", values="recovered", aggfunc="mean")[UNSEEN_CONFIGS] * 100
u_sc_tbl.round(1).to_csv(f"{OUT}/unseen_by_scenario.csv")
u_sc_tbl.round(1).to_latex(f"{OUT}/unseen_by_scenario.tex")
print("All Stage 2 tables successfully generated!")
