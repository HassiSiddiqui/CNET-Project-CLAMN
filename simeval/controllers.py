"""controllers.py — all 11 configurations (B0-B3, FULL, A1-A6) on the shared simulator.

Everything that differs between configurations is *code* (what the controller does), never a metric multiplier.
ASSUMED (not measured) quantities are the SLM error model (p_diag, p_hal, confidence calibration), LLM call latency
and the twin-noise; they are explicit parameters in P and swept in the sensitivity study.
"""
from __future__ import annotations
import math, time
import numpy as np
from .netsim import Sim, T0, T_END, Topo
from .nn import NF, PH, GL

CFGS = {  # llm, verify, debate(None|'gated'|'always'), rl(None|'graph'|'flat'), rollback
    "B0": None, "B2": "rules", "B3": "dqn",
    "B1": dict(verify=0, debate=None, rl=None, rollback=0),
    "FULL": dict(verify=1, debate="gated", rl="graph", rollback=1),
    "FULL_v2": dict(verify=1, debate="gated", rl="graph", rollback=1, v2=True),
    "A1": dict(verify=0, debate="gated", rl="graph", rollback=1),
    "A2": dict(verify=1, debate=None, rl="graph", rollback=1),
    "A3": dict(verify=1, debate="gated", rl=None, rollback=1),
    "A4": dict(verify=1, debate="gated", rl="graph", rollback=0),
    "A5": dict(verify=1, debate="gated", rl="flat", rollback=1),
    "A6": dict(verify=1, debate="always", rl="graph", rollback=1),
}
K = 4
CONF = {"link": "misconfig", "ddos": "congestion", "misconfig": "link", "congestion": "ddos"}
DEF_P = dict(p_diag=.85, p_hal=.10, llm_mean=1.5, twin_sigma=.03, l3_s=1.0, tau_c=.75, tau_r=.30, watch=10, tokens=300)


def step_reward(sim, ph, ratio):
    T = sim.T; u = ph["util"][np.isfinite(ph["util"])]
    return 1.0 * ratio - .3 * min(np.mean(ph["lat"]) / 200, 1) - .5 * (1 - ratio) - .2 * min(u.max() if len(u) else 0, 1.5)


class Ctl:
    def __init__(s, sim: Sim, cfg: str, P: dict, rng, models: dict, train=False, oracle=False):
        s.sim, s.cfg, s.P, s.rng, s.m, s.train, s.oracle = sim, cfg, P, rng, models, train, oracle
        sp = CFGS.get(cfg); s.spec = sp if isinstance(sp, dict) else None
        s.pending = []; s.hist_down = {}; s.trig = 0; s.cool = 0; s.watch = None; s.fallback = False
        s.last_act = -99; s.last_apply = -1; s.cd_flow = {}; s.dec_buf = []; s.rewards = []
        s.st = dict(llm_calls=0, tokens=0, rollbacks=0, rejected=0, halluc_gen=0, halluc_exec=0, actions_gen=0, actions_exec=0,
                    debate=0, diag_ok=None, plans=0, wall=0.0, reroutes=0)
        s.dqn_prev = None

    # ---------------- helpers
    def believed_down(s, t, obs):
        for e in np.where(obs["down"])[0]: s.hist_down[int(e)] = t
        return {e for e, tt in s.hist_down.items() if t - tt <= 4}

    def evidence(s, t, obs):
        T, sim = s.sim.T, s.sim
        down = s.believed_down(t, obs)
        att = [h for h in T.hosts if obs["tx"][h] > 2 * sim.base_tx[h] + .05 * T.cap[T.eid[(h, next(iter(T.G[h])))]]]
        mis = [n for n, v in obs["node_drop"].items() if v > 0]
        cong = [int(e) for e in np.where((obs["util"] > .95) & (~obs["down"]))[0]]
        cl = []
        if down: cl.append("link")
        if att: cl.append("ddos")
        if mis: cl.append("misconfig")
        if cong and not att and not down: cl.append("congestion")
        return dict(cls=cl, down=down, att=att, mis=mis, cong=cong)

    def feats(s, obs, i, cands, est_load, down):
        T, sim = s.sim.T, s.sim; f = sim.flows[i]
        cap = T.cap
        X = np.zeros((T.N, NF))
        for n in T.nodes:
            j = T.nid[n]; es = [T.eid[(n, m)] for m in T.G[n]]
            X[j] = [n in T.switches, len(es) / 6, min(obs["util"][es].max(), 3) / 3, float(any(e in down for e in es))]
        P = np.zeros((K, T.N)); Phi = np.zeros((K, PH)); mask = np.zeros(K, bool)
        for k, p in enumerate(cands[:K]):
            pe = T.path_edges(p); mask[k] = True
            for n in p: P[k, T.nid[n]] = 1
            P[k] /= P[k].sum()
            own = np.zeros(T.E)
            if k == 0 and cands[0] == f.path: own[pe] = f.d
            ld = est_load - own[:T.E] if False else est_load.copy()
            ld_path = ld[pe] + (0 if (cands[0] == f.path and k == 0) else f.d)
            u = ld_path / cap[pe]
            Phi[k] = [min(u.max(), 3) / 3, T.delay[pe].sum() / 50, len(pe) / 10,
                      min(np.clip((cap[pe] - ld[pe]).min() / max(f.d, 1e-6), 0, 5) / 5, 1), float(p == f.path)]
        finite = obs["util"][obs["util"] < 5]
        glob = np.array([min(finite.mean(), 2) / 2, min(obs["util"].max(), 3) / 3, len(down) / 5,
                         float(np.mean(obs["good"] < .9 * sim.d0))])
        return X, T.A_mean, P, Phi, glob, mask

    def cand_paths(s, i, down, avoid=()):
        T, f = s.sim.T, s.sim.flows[i]
        dn = set(down) | set(avoid)
        al = T.alive_paths(f.src, f.dst, dn, K)
        if f.path in al: al.remove(f.path)
        cur = [f.path] if not any(e in dn for e in T.path_edges(f.path)) else []
        return (cur + al)[:K]

    def concretize(s, flows, obs, down, avoid, use_rl, greedy=True, record=True):
        """choose a path per flow -> list of (flow, path). RL = PPO policy; else static shortest alive path."""
        T, sim = s.sim.T, s.sim; out = []
        est = np.where(obs["util"] < 5, obs["util"], 1.0) * T.cap
        for i in sorted(flows, key=lambda i: -sim.flows[i].d):
            c = s.cand_paths(i, down, avoid if not use_rl else avoid)
            if not c: continue
            if use_rl:
                pol = s.m["ppo_graph"] if use_rl == "graph" else s.m["ppo_flat"]
                fe = s.feats(obs, i, c, est, down)
                a, lp, v, pi = pol.act(fe, greedy=greedy and not s.train, rng=s.rng)
                if s.train and record: s.dec_buf.append(dict(feats=fe, a=a, logp=lp, v=v, t=obs["t"]))
                a = min(a, len(c) - 1)
            else:
                al = T.alive_paths(sim.flows[i].src, sim.flows[i].dst, set(down) | set(avoid), 1)
                c = al or c; a = 0
            p = c[a]
            f = sim.flows[i]; est[T.path_edges(f.path)] -= f.d; est[T.path_edges(p)] += f.d
            out.append((i, p))
        return out

    # ---------------- plan generation (SLM error model) and verification
    def diagnose(s, truth, rng):
        P = s.P; dc, conf = [], []
        for c in truth:
            ok = rng.random() < P["p_diag"]
            dc.append(c if ok else CONF[c]); conf.append(rng.beta(8, 2) if ok else rng.beta(4, 4))
        return dc, float(np.mean(conf)) if conf else 1.0

    def llm_call(s, rng):
        s.st["llm_calls"] += 1; s.st["tokens"] += int(rng.normal(s.P["tokens"], 60))
        return float(rng.lognormal(math.log(s.P["llm_mean"]) - .06, .35))

    def build_actions(s, dc, ev, obs, use_rl, rng):
        """clean actions for diagnosed classes; targets come from telemetry evidence (fallback guesses if absent)."""
        T, sim = s.sim.T, s.sim; acts = []; down = ev["down"]
        if "link" in dc or "congestion" in dc:
            avoid = set(down) if "link" in dc else set()
            tgt_e = set(down)
            if "congestion" in dc or not tgt_e:
                tgt_e |= set(ev["cong"]) or {int(np.argmax(np.where(obs["util"] < 5, obs["util"], 0)))}
            flows = [i for i, f in enumerate(sim.flows) if any(e in tgt_e for e in f.pe)]
            avoid = avoid | (set(ev["cong"]) if "congestion" in dc and not use_rl else set())
            for i, p in s.concretize(flows, obs, down, avoid, use_rl):
                acts.append(dict(type="reroute", target=p[1], params=dict(flow=i, path=p)))
        if "ddos" in dc:
            tg = ev["att"] or [max(T.hosts, key=lambda h: obs["tx"][h])]
            acts += [dict(type="acl_add", target=h, params={}) for h in tg]
        if "misconfig" in dc:
            n = ev["mis"][0] if ev["mis"] else str(rng.choice(T.switches))
            acts.append(dict(type="config_revert", target=n, params={}))
        return acts

    def hallucinate(s, acts, ev, rng):
        T = s.sim.T
        for a in acts:
            s.st["actions_gen"] += 1
            if rng.random() >= s.P["p_hal"]: continue
            a["halluc"] = True; s.st["halluc_gen"] += 1
            if rng.random() < .5: a["target"] = "s99"; a["params"] = dict(a["params"], bad=True)          # nonexistent device
            elif a["type"] == "reroute":
                dn = list(ev["down"]); f = s.sim.flows[a["params"]["flow"]]
                bad = [p for p in T.alive_paths(f.src, f.dst, set(), 4) if any(e in dn for e in T.path_edges(p))] if dn else []
                a["params"] = dict(a["params"], path=bad[0] if bad else a["params"]["path"][:])
            elif a["type"] == "acl_add":
                a["target"] = str(rng.choice([h for h in T.hosts if h not in ev["att"]]))
            else: a["target"] = str(rng.choice([n for n in T.switches if n != a["target"]]))
        return acts

    def verify(s, acts, ev, obs, down, rng):
        T, sim = s.sim.T, s.sim
        for a in acts:  # L1: schema + whitelist + inventory
            if a["type"] not in {"reroute", "acl_add", "config_revert"} or a["target"] not in T.G: return 1
        for a in acts:  # L2: invariants consistent with telemetry
            if a["type"] == "reroute":
                p = a["params"]["path"]; f = sim.flows[a["params"]["flow"]]
                if len(set(p)) != len(p) or p[0] != f.src or p[-1] != f.dst: return 2
                if any((p[j], p[j + 1]) not in T.eid for j in range(len(p) - 1)): return 2
                if any(e in down for e in T.path_edges(p)): return 2
            elif a["type"] == "acl_add" and a["target"] not in ev["att"]: return 2
            elif a["type"] == "config_revert" and a["target"] not in ev["mis"]: return 2
        c = sim.clone(); s.apply(c, acts, count=False)
        ph = c.physics(); pred = ph["good"].sum() / sim.d0.sum() + rng.normal(0, s.P["twin_sigma"])
        l3_thresh = 0.0 if (s.spec and s.spec.get("v2")) else .01
        if pred < obs["ratio"] + l3_thresh: return 3                # L3: dry-run must predict improvement (>=0 in v2)
        return 0

    def apply(s, sim, acts, count=True):
        for a in acts:
            if a["type"] == "reroute":
                ok = sim.set_path(a["params"]["flow"], a["params"]["path"]) if a["target"] in sim.T.G else False
                if count and ok: s.st["reroutes"] += 1
            elif a["type"] == "acl_add" and a["target"] in sim.T.G: sim.blocked.add(a["target"])
            elif a["type"] == "config_revert" and a["target"] == sim.mis_node: sim.reverted = True
            if count:
                s.st["actions_exec"] += 1
                if a.get("halluc"): s.st["halluc_exec"] += 1

    def risk(s, acts):
        n = sum(a["type"] == "reroute" for a in acts)
        return min(1.0, .1 * len(acts) + .5 * n / max(len(s.sim.flows), 1))

    # ---------------- LLM pipeline
    def plan(s, t, obs, ev):
        rng, P, spec = s.rng, s.P, s.spec; lat = 0.0; s.st["plans"] += 1
        use_rl = spec["rl"] if not s.oracle else "graph"
        tries = 0
        while tries < 3:
            tries += 1
            if s.oracle:
                dc, conf = list(ev["cls"]), 1.0
            else:
                lat += s.llm_call(rng); dc, conf = s.diagnose(ev["cls"], rng)
                acts = None
                conflict = len(ev["cls"]) >= 2
                risk = s.risk(s.build_actions(dc, ev, obs, use_rl, rng)) if spec["debate"] else 0
                if spec["debate"] and (spec["debate"] == "always" or conf < P["tau_c"] or risk > P["tau_r"] or conflict):
                    s.st["debate"] += 1
                    votes = {c: [] for c in ev["cls"]}
                    for rnd in range(2):
                        calls = [s.llm_call(rng) for _ in range(4)]; lat += max(calls)
                        for c in ev["cls"]:
                            for _ in range(4):
                                ok = rng.random() < P["p_diag"]; votes[c].append((c if ok else CONF[c], rng.beta(8, 2) if ok else rng.beta(4, 4)))
                        # first-attempt diagnosis also votes
                        agree = all(max(set(v for v, _ in votes[c]), key=lambda x: sum(w for y, w in votes[c] if y == x)) == c for c in ev["cls"])
                        if agree: break
                    dc = []
                    for c in ev["cls"]:
                        tot = {}
                        for y, w in votes[c]: tot[y] = tot.get(y, 0) + w
                        dc.append(max(tot, key=tot.get))
                    lat += (0.5 if (spec and spec.get("v2")) else 1.0) * s.llm_call(rng)  # proposer writes final plan
            acts = s.build_actions(dc, ev, obs, use_rl, rng)
            if s.st["diag_ok"] is None: s.st["diag_ok"] = int(sorted(dc) == sorted(ev["cls"]))
            if not s.oracle: acts = s.hallucinate(acts, ev, rng)
            if not spec["verify"] or s.oracle: return acts, lat
            if not (spec and spec.get("v2")):
                lat += P["l3_s"]                                       # in v2, L3 dry-run simulation overlaps with generation
            if s.verify(acts, ev, obs, ev["down"], rng) == 0: return acts, lat
            s.st["rejected"] += 1
        s.fallback = True
        return [], lat

    # ---------------- baselines
    def rules(s, t, obs):
        T, sim = s.sim.T, s.sim; acts = []; down = s.believed_down(t, obs)
        ev = s.evidence(t, obs)
        for h in ev["att"]: acts.append(dict(type="acl_add", target=h, params={}))
        est = np.where(obs["util"] < 5, obs["util"], 1.0)
        for i, f in enumerate(sim.flows):
            if t - s.cd_flow.get(i, -99) < 5: continue
            cross_down = any(e in down for e in f.pe)
            hot = [e for e in f.pe if est[e] > .8]
            dead = obs["good"][i] < .1 * sim.d0[i] and not cross_down
            if cross_down:
                al = T.alive_paths(f.src, f.dst, down, 1)
                if al: acts.append(dict(type="reroute", target=al[0][1], params=dict(flow=i, path=al[0]))); s.cd_flow[i] = t
            elif hot and sim.d0[i] == max(sim.d0[j] for j, g in enumerate(sim.flows) if set(g.pe) & set(hot)):
                c = [p for p in T.alive_paths(f.src, f.dst, down, 3) if p != f.path]
                if c:
                    p = min(c, key=lambda p: est[T.path_edges(p)].max()); acts.append(dict(type="reroute", target=p[1], params=dict(flow=i, path=p))); s.cd_flow[i] = t
            elif dead:
                c = [p for p in T.alive_paths(f.src, f.dst, down, 3) if p != f.path]
                if c: acts.append(dict(type="reroute", target=c[0][1], params=dict(flow=i, path=c[0]))); s.cd_flow[i] = t
        if acts: s.pending.append((t + 1, acts, None))

    def dqn_state(s, t, obs):
        T, sim = s.sim.T, s.sim; ev = s.evidence(t, obs); fin = obs["util"][obs["util"] < 5]
        txr = max(obs["tx"][h] / max(sim.base_tx[h], 1e-6) for h in T.hosts if sim.base_tx[h] > 0)
        return np.array([min(fin.mean(), 2) / 2, min(obs["util"].max(), 3) / 3, len(ev["down"]) / 3,
                         float(np.mean(obs["good"] < .9 * sim.d0)), min(txr, 10) / 10, float(bool(ev["mis"])),
                         obs["ratio"], min(t - s.last_act, 10) / 10]), ev

    def dqn_act(s, t, obs, ph):
        sim, T = s.sim, s.sim.T; x, ev = s.dqn_state(t, obs); d = s.m["dqn"]
        r = step_reward(sim, ph, obs["ratio"])
        if s.dqn_prev is not None and s.train:
            d.remember((s.dqn_prev[0], s.dqn_prev[1], r - (.05 if s.dqn_prev[1] else 0), x, False)); d.train_step()
        a = d.act(x, greedy=not s.train); s.dqn_prev = (x, a)
        down = ev["down"]; acts = []
        aff = [i for i in range(len(sim.flows)) if obs["good"][i] < .9 * sim.d0[i]]
        if a in (1, 2, 5):
            fl = aff if a != 5 else [i for i, f in enumerate(sim.flows) if any(obs["util"][e] > .95 for e in f.pe)]
            est = np.where(obs["util"] < 5, obs["util"], 1.0)
            for i in fl:
                f = sim.flows[i]; c = T.alive_paths(f.src, f.dst, down, 3)
                if c:
                    p = c[0] if a == 1 else min(c, key=lambda p: est[T.path_edges(p)].max())
                    if p != f.path: acts.append(dict(type="reroute", target=p[1], params=dict(flow=i, path=p)))
        elif a == 3: acts.append(dict(type="acl_add", target=max(T.hosts, key=lambda h: obs["tx"][h]), params={}))
        elif a == 4 and ev["mis"]: acts.append(dict(type="config_revert", target=ev["mis"][0], params={}))
        if acts: s.pending.append((t + 1, acts, None)); s.last_act = t

    # ---------------- main per-second hooks
    def apply_due(s, t):
        due = [p for p in s.pending if p[0] <= t]; s.pending = [p for p in s.pending if p[0] > t]
        for _, acts, snap in due:
            if acts == "ROLLBACK": s.sim.restore(snap); continue
            if acts == "PARTIAL_ROLLBACK":
                failed_acts, prev_snap = snap
                for a in failed_acts:
                    if a["type"] == "reroute":
                        fl_idx = a["params"]["flow"]
                        s.sim.set_path(fl_idx, prev_snap["paths"][fl_idx])
                    elif a["type"] == "acl_add":
                        if a["target"] not in s.sim.attackers:
                            s.sim.blocked.discard(a["target"])
                    elif a["type"] == "config_revert":
                        if a["target"] != s.sim.mis_node:
                            s.sim.reverted = prev_snap["reverted"]
                continue
            if snap == "TAKE": snap = s.sim.snapshot()
            s.apply(s.sim, acts); s.last_apply = t
            if s.spec and s.spec.get("rollback") and not s.oracle and acts:
                s.watch = dict(end=t + s.P["watch"], snap=snap or s.sim.snapshot(), r0=s.ratio_now, acts=acts)

    def decide(s, t, obs, ph):
        t0 = time.perf_counter(); s.ratio_now = obs["ratio"]
        if s.cfg == "B0": pass
        elif s.cfg == "B2": s.rules(t, obs)
        elif s.cfg == "B3": s.dqn_act(t, obs, ph)
        elif s.cfg == "ppo_train": s.ppo_train_tick(t, obs, ph)
        else: s.llm_tick(t, obs)
        s.st["wall"] += time.perf_counter() - t0

    def llm_tick(s, t, obs):
        if s.fallback: return s.rules(t, obs)
        ev = s.evidence(t, obs)
        trig = bool(ev["cls"]) or obs["ratio"] < .9
        s.trig = s.trig + 1 if trig else 0
        if s.watch and t >= s.watch["end"]:
            w = s.watch; s.watch = None
            if obs["ratio"] < w["r0"] + .02 and obs["ratio"] < .95:
                if s.spec and s.spec.get("v2"):
                    s.pending.append((t + 1, "PARTIAL_ROLLBACK", (w.get("acts", []), w["snap"])))
                else:
                    s.pending.append((t + 1, "ROLLBACK", w["snap"]))
                s.st["rollbacks"] += 1; s.fallback = True
            s.cool = t + 1
            return
        if s.watch or any(p[2] == "TAKE" for p in s.pending) or t < s.cool: return
        if s.trig >= 2 and ev["cls"]:
            acts, lat = s.plan(t, obs, ev)
            if acts:
                s.pending.append((t + max(1, math.ceil(lat)), acts, "TAKE"))
                s.cool = t + max(1, math.ceil(lat)) + (s.P["watch"] if s.spec["rollback"] else 5)
            else: s.cool = t + 1 + math.ceil(lat)

    def ppo_train_tick(s, t, obs, ph):
        s.rewards.append((t, step_reward(s.sim, ph, obs["ratio"])))
        ev = s.evidence(t, obs); acts = []
        if t < T0 or t % 2: return
        if "ddos" in ev["cls"]: acts += [dict(type="acl_add", target=h, params={}) for h in ev["att"]]
        if "misconfig" in ev["cls"]: acts.append(dict(type="config_revert", target=ev["mis"][0], params={}))
        if "link" in ev["cls"] or "congestion" in ev["cls"]:
            acts += s.build_actions([c for c in ev["cls"] if c in ("link", "congestion")], ev, obs, "graph", s.rng)
        if acts: s.pending.append((t + 1, acts, None))


def run_episode(topo, scn, cfg, seed, P, models, train_ctl=None, return_ctl=False):
    sim = Sim(topo, scn, seed); rng = np.random.default_rng(seed * 7919 + sum(map(ord, cfg)) % 1000)
    ctl = train_ctl(sim, rng) if train_ctl else Ctl(sim, cfg, P, rng, models)
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
    out = dict(topology=topo, scenario=scn, config=cfg, seed=seed, recovered=int(rec), throughput_recovery_s=trt, mttr_s=min(cl, T_END - T0),
               packet_loss_pct=float((1 - r[T0:]).mean() * 100), availability_pct=float(np.mean(avail) * 100),
               latency_p50_ms=float(np.nanpercentile(la, 50)), latency_p95_ms=float(np.nanpercentile(la, 95)),
               llm_calls=ctl.st["llm_calls"], tokens=ctl.st["tokens"], rollbacks=ctl.st["rollbacks"], plans_rejected=ctl.st["rejected"],
               halluc_generated=ctl.st["halluc_gen"], halluc_executed=ctl.st["halluc_exec"], actions_generated=ctl.st["actions_gen"],
               actions_executed=ctl.st["actions_exec"], reroutes=ctl.st["reroutes"], debate_invoked=ctl.st["debate"],
               diag_correct=ctl.st["diag_ok"] if ctl.st["diag_ok"] is not None else -1, ctl_ms=ctl.st["wall"] * 1000, fallback=int(ctl.fallback))
    return (out, ctl, sim) if return_ctl else out
