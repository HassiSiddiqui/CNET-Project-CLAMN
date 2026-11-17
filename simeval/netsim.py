"""netsim.py — 1 Hz flow-level network simulator with fault injection and noisy telemetry.

Physics (all mechanism-driven, nothing hard-coded per configuration):
  * every link has capacity/delay from src/topologies.py; legit flows are single-path;
  * link scale s_e = min(1, cap_e/load_e); flow throughput multiplier = min_e s_e along its path;
  * queueing delay grows with utilisation; down links / blackholing switches give zero goodput;
  * optional overload-collapse (S4/S5): a link above 100% util for 8 s fails.
"""
from __future__ import annotations
import os, sys
from dataclasses import dataclass
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.topologies import build_networkx  # noqa: E402

T_END, T0 = 90, 20
SCEN = ["S1_link_failure", "S2_ddos", "S3_misconfig", "S4_cascading", "S5_combined"]
TOPOS = ["T1_tree", "T2_abilene", "T3_fattree"]
APPLICABLE = {  # T1 (3-switch tree) has no redundant paths: link-failure scenarios are partitioning there
    "T1_tree": ["S2_ddos", "S3_misconfig"],
    "T2_abilene": SCEN, "T3_fattree": SCEN}
LOAD = {"S1_link_failure": .50, "S2_ddos": .50, "S3_misconfig": .50, "S4_cascading": .60, "S5_combined": .55}


class Topo:
    _cache: dict = {}

    def __new__(cls, name):
        if name in cls._cache:
            return cls._cache[name]
        o = super().__new__(cls); cls._cache[name] = o; o._init(name); return o

    def _init(self, name):
        G = build_networkx(name)
        self.name, self.G = name, G
        self.nodes = sorted(G.nodes()); self.nid = {n: i for i, n in enumerate(self.nodes)}
        self.hosts = [n for n in self.nodes if G.nodes[n].get("role") == "host"]
        self.switches = [n for n in self.nodes if n not in set(self.hosts)]
        self.edges = [tuple(sorted(e)) for e in G.edges()]
        self.eid = {}
        for i, (u, v) in enumerate(self.edges):
            self.eid[(u, v)] = i; self.eid[(v, u)] = i
        self.E = len(self.edges)
        self.cap = np.array([G.edges[e]["capacity"] for e in self.edges], float)
        self.delay = np.array([G.edges[e]["delay"] for e in self.edges], float)
        self.sw_edges = [i for i, (u, v) in enumerate(self.edges) if u in self.switches and v in self.switches]
        # non-bridge switch-switch links (removal keeps the network connected)
        br = set(tuple(sorted(b)) for b in nx.bridges(G))
        self.redundant_edges = [i for i in self.sw_edges if self.edges[i] not in br]
        N = len(self.nodes)
        A = np.zeros((N, N))
        for u, v in self.edges:
            A[self.nid[u], self.nid[v]] = A[self.nid[v], self.nid[u]] = 1
        deg = np.maximum(A.sum(1, keepdims=True), 1)
        self.A_mean = A / deg
        self.N = N

    def path_edges(self, p):
        return np.array([self.eid[(p[i], p[i + 1])] for i in range(len(p) - 1)], int)

    def alive_paths(self, src, dst, down: set, k=3):
        """k shortest (by delay) simple paths avoiding links in `down`."""
        H = self.G.copy()
        H.remove_edges_from([self.edges[e] for e in down])
        if src not in H or dst not in H or not nx.has_path(H, src, dst):
            return []
        out = []
        for p in nx.shortest_simple_paths(H, src, dst, weight="delay"):
            out.append(p)
            if len(out) >= k: break
        return out


@dataclass
class Flow:
    src: str; dst: str; d: float; path: list; pe: np.ndarray


class Sim:
    def __init__(self, topo: str, scenario: str, seed: int, collapse: bool | None = None, noise=0.02):
        self.T = Topo(topo); self.scn = scenario; self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.collapse = scenario in ("S4_cascading", "S5_combined") if collapse is None else collapse
        T = self.T
        # ----- traffic: legit flows on delay-shortest paths -----
        H = T.hosts; nf = {"T1_tree": 4, "T2_abilene": 14, "T3_fattree": 16}[topo]
        self.flows: list[Flow] = []
        seen = set()
        while len(self.flows) < nf:
            s, d = self.rng.choice(H, 2, replace=False)
            if (s, d) in seen: continue
            if topo == "T1_tree" and len(self.flows) < 2 and False: pass
            seen.add((s, d))
            p = T.alive_paths(s, d, set(), 1)[0]
            self.flows.append(Flow(s, d, float(self.rng.uniform(0.5, 1.5)), p, T.path_edges(p)))
        load = np.zeros(T.E)
        for f in self.flows: load[f.pe] += f.d
        scale = LOAD[scenario] / (load / T.cap).max()
        for f in self.flows: f.d *= scale
        self.d0 = np.array([f.d for f in self.flows])
        self.base_tx = {h: sum(f.d for f in self.flows if f.src == h) for h in H}
        # ----- fault state (hidden) -----
        self.cap_mult = np.ones(T.E); self.over = np.zeros(T.E, int)
        self.flap = None; self.attackers = {}; self.blocked = set(); self.mis_node = None
        self.mis_flows = set(); self.reverted = False; self.events = []
        self._att_cache = {}
        self.t = 0; self._plan_fault()
        self.baseline_rows = []

    # ------------------------------------------------------------------ faults
    def _busiest_redundant_edge(self):
        T = self.T; cnt = np.zeros(T.E)
        for f in self.flows: cnt[f.pe] += 1
        cand = [e for e in T.redundant_edges if cnt[e] > 0]
        return max(cand, key=lambda e: (cnt[e], self.rng.random())) if cand else None

    def _plan_fault(self):
        T, r = self.T, self.rng; sc = self.scn
        self.fault = {}
        if sc in ("S1_link_failure", "S5_combined", "S4_cascading"):
            if sc == "S1_link_failure":
                cnt = np.zeros(T.E)
                for f in self.flows: cnt[f.pe] += 1
                cand = [e for e in T.redundant_edges if cnt[e] > 0]
                e = int(r.choice(cand)); mode = str(r.choice(["hard", "flapping", "degraded"]))
            else:
                e = self._busiest_redundant_edge(); mode = "hard"
            self.fault.update(link=e, link_mode=mode, link_t=T0)
        if sc in ("S2_ddos", "S5_combined"):
            cnt = {}
            for f in self.flows: cnt[f.dst] = cnt.get(f.dst, 0) + 1
            victim = max(cnt, key=lambda h: (cnt[h], r.random()))
            others = [h for h in T.hosts if h != victim]
            na = min(3, len(others)); att = [str(x) for x in r.choice(others, na, replace=False)]
            inten = str(r.choice(["low", "medium", "high"]))
            tot = {"low": .8, "medium": 1.2, "high": 2.0}[inten] * T.cap[T.eid[(victim, next(iter(T.G[victim])))]]
            self.fault.update(victim=victim, att=att, att_rate=tot / na, ddos_t=T0 + (5 if sc == "S5_combined" else 0), intensity=inten)
        if sc == "S3_misconfig":
            cnt = {}
            for f in self.flows:
                for n in f.path[1:-1]: cnt[n] = cnt.get(n, 0) + 1
            cand = [n for n in cnt if cnt[n] >= 2] or list(cnt)
            m = str(r.choice(cand)); via = [i for i, f in enumerate(self.flows) if m in f.path[1:-1]]
            hit = {i for i in via if r.random() < .7} or {via[0]}
            self.fault.update(mis=m, mis_flows=hit, mis_t=T0, mis_mode=str(r.choice(["wrong_next_hop", "broad_acl", "missing_route", "mtu_mismatch"])))

    def _inject(self):
        t, F = self.t, self.fault
        if "link" in F and t == F["link_t"]:
            e, mode = F["link"], F["link_mode"]
            if mode == "hard": self.cap_mult[e] = 0
            elif mode == "degraded": self.cap_mult[e] = 0.2
            else: self.flap = e
        if self.flap is not None:
            self.cap_mult[self.flap] = 0.0 if ((t - T0) // 2) % 2 == 0 else 1.0
        if "att" in F and t == F["ddos_t"]:
            for a in F["att"]: self.attackers[a] = F["att_rate"]
        if "mis" in F and t == F["mis_t"]:
            self.mis_node, self.mis_flows = F["mis"], set(F["mis_flows"])

    # ------------------------------------------------------------------ physics
    def physics(self):
        T = self.T; cap = T.cap * self.cap_mult
        load = np.zeros(T.E)
        for f in self.flows: load[f.pe] += f.d
        att_pe = {}
        for a, r in self.attackers.items():
            if a in self.blocked: continue
            if a not in self._att_cache:
                self._att_cache[a] = T.path_edges(T.alive_paths(a, self.fault["victim"], set(), 1)[0])
            att_pe[a] = self._att_cache[a]; load[att_pe[a]] += r
        with np.errstate(divide="ignore", invalid="ignore"):
            util = np.where(cap > 0, load / np.maximum(cap, 1e-9), np.where(load > 0, np.inf, 0.0))
            s = np.where(load > cap, cap / np.maximum(load, 1e-9), 1.0)
        mult = np.array([s[f.pe].min() for f in self.flows])
        for i, f in enumerate(self.flows):
            if f.src in self.blocked and f.src not in self.attackers: mult[i] = 0  # ACL on attack signature spares an attacker's own legit flows; a legit host is fully blocked
            if self.mis_node and not self.reverted and i in self.mis_flows and self.mis_node in f.path[1:-1]: mult[i] = 0
        q = np.where(util < .95, 20 * np.minimum(util, .95) / (1 - np.minimum(util, .95)), 100.0); q = np.minimum(q, 100)
        lat = np.array([(T.delay[f.pe] + q[f.pe]).sum() for f in self.flows])
        good = self.d0 * 0 + np.array([f.d for f in self.flows]) * mult
        return dict(util=util, load=load, cap=cap, mult=mult, good=good, lat=lat)

    def step(self):
        """advance one second; returns physics dict for the second that just elapsed."""
        self._inject()
        ph = self.physics(); self.last = ph
        if self.collapse and self.t >= T0:
            self.over = np.where(ph["util"] > 1.0, self.over + 1, 0)
            for e in np.where((self.over >= 8) & (self.cap_mult > 0))[0]:
                self.cap_mult[e] = 0; self.events.append((self.t, "collapse", int(e)))
        self.t += 1
        return ph

    # ------------------------------------------------------------------ telemetry / actuation
    def observe(self, ph=None):
        ph = ph or self.last; T = self.T; r = self.rng
        util = ph["util"].copy(); fin = np.isfinite(util)
        util_obs = np.where(fin, util + r.normal(0, .02, T.E), 5.0).clip(0, 5)
        down = (ph["cap"] <= 0)
        good = ph["good"] * (1 + r.normal(0, .02, len(self.flows)))
        tx = {h: self.base_tx[h] * (1 + r.normal(0, .03)) for h in T.hosts}
        for a, rt in self.attackers.items():
            if a not in self.blocked: tx[a] += rt
        node_drop = {n: 0.0 for n in T.switches}
        if self.mis_node and not self.reverted:
            node_drop[self.mis_node] = sum(self.flows[i].d for i in self.mis_flows if self.mis_node in self.flows[i].path[1:-1])
        return dict(t=self.t, util=util_obs, down=down, good=np.maximum(good, 0), tx=tx, node_drop=node_drop,
                    ratio=float(max(good.sum(), 0) / self.d0.sum()))

    def set_path(self, i, path):
        T = self.T
        if any((path[j], path[j + 1]) not in T.eid for j in range(len(path) - 1)): return False  # nonexistent link: rejected by switch
        self.flows[i].path = list(path); self.flows[i].pe = T.path_edges(path); return True

    def snapshot(self):
        return dict(paths=[(list(f.path)) for f in self.flows], blocked=set(self.blocked), reverted=self.reverted)

    def restore(self, s):
        for f, p in zip(self.flows, s["paths"]):
            f.path = list(p); f.pe = self.T.path_edges(p)
        self.blocked = set(s["blocked"]); self.reverted = s["reverted"]

    def clone(self):
        import copy
        c = copy.copy(self)
        c.flows = [Flow(f.src, f.dst, f.d, list(f.path), f.pe.copy()) for f in self.flows]
        c.cap_mult = self.cap_mult.copy(); c.over = self.over.copy(); c.blocked = set(self.blocked)
        c.attackers = dict(self.attackers); c.events = []; c.rng = np.random.default_rng(self.rng.integers(1 << 30))
        return c
