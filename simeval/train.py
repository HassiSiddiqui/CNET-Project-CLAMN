"""train.py — trains the PPO path policy (GraphSAGE and flat) and the single DQN on *training* seeds only."""
import sys, time, pickle, argparse
import numpy as np
from .netsim import APPLICABLE, T0
from .controllers import Ctl, run_episode, DEF_P
from .nn import PathPolicy, DQN

TE = [(t, s) for t, ss in APPLICABLE.items() for s in ss if t != "T1_tree" and s != "S3_misconfig"]
ALL = [(t, s) for t, ss in APPLICABLE.items() for s in ss]


def train_ppo(use_graph, iters, eps, log):
    pol = PathPolicy(use_graph, seed=0); rng = np.random.default_rng(1); seed = 1
    for it in range(iters):
        batch, rets = [], []
        for _ in range(eps):
            topo, scn = TE[rng.integers(len(TE))]; seed += 1
            models = dict(ppo_graph=pol, ppo_flat=pol)
            def mk(sim, r): return Ctl(sim, "ppo_train", DEF_P, r, models, train=True, oracle=True)
            out, ctl, sim = run_episode(topo, scn, "ppo_train", seed, DEF_P, models, train_ctl=mk, return_ctl=True)
            rw = dict(ctl.rewards)
            groups = {}
            for d in ctl.dec_buf: groups.setdefault(d["t"], []).append(d)
            for t, ds in groups.items():
                r = np.mean([rw[x] for x in range(t + 1, t + 4) if x in rw] or [0]) - .1 * len(ds) / len(sim.flows)
                for d in ds:
                    batch.append(dict(feats=d["feats"], a=d["a"], logp=d["logp"], adv=r - d["v"], ret=r)); rets.append(r)
        if batch: pol.update(batch, rng=rng)
        log.append((it, len(batch), float(np.mean(rets)) if rets else 0.0))
        if it % 5 == 0: print(f"[ppo graph={use_graph}] it {it} decisions {len(batch)} mean_r {log[-1][2]:.3f}", flush=True)
    return pol


def train_dqn(episodes, log):
    d = DQN(seed=0); rng = np.random.default_rng(2); seed = 50000
    for ep in range(episodes):
        topo, scn = ALL[rng.integers(len(ALL))]; seed += 1; d.eps = max(.05, 1 - ep / (.6 * episodes))
        out = run_episode(topo, scn, "B3", seed, DEF_P, dict(dqn=d), train_ctl=lambda sim, r: Ctl(sim, "B3", DEF_P, r, dict(dqn=d), train=True))
        if ep % 5 == 0: d.sync()
        log.append((ep, out["packet_loss_pct"]))
        if ep % 100 == 0: print(f"[dqn] ep {ep} eps {d.eps:.2f} loss% {out['packet_loss_pct']:.1f}", flush=True)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=40); ap.add_argument("--eps", type=int, default=24)
    ap.add_argument("--dqn", type=int, default=600); ap.add_argument("--out", default="results/sim/models.pkl"); a = ap.parse_args()
    logs = dict(ppo_graph=[], ppo_flat=[], dqn=[]); t = time.time()
    g = train_ppo(True, a.iters, a.eps, logs["ppo_graph"]); f = train_ppo(False, a.iters, a.eps, logs["ppo_flat"])
    q = train_dqn(a.dqn, logs["dqn"]); q.eps = 0
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pickle.dump(dict(ppo_graph=g, ppo_flat=f, dqn=q, logs=logs), open(a.out, "wb")); print("done", time.time() - t)
