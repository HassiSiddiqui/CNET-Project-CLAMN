"""run.py — evaluation matrix on held-out eval seeds (1001..1030), plus SLM-error sensitivity sweep."""
import sys, pickle, argparse, time
import pandas as pd, numpy as np
from .netsim import APPLICABLE
from .controllers import run_episode, DEF_P, CFGS

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); ap.add_argument("--models", default="results/sim/models.pkl")
ap.add_argument("--out", default="results/sim/runs_sim.csv"); ap.add_argument("--sweep", action="store_true"); a = ap.parse_args()
M = pickle.load(open(a.models, "rb")); t0 = time.time(); rows = []
seeds = range(1001, 1001 + a.n)
if not a.sweep:
    for topo, scs in APPLICABLE.items():
        for sc in scs:
            for cfg in CFGS:
                for s in seeds: rows.append(run_episode(topo, sc, cfg, s, DEF_P, M))
else:  # sensitivity of FULL vs B1 to the *assumed* SLM error rates (Abilene, all scenarios)
    for pd_ in (.6, .75, .85, .95):
        for ph in (0., .1, .2, .3):
            P = dict(DEF_P, p_diag=pd_, p_hal=ph)
            for sc in APPLICABLE["T2_abilene"]:
                for cfg in ("B1", "FULL", "A1"):
                    for s in seeds: rows.append(dict(run_episode("T2_abilene", sc, cfg, s, P, M), p_diag=pd_, p_hal=ph))
pd.DataFrame(rows).to_csv(a.out, index=False); print(len(rows), "episodes in", round(time.time() - t0), "s ->", a.out)
