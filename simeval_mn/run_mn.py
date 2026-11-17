"""simeval_mn/run_mn.py — Stage 3 Mininet harness runner.

Runs Mininet episodes for Tier M1 (T1 + T2, S1/S2/S3/S5, B0/B1/B2/B3/FULL, N=5 per cell).
Results written to results/mn/runs_mn.csv (resumable: skips existing seed/cfg/topo/scn combos).

HONESTY NOTE: Ctl decisions use the simeval physics (SLM error model); real Mininet
telemetry is collected and logged separately to results/mn/mn_telemetry_log.jsonl.
The 'recovered' metric is from the simulator physics, not raw Mininet throughput.
This is disclosed in the paper as a hybrid evaluation (Ctl integrated with real
switch-level execution, physics from calibrated flow-level simulator).

Usage (as root in WSL2):
    cd /mnt/h/CNET Project
    python3 simeval_mn/run_mn.py [--n 5] [--topos T1_tree T2_abilene] [--dry-run]
"""
import os, sys, argparse, csv, json, time, pickle, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

OUT_CSV = "results/mn/runs_mn.csv"
OUT_TEL = "results/mn/mn_telemetry_log.jsonl"
os.makedirs("results/mn", exist_ok=True)

FIELDNAMES = [
    "topology","scenario","config","seed","recovered","throughput_recovery_s","mttr_s",
    "packet_loss_pct","availability_pct","actions_applied","actions_accepted","wall_s","fallback"
]

TIER_M1_TOPOS = ["T1_tree", "T2_abilene"]
TIER_M1_SCENS = ["S1_link_failure", "S2_ddos", "S3_misconfig", "S5_combined"]
TIER_M1_CFGS  = ["B0", "B1", "B2", "B3", "FULL"]

# Scenarios not applicable to T1 (no redundant paths for link failure / S5)
APPLICABLE_MN = {
    "T1_tree":    ["S2_ddos", "S3_misconfig"],
    "T2_abilene": TIER_M1_SCENS,
}


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for row in csv.DictReader(f):
            done.add((row["topology"], row["scenario"], row["config"], row["seed"]))
    return done


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDNAMES})


def run_one(topo_name, scn, cfg, seed, models, dry_run=False):
    """Spin up Mininet, inject fault, run episode, tear down. Returns result dict."""
    from mininet.log import setLogLevel
    from mininet.node import OVSSwitch
    setLogLevel("warning")

    from simeval_mn.mn_topo import build_net
    from simeval_mn.mn_faults import INJECTORS, clear_faults
    from simeval_mn.mn_controller import install_l2_learning, MininetEpisodeRunner

    if dry_run:
        # Return a stub result without starting Mininet (for testing)
        return dict(topology=topo_name, scenario=scn, config=cfg, seed=seed,
                    recovered=1, throughput_recovery_s=15, mttr_s=20,
                    packet_loss_pct=5.0, availability_pct=94.0,
                    actions_applied=2, actions_accepted=2, wall_s=0.1, fallback=0)

    net = build_net(topo_name)
    result = None
    try:
        net.start()
        install_l2_learning(net)
        time.sleep(2)   # Let OVS converge

        # Inject fault at T=20
        import random
        rng = random.Random(seed)
        inj_fn = INJECTORS.get(scn)
        fault_info = {"scenario": scn}
        if inj_fn and cfg != "B0":
            fault_info = inj_fn(net, rng=rng)

        runner = MininetEpisodeRunner(net, cfg, models, seed, T_total=90, T_fault=20)
        result = runner.run(fault_info)

    except Exception as e:
        traceback.print_exc()
        result = dict(topology=topo_name, scenario=scn, config=cfg, seed=seed,
                      recovered=0, throughput_recovery_s=70, mttr_s=70,
                      packet_loss_pct=100.0, availability_pct=0.0,
                      actions_applied=0, actions_accepted=0,
                      wall_s=0.0, fallback=1,
                      error=str(e))
    finally:
        try:
            clear_faults(net)
            net.stop()
        except Exception:
            pass

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",     type=int, default=5,  help="Seeds per cell (Tier M1=5)")
    ap.add_argument("--topos", nargs="+", default=TIER_M1_TOPOS)
    ap.add_argument("--scens", nargs="+", default=TIER_M1_SCENS)
    ap.add_argument("--cfgs",  nargs="+", default=TIER_M1_CFGS)
    ap.add_argument("--models",default="results/sim/models.pkl")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip Mininet, produce stub rows (for CI / import testing)")
    a = ap.parse_args()

    models = pickle.load(open(a.models, "rb"))
    done = load_done(OUT_CSV)

    total = sum(
        len(a.cfgs) * a.n
        for t in a.topos for s in APPLICABLE_MN.get(t, a.scens)
        if s in a.scens
    )
    completed = 0; t0 = time.time()
    print(f"Tier M1: {total} episodes planned, {len(done)} already done")

    for topo in a.topos:
        applicable = [s for s in a.scens if s in APPLICABLE_MN.get(topo, a.scens)]
        for scn in applicable:
            for cfg in a.cfgs:
                for seed in range(3001, 3001 + a.n):
                    key = (topo, scn, cfg, str(seed))
                    if key in done:
                        completed += 1; continue
                    try:
                        row = run_one(topo, scn, cfg, seed, models, dry_run=a.dry_run)
                        append_row(OUT_CSV, row)
                        done.add(key); completed += 1
                        print(f"[{completed}/{total}] {topo} {scn} {cfg} s{seed} "
                              f"rec={row.get('recovered')} mttr={row.get('mttr_s')} "
                              f"wall={row.get('wall_s')}s", flush=True)
                    except Exception as e:
                        print(f"ERROR {topo} {scn} {cfg} s{seed}: {e}", flush=True)

    print(f"Done. {completed} episodes in {time.time()-t0:.1f}s -> {OUT_CSV}")


if __name__ == "__main__":
    main()
