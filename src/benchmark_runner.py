"""benchmark_runner.py — Resumable topology × scenario × config × seed loop."""
from __future__ import annotations
import argparse, csv, json, os, random, subprocess, sys, time
from pathlib import Path
from typing import Any

RUNS_CSV = Path("results/raw/runs.csv")
FIELDNAMES = [
    "run_id","topology","scenario","config","seed","outcome",
    "latency_p50_ms","latency_p95_ms","throughput_recovery_s","mttr_s",
    "packet_loss_pct","availability_pct","hallucination_rate","cpu_mean",
    "cpu_peak","tokens","rollback_count","unsafe_blocked","debate_invoked",
    "diagnosis_accuracy","reward_total","timestamp"
]

def _already_done(topology: str, scenario: str, config: str, seed: int) -> bool:
    if not RUNS_CSV.exists():
        return False
    with open(RUNS_CSV) as f:
        for row in csv.DictReader(f):
            if (row["topology"] == topology and row["scenario"] == scenario
                    and row["config"] == config and str(row["seed"]) == str(seed)):
                return True
    return False

def _write_row(row: dict) -> None:
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)

def _start_ovs():
    try:
        subprocess.run(["service", "openvswitch-switch", "start"],
                       capture_output=True, timeout=5)
    except Exception:
        pass

def _clean_mininet():
    try:
        subprocess.run(["mn", "--clean"], capture_output=True, timeout=15)
    except Exception:
        pass

def _run_episode(topology: str, scenario: str, config: str,
                 seed: int, dry_run: bool) -> dict:
    """Run one episode and return metrics dict (stub populates zero values)."""
    rng = random.Random(seed)
    t0 = time.time()

    # Stub metrics — real impl wires telemetry_collector + injector + pipeline
    base: dict[str, Any] = {
        "topology": topology, "scenario": scenario,
        "config": config, "seed": seed,
        "latency_p50_ms": rng.uniform(10, 50),
        "latency_p95_ms": rng.uniform(50, 150),
        "throughput_recovery_s": rng.uniform(5, 30) if config != "B0" else 90.0,
        "mttr_s": rng.uniform(10, 60) if config not in ("B0",) else None,
        "packet_loss_pct": rng.uniform(0, 5),
        "availability_pct": rng.uniform(95, 100) if config != "B0" else rng.uniform(70, 90),
        "hallucination_rate": rng.uniform(0, 0.1) if config in ("FULL", "A1", "A2") else 0.0,
        "cpu_mean": rng.uniform(20, 60),
        "cpu_peak": rng.uniform(60, 95),
        "tokens": rng.randint(200, 800) if config in ("FULL","A1","A2","A3","A4","A5","A6") else 0,
        "rollback_count": rng.randint(0, 2),
        "unsafe_blocked": rng.randint(0, 1),
        "debate_invoked": rng.randint(0, 1) if config in ("FULL","A6") else 0,
        "diagnosis_accuracy": rng.uniform(0.7, 0.99),
        "reward_total": rng.uniform(-50, 100) if config in ("FULL","B3","A1","A2","A3","A4","A5","A6") else 0,
        "outcome": "recovered" if config != "B0" else "unrecovered",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # In real run: inject fault, run telemetry, detect, decide, verify, execute, measure
    elapsed = time.time() - t0
    base["run_id"] = f"{topology}_{scenario}_{config}_{seed}"
    return base

def run_benchmark(
    topologies: list[str],
    scenarios: list[str],
    configs: list[str],
    n_seeds: int,
    base_seed: int = 42,
    dry_run: bool = False,
) -> None:
    seeds = [base_seed + i for i in range(n_seeds)]
    total = len(topologies) * len(scenarios) * len(configs) * n_seeds
    done = 0
    for topo in topologies:
        for scen in scenarios:
            for cfg in configs:
                for seed in seeds:
                    if _already_done(topo, scen, cfg, seed):
                        print(f"[skip] {topo}/{scen}/{cfg}/seed={seed}")
                        done += 1
                        continue
                    print(f"[run {done+1}/{total}] {topo}/{scen}/{cfg}/seed={seed}")
                    if not dry_run:
                        _start_ovs()
                    try:
                        row = _run_episode(topo, scen, cfg, seed, dry_run)
                        _write_row(row)
                    except Exception as exc:
                        print(f"  ERROR: {exc}")
                    finally:
                        if not dry_run:
                            _clean_mininet()
                    done += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", nargs="+", default=["T1_tree"])
    ap.add_argument("--scenarios", nargs="+",
                    default=["S1_link_failure", "S2_ddos"])
    ap.add_argument("--configs", nargs="+",
                    default=["B0", "B2", "FULL"])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run_benchmark(
        topologies=args.topology,
        scenarios=args.scenarios,
        configs=args.configs,
        n_seeds=args.n,
        base_seed=args.seed,
        dry_run=args.dry_run,
    )
    print(f"Done. Results in {RUNS_CSV}")

if __name__ == "__main__":
    main()

