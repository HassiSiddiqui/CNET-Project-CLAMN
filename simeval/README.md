# simeval — flow-level evaluation harness (replaces the stubbed benchmark)
`src/benchmark_runner.py::_run_episode` generates random numbers (CHG-07); do NOT use `results/raw/runs.csv` for any claim.
Reproduce: `python -m simeval.train --iters 40 --eps 30 --dqn 1000` -> `python -m simeval.run --n 30` -> `python -m simeval.run --n 15 --sweep --out results/sim/runs_sweep.csv` -> `python -m simeval.analyze`.
Assumed (not measured): SLM error model (p_diag, p_hal, confidence calibration), LLM latency, twin noise. Mininet episodes were NOT run.
