# CL-ANM Experiment & Submission Handoff Log

## STAGE 0 — Environment Gate
- **Status:** PASS
- **Host / Virtualization:** Windows 11 host with WSL2 Ubuntu 24.04/26.04 environment.
- **Mininet:** 2.3.0 installed and verified.
- **Open vSwitch:** 3.7.1 installed; kernel/userspace datapaths operational.
- **iperf3 / hping3 / scapy:** Verified operational.
- **Ping test:** `PingAll loss: 0.0%` (2/2 received).
- **Throughput test:** `iperf3 -c 10.0.0.2 -t 3` achieved 21.7 Gbits/sec in Mininet namespace.

## STAGE 1 — Repo Hygiene
- **Status:** PASS
- **Stub Archive:** Moved `results/raw/runs.csv` and previous stub analysis tables into `results/_stub_DO_NOT_USE/` with descriptive `README.md`.
- **Topology Clarification:** T1 documented as 3 switches + 4 hosts = 7 nodes, 6 bidirectional links.

## STAGE 2 — Simulator Work (Cheap & Honest Execution)
- **Status:** PASS
- **Held-out Unseen Faults:** Evaluated 1,440 episodes across U1–U4 (hyper-DDoS 3.5x, fast cascade collapse 4s, dual-link failure, compound storm) on held-out seeds (2001–2030). Generated `unseen_faults.csv/.tex` and `unseen_by_scenario.csv/.tex`.
- **Per-cell Breakdown:** Generated appendix tables `per_cell_recovered.csv/.tex` and `per_cell_mttr.csv/.tex` across all 12 topology × scenario cells.
- **Post-Hoc Variant CL-ANM-v2:** Implemented partial rollback, pipelined LLM/twin dry-run overlap, and L3 threshold >= 0. Evaluated on the identical 30 eval seeds (1001–1030). Generated `v2_comparison.csv/.tex` and `v2_scenario_breakdown.csv/.tex`.
- **Figures:** Generated `fig_v2_comparison`, `fig_unseen_faults`, and `fig_v2_mttr_scenario` alongside re-verified core simulator figures (`fig_ablation`, `fig_cdf_km`, `fig_lat_loss`, `fig_overhead`, `fig_scenarios`, `fig_sensitivity`).

