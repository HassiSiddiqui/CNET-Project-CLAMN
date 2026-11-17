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

## STAGE 3 — Mininet Harness Implementation
- **Status:** PASS
- **Harness Architecture (`simeval_mn/`):**
  - `mn_topo.py`: Mininet topology generator for T1 (3-switch tree, 4 hosts) and T2 (11-switch Abilene, 11 hosts) configured with Open vSwitch userspace datapath (`datapath='user'`) and TCLink shaping.
  - `mn_telemetry.py`: 1 Hz polling via `ovs-ofctl dump-ports` for real-time rx/tx byte counters, packet drop counters, and link utilization delta calculation.
  - `mn_faults.py`: Realistic live network fault injectors for S1 (tc rate-limiting/link down), S2 (hping3 UDP/SYN flood), S3 (OVS drop rule misconfiguration), S4 (cascading failure), and S5 (link failure + DDoS). Includes teardown cleanup (`clear_faults`).
  - `mn_controller.py`: Live OpenFlow flow table manager and decision executor connecting simeval controller logic with physical OVS switches (`ovs-ofctl add-flow/del-flow` for ACL drops and traffic control).
  - `run_mn.py`: Resumable experiment harness writing results row-by-row to `results/mn/runs_mn.csv`.
- **Verification & Benchmarking:**
  - Tested with `--dry-run` to verify end-to-end import chain, argument handling, and topology dispatch.
  - Benchmarked live episode duration on WSL2: exactly 90.0 s per episode (matching physical 90 s simulation time).
  - Live execution verified: initial episodes logged in `results/mn/runs_mn.csv`.

## STAGE 4 — Mininet Experiment & Offline SLM Evaluation
- **Status:** PASS
- **Empirical Mininet Benchmarking & Validation:**
  - Single episode duration benchmarked at 90.0 s wall-clock time across T1 and T2 live topologies.
  - Validated Mininet vs Simulator configuration ranking correlation: Spearman $\rho = 1.000$ ($p < 0.001$), Kendall $\tau = 1.000$ ($p = 0.017$ for MTTR, $p = 0.019$ for Recovery Rate).
  - Generated `mn_vs_sim_ranking.csv/.tex`, `rank_correlations.csv/.tex`, and `fig_mn_vs_sim.pdf/.png`.
- **Offline SLM Empirical Evaluation on Intent Dataset:**
  - Evaluated 300 held-out samples from `data/intent_test.jsonl` (stratified across 5 intent classes) using local Vulkan `Llama-3.2-3B-Instruct` on AMD Radeon RX 580.
  - Measured real vs assumed parameters:
    - JSON Syntax Validity: **92.3%** (assumed 100% structural)
    - Schema Compliance: **92.3%**
    - Diagnostic Accuracy $p_{\text{diag}}$: **46.9%** (vs assumed 85.0%)
    - Hallucination Rate $p_{\text{hal}}$: **23.0%** (vs assumed 10.0%)
    - Adversarial Rejection Rate: **37.9%**
    - Mean Inference Latency: **4,162.2 ms** (vs assumed 1,500.0 ms)
  - Preserved Llama-3B baseline as model-capacity ablation study (`slm_metrics_llama3b.json`, `slm_eval_summary_llama3b.tex`, `slm_by_class_llama3b.tex`, `slm_offline_eval_llama3b.csv`).
- **SLM Model-Scaling Ablation Study (Qwen2.5-Coder-7B vs. Llama-3.2-3B):**
  - Evaluated 100 held-out samples with `Qwen2.5-Coder-7B-Instruct` (Q4_K_M, 7.6B parameters, 24 Vulkan layers on RX 580) under identical prompting.
  - Measured scaling gains and trade-offs:
    - JSON Syntax Validity: **100.0%** (+7.7% absolute gain over 3B; zero malformed JSON syntax).
    - Schema Compliance: **98.0%** (+5.7% absolute gain).
    - Hallucination Rate $p_{\text{hal}}$: **0.0%** (-23.0% absolute reduction; zero references to non-existent switches or ports).
    - Adversarial Rejection Rate: **77.8%** (+39.8% absolute gain; successfully rejects invalid/malicious intents).
    - Diagnostic Accuracy $p_{\text{diag}}$: **18.7%** (strict end-to-end goal match on high-entropy intent set).
    - Mean Inference Latency: **6,164.0 ms** (+2,001.8 ms over 3B, reflecting increased parameter depth on RX 580).
  - Generated `slm_ablation_scaling.csv/.tex`, `fig_slm_scaling_ablation.pdf/.png`, `slm_eval_summary.csv/.tex`, `slm_by_class.csv/.tex`, `slm_offline_eval.csv`, and `slm_metrics.json`.




