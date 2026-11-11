# PROJECT AGENT HANDOFF CONTEXT & SYSTEM PROMPT

## Project Title
**CL-ANM (Closed-Loop Autonomous Network Management)** — Phase 3 Computer Networks Project.

## Ground Truth Architecture & Constraints
- **Hardware/Host:** Windows 11 host with WSL2 Ubuntu 26.04. AMD Radeon RX 580 (Polaris, 8GB VRAM). **NO CUDA**.
- **PyTorch/ML:** DirectML (`torch_directml`) on Windows or CPU in WSL2. Do NOT install/use CUDA packages (`torch.cuda`, bitsandbytes CUDA).
- **LLM Strategy:**
  1. Primary: Groq Cloud API (`llama-3.3-70b-versatile` or `qwen-2.5-32b`) using `GROQ_API_KEY` from `.env`.
  2. Local fallback: `llama.cpp` Vulkan backend (`C:\Users\Sultan\Downloads\llama-b12255-bin-win-vulkan-x64\llama-cli.exe` with flags `-ngl 99 -c 2048`).
  3. Caching: All LLM queries hashed and cached into `results/checkpoints/llm_cache.json`.
- **Reproducibility & Standards:** Global seed 42. Strict append-only runs recorded into `results/raw/runs.csv`. No fabricated values/numbers. Every module has CLI + typehints + pytest smoke test. Deviations in `CHANGES.md`, design choices in `DECISIONS.md`.

---

## Current Status & File Manifest

### Completed Code Modules (`src/`):
- `topologies.py`: T1 Tree, T2 Abilene, T3 Fat-Tree (exposing `build_networkx()`, `build_mininet()`, and `get_inventory()`).
- `faults.py`: S1 (link failure), S2 (DDoS), S3 (misconfig), S4 (cascading), S5 (DDoS + link failure) with JSONL ground truth.
- `telemetry_collector.py`: OVS port/flow polling (1s), robust IQR scaling, reset handling, staleness masking, flow entropy.
- `log_parser.py`: Regex parsing + Drain3 template clustering.
- `detector.py`: HistGradientBoosting classifier predicting fault posterior $p_t$.
- `rag.py`: MiniLM retrieval over OVS/Ryu/FRR docs with OOD detection.
- `verifier.py`: 3-layer verification (L1 Schema/Whitelist, L2 NetworkX Invariants, L3 Dry-Run Simulation) with pre-check hallucination detection.
- `executor.py`: Snapshot -> Apply (with backoff/retry) -> 10s Watch -> Commit or Rollback + Fallback chain (RL -> Rules -> Escalation) + Circuit Breaker.
- `slm_agent.py`: Groq/GGUF/OpenAI caller with JSON repair and verifier re-prompt loop (max $N=3$).
- `debate.py`: Proposer, Critic, Security, and Safety roles with confidence-weighted voting.
- `graph_encoder.py`: PyG / PyTorch 2-layer GraphSAGE (64-d) feature extractor for SB3 + Flat MLP baseline.
- `fluid_env.py`: Gymnasium flow-level NetworkX simulator for PPO pre-training.
- `ppo_agent.py`: SB3 PPO setup with Mahalanobis OOD guard and checkpoint resuming.
- `benchmark_runner.py`: Resumable matrix runner across topologies x scenarios x configs x seeds.

### Evaluation & Training Modules:
- `evaluation/stats.py`: Shapiro-Wilk, Levene, ANOVA/Kruskal-Wallis, Welch t-test with Holm correction, Kaplan-Meier estimator.
- `evaluation/compare_literature.py`: Benchmarks against Makiyah et al., Mwangi et al. (OMARF), and Lira et al.
- `evaluation/failure_taxonomy.py`: Categorizes failed episodes from audit logs into taxonomy tables.
- `plots/make_plots.py`: Generates MTTR CDF, ablation bars, latency vs throughput, KM curves, overhead, and hallucination rates into PDF/PNG.
- `train/train_detector.py`: Trains fault detector from JSONL or synthetic traces.
- `train/train_ppo.py`: Pre-trains PPO agent on `FluidNetEnv` with checkpointing.
- `train/train_slm.py`: Offline SLM evaluation on intent dataset using Groq API.

### Environment & Virtual Environment:
- WSL2 root environment configured with OVS 3.7.1 and Mininet 2.3.0.
- Venv path: `/mnt/h/CNET Project/.venv` (Python 3.11.18 via uv).
- Mininet library linked via `/mnt/h/CNET Project/.venv/lib/python3.11/site-packages/mininet.pth`.

---

## What Left to Complete the Deliverable

1. **WSL Dependencies Finish:** Run remaining package installation in WSL:
   ```bash
   cd "/mnt/h/CNET Project"
   /root/.local/bin/uv pip install networkx pyyaml pydantic pytest psutil matplotlib scikit-learn xgboost drain3 pandas scipy statsmodels scikit-posthocs pingouin lifelines tabulate gymnasium stable-baselines3 sentence-transformers groq openai torch torchvision
   ```
2. **Intent Dataset Generation:**
   ```bash
   .venv/bin/python data/gen_intent_dataset.py --total 8000 --seed 42 --out-dir data
   ```
3. **Detector & PPO Quick Pre-training:**
   ```bash
   .venv/bin/python train/train_detector.py --samples 2000
   .venv/bin/python train/train_ppo.py --steps 5000
   ```
4. **Dry Run Benchmark Execution (N=2):**
   ```bash
   wsl -u root bash -c "cd '/mnt/h/CNET Project' && /etc/init.d/openvswitch-switch start && .venv/bin/python src/benchmark_runner.py --topology T1_tree --scenarios S1_link_failure S2_ddos --configs B0 B2 FULL --n 2 --seed 42 --dry-run"
   ```
5. **Run Evaluation, Literature Comparison & Plots:**
   ```bash
   .venv/bin/python evaluation/stats.py --input results/raw/runs.csv --output results/tables/
   .venv/bin/python evaluation/compare_literature.py --input results/raw/runs.csv --output results/comparison_existing.csv
   .venv/bin/python evaluation/failure_taxonomy.py --audit-log results/logs/audit.jsonl --output-dir results/tables/
   .venv/bin/python plots/make_plots.py --input results/raw/runs.csv --output results/figures/
   ```
6. **Project Root Files Needed:**
   - `Makefile` (wiring all targets above).
   - `README.md` (clean run instructions, dependencies, and architecture overview).

