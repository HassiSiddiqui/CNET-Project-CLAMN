# CL-ANM Project Checkpoint (CHECKPOINT.md)

**Saved At:** 2026-11-11 04:08 (Local Time)  
**Status:** Milestone 1 & 2 Core Infrastructure Completed; Milestone 3 In-Progress. Ready for Seamless Resume.

---

## 1. Summary of Completed Work

### Environment & Testbed Provisioning (Executed & Verified)
- **Linux SDN Subsystem (WSL2 Ubuntu):** Fully configured with passwordless root access for Mininet.
- **Open vSwitch (OVS 3.7.1):** Installed and operational (`service openvswitch-switch start`).
- **Mininet 2.3.0:** Verified end-to-end with live OpenFlow 1.3 switch creation and network ping (`0% dropped, 2/2 received`).
- **Python 3.11 Runtime:** Python 3.11.18 installed and linked to `/usr/local/bin/python3.11` alongside fast package manager `uv`.
- **Mininet Python Bindings:** Configured via `.pth` in Python 3.11 environment.

### Hardware & LLM Backend Configuration
- **Target GPU:** AMD Radeon RX 580 2048SP (8GB VRAM, Polaris architecture).
- **Execution Rule:** `CUDA = FALSE` (Strictly enforced: no `torch.cuda` or CUDA-only packages).
- **Local LLM Engine:** llama.cpp Vulkan backend located at:
  - Binary: `C:\Users\Sultan\Downloads\llama-b12255-bin-win-vulkan-x64\llama-cli.exe`
  - Model: `C:\Users\Sultan\Downloads\llama-b12255-bin-win-vulkan-x64\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf`
  - Mandatory Flags: `-ngl 99 -c 2048`
- **Cloud LLM Support:** Free open-source inference via **Groq** (`GROQ_API_KEY`) and OpenAI (`OPENAI_API_KEY`), backed by disk caching in `results/checkpoints/llm_cache.json` to prevent token depletion.

### Repository Files & Modules Created
1. **Governance & Metadata:**
   - [`DECISIONS.md`](file:///h:/CNET%20Project/DECISIONS.md): Architectural decisions (D-01 through D-07).
   - [`CHANGES.md`](file:///h:/CNET%20Project/CHANGES.md): Phase 2 deviations log + Instructor feedback section marked `FEEDBACK_TODO`.
   - [`env_manifest.json`](file:///h:/CNET%20Project/env_manifest.json): Machine & software specification.
   - [`.env.example`](file:///h:/CNET%20Project/.env.example): Template for API keys (`GROQ_API_KEY`, `OPENAI_API_KEY`).
   - [`docs/DATASET.md`](file:///h:/CNET%20Project/docs/DATASET.md): Schemas for telemetry, ground truth, intent pairs, and `runs.csv`.

2. **Configuration (`config/`):**
   - [`config/default.yaml`](file:///h:/CNET%20Project/config/default.yaml): Global seeds, 90s episode parameters, detection thresholds, debate settings.
   - [`config/topologies.yaml`](file:///h:/CNET%20Project/config/topologies.yaml): T1 (tree), T2 (Abilene 11-node), T3 (k=4 Fat-Tree).
   - [`config/faults.yaml`](file:///h:/CNET%20Project/config/faults.yaml): S1 (link failure), S2 (DDoS), S3 (misconfig), S4 (cascading), S5 (DDoS + link).
   - [`config/ppo.yaml`](file:///h:/CNET%20Project/config/ppo.yaml): PPO and GraphSAGE encoder hyperparameters.

3. **Core Modules (`src/`):**
   - [`src/topologies.py`](file:///h:/CNET%20Project/src/topologies.py): `build_networkx()`, `build_mininet()`, and `get_inventory()` across T1, T2, and T3.
   - [`src/faults.py`](file:///h:/CNET%20Project/src/faults.py): Deterministic fault injector for S1–S5 with ground-truth logging.
   - [`src/telemetry_collector.py`](file:///h:/CNET%20Project/src/telemetry_collector.py): 1s polling, rate conversions, robust median/IQR scaling, staleness mask $m_t$, and flow entropy.
   - [`src/log_parser.py`](file:///h:/CNET%20Project/src/log_parser.py): Regex matching for known OVS/controller logs + Drain3 clustering for unstructured logs.
   - [`src/detector.py`](file:///h:/CNET%20Project/src/detector.py): Gradient Boosted Tree / HistGradientBoosting fault detector predicting posterior $p_t \in \{\text{none}, \text{link}, \text{ddos}, \text{misconfig}, \text{cascade}\}$.
   - [`src/rag.py`](file:///h:/CNET%20Project/src/rag.py): Document corpus for OVS, Ryu, and FRR with top-k retrieval and OOD intent detection.

4. **Data Generation (`data/`):**
   - [`data/gen_intent_dataset.py`](file:///h:/CNET%20Project/data/gen_intent_dataset.py): Generator for ~8,000 intent-config pairs across 5 intent classes with machine-checkable predicates and ~10% adversarial negatives.

5. **Test Suite (`tests/`):**
   - [`tests/test_topologies.py`](file:///h:/CNET%20Project/tests/test_topologies.py)
   - [`tests/test_faults.py`](file:///h:/CNET%20Project/tests/test_faults.py)
   - [`tests/test_telemetry.py`](file:///h:/CNET%20Project/tests/test_telemetry.py)
   - [`tests/test_detector.py`](file:///h:/CNET%20Project/tests/test_detector.py)

---

## 2. Where to Place Your API Key (When Resuming)

Create a `.env` file in `h:\CNET Project\` (or copy `.env.example`):
```bash
cp .env.example .env
```
Add your free Groq API key:
```ini
LLM_BACKEND=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
*(If you want to use OpenAI or local Vulkan GGUF instead, set `LLM_BACKEND=openai` or `LLM_BACKEND=gguf` in `.env`.)*

---

## 3. Immediate Resume Step

When you are ready to continue:
1. Simply prompt: `"Resume CL-ANM from checkpoint."`
2. Next tasks to complete:
   - Module completion: `src/slm_agent.py` + `src/debate.py` (with token caching)
   - Module completion: `src/graph_encoder.py` + `src/fluid_env.py` + `src/ppo_agent.py`
   - Module completion: `src/verifier.py` + `src/executor.py` + `src/benchmark_runner.py`
   - Dry run execution (N=2, tree topology) -> stats, plots, and comparison tables.

