# CL-ANM: Closed-Loop Autonomous Network Management (Phase 3)

## Overview
End-to-end system combining GNN-enhanced RL, an on-prem SLM (Groq API or GGUF), multi-role debate, 3-layer verification, and safe execution for autonomous fault management in Mininet/OVS networks.

**Hardware:** AMD Radeon RX 580 (8GB VRAM, Polaris). No CUDA — Groq API for LLM, DirectML/CPU for PyTorch.

---

## Quick Start

### 1. Environment Setup (WSL2 root)
```bash
# OVS + Mininet (already installed)
service openvswitch-switch start
mn --test pingall   # verify: 0% dropped

# Install Python deps (uses cached packages ~2.5GB)
cd '/mnt/h/CNET Project'
make install
```

### 2. Configure API Keys
```bash
cp .env.example .env
# Edit .env: add GROQ_API_KEY=gsk_...
# Get free key at: https://console.groq.com
```

### 3. Full Pipeline
```bash
make dataset          # Generate 8,000 intent-config pairs (~30s)
make train-detector   # Train fault detector on synthetic data (~2min)
make train-ppo        # Pre-train PPO on fluid env (~10min, checkpoints every 10k steps)
make dry-run          # N=2 benchmark on T1/S1/S2 with B0,B2,FULL (~5min)
make stats            # Statistical tests + LaTeX tables
make literature       # Literature comparison tables
make plots            # All publication figures → results/figures/
```

---

## Architecture

```
Telemetry → Detector (p_t) → SLM Agent → Debate → Verifier → Executor
     ↑                              ↑           ↑
  OVS/OFP                        RAG        NetworkX
                        ↑
                   PPO (GraphSAGE)
```

### Topologies
| ID | Description | Nodes | Links |
|----|-------------|-------|-------|
| T1_tree | Balanced tree, r=2, h=2 | 7 | 6 |
| T2_abilene | US backbone (Topology Zoo) | 11 | 14 |
| T3_fattree | k=4 fat-tree | 20 sw + 16 hosts | 48 |

### Fault Scenarios
| ID | Type | Variants |
|----|------|----------|
| S1 | Link failure | hard / flapping / degraded |
| S2 | DDoS | UDP flood / SYN flood × 3 intensities |
| S3 | Misconfiguration | wrong next-hop / broad ACL / MTU mismatch / missing route |
| S4 | Cascading | primary fail → backup overload |
| S5 | Combined | DDoS + link failure |

### Configurations (--config flag)
| Config | Description |
|--------|-------------|
| B0 | Unmanaged baseline |
| B1 | GPT-4o zero-shot (needs OPENAI_API_KEY) |
| B2 | Static rule script |
| B3 | DQN (no SLM/debate/verifier) |
| FULL | Complete CL-ANM pipeline |
| A1 | --no-verify (ablation) |
| A2 | --no-debate |
| A3 | --no-rl (shortest-path rerouting) |
| A4 | --no-rollback |
| A5 | --no-graph (flat MLP) |
| A6 | --debate-always |

---

## LLM Backends

### Groq (default, recommended)
```bash
# In .env: GROQ_API_KEY=gsk_...
# Model: llama-3.3-70b-versatile (free tier)
python src/slm_agent.py --prompt-file prompt.json --backend groq
```

### Local GGUF (Vulkan)
```bash
# In .env: LLAMA_EXE=C:\Users\Sultan\Downloads\llama-b12255-bin-win-vulkan-x64\llama-cli.exe
# Models: ...\models\Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
python src/slm_agent.py --prompt-file prompt.json --backend gguf \
    --gguf-model "C:\...\models\Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
```

---

## Module Reference

| Module | CLI Example |
|--------|-------------|
| `src/topologies.py` | `python src/topologies.py --topology T1_tree --mode networkx` |
| `src/faults.py` | `python src/faults.py --scenario S1_link_failure --seed 42` |
| `src/detector.py` | `python src/detector.py --train --samples 2000` |
| `src/verifier.py` | `python src/verifier.py plan.json --inventory inv.json` |
| `src/executor.py` | `python src/executor.py plan.json --dry-run` |
| `src/benchmark_runner.py` | `python src/benchmark_runner.py --topology T1_tree --n 2 --dry-run` |
| `evaluation/stats.py` | `python evaluation/stats.py --input results/raw/runs.csv` |
| `plots/make_plots.py` | `python plots/make_plots.py --input results/raw/runs.csv` |

---

## Key Files
```
config/default.yaml      # Global seed, episode params, thresholds
config/topologies.yaml   # T1/T2/T3 definitions
config/faults.yaml       # S1-S5 scenarios
config/ppo.yaml          # PPO & GraphSAGE hyperparameters
results/raw/runs.csv     # Append-only benchmark results
results/checkpoints/     # Detector pkl, PPO zip snapshots, LLM cache
results/figures/         # PDF/PNG plots
results/tables/          # CSV + LaTeX tables
docs/DATASET.md          # Schema documentation
CHANGES.md               # Phase 2 deviations
DECISIONS.md             # Engineering decisions
```

---

## Seed & Reproducibility
- Global seed: **42** (`config/default.yaml → seed: 42`)
- Per-run seed logged in `results/raw/runs.csv`
- Dataset split by `template_id` (70/15/15) — no test leakage

## Key Metrics (in runs.csv)
`latency_p50_ms`, `latency_p95_ms`, `throughput_recovery_s`, `mttr_s`, `packet_loss_pct`, `availability_pct`, `hallucination_rate`, `cpu_mean`, `cpu_peak`, `tokens`, `rollback_count`, `unsafe_blocked`, `debate_invoked`, `diagnosis_accuracy`, `reward_total`, `outcome`

---

## Citation / References
- Makiyah et al. — recovery cycle comparison (T1 tree)
- Mwangi et al. (OMARF) — MTTR/throughput/latency comparison
- Lira et al. — SLM syntax/format/goal accuracy vs GPT-4o

