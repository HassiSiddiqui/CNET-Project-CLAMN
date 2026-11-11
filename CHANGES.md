# CHANGES.md — Phase 2 Deviations & Engineering Decisions

## CHG-01: Episode Duration
- **What:** Episode shortened from 5 min (Phase 2) to 90s.
- **Why:** Compute feasibility; 90s captures full fault → detect → remediate → verify cycle.
- **Impact:** Recovery times are proportionally shorter; comparisons with Makiyah (cycle-based) normalized accordingly.

## CHG-02: ONOS Controller
- **What:** ONOS classified as Tier D (optional); skipped.
- **Why:** RAM requirements (>4GB JVM) conflict with limited host memory when Mininet is running.
- **Impact:** Only Ryu REST dialect executed live. ONOS deferred to future work.

## CHG-03: LLM Backend
- **What:** OpenAI GPT-4o replaced as primary with Groq free API.
- **Why:** No paid OpenAI key available; Groq provides `llama-3.3-70b-versatile` free tier.
- **Impact:** B1 baseline (GPT-4o) is optional; requires `OPENAI_API_KEY` in `.env`. All results from Groq are marked as such. LLM latency reported separately.

## CHG-04: FRR vtysh
- **What:** FRR live execution in Mininet deferred; syntax/semantic checks only.
- **Why:** GNS3 integration not feasible in WSL2 environment within project timeline.
- **Impact:** FRR dialect generates dataset + does syntax validation but does not execute live. Noted as limitation.

## CHG-05: QLoRA Fine-Tuning
- **What:** QLoRA 4-bit SLM fine-tuning (Qwen2.5-3B/Phi-3-mini) replaced with Groq few-shot eval.
- **Why:** AMD RX 580 (Polaris) has no CUDA and bitsandbytes is CUDA-only. DirectML lacks bitsandbytes support.
- **Impact:** `train/train_slm.py --mode finetune` prints instructions for CUDA environment. Offline eval uses Groq API instead. SLM comparison uses base vs Groq (no fine-tune delta).

## CHG-06: GraphSAGE Implementation
- **What:** Full PyG `SAGEConv` with edge_index used only when `torch-geometric` installed; falls back to linear mean-pool.
- **Why:** PyG install requires matching CUDA/CPU torch build; not guaranteed in all environments.
- **Impact:** `--no-graph` ablation always available. Full GNN only when PyG present.

## CHG-07: Benchmark Stubs
- **What:** `benchmark_runner.py` populates stub random metrics when OVS/Mininet not reachable.
- **Why:** Enables dry-run testing and stats pipeline validation without live Mininet.
- **Impact:** Dry-run (`--dry-run`) results are for pipeline validation only; real benchmark requires WSL root with OVS running.

---

## FEEDBACK_TODO
*To be updated after instructor Phase 2 feedback:*
- [ ] Confirm required topology node counts
- [ ] Confirm required fault intensity ranges
- [ ] Confirm acceptable episode duration
- [ ] Confirm acceptable LLM alternatives (non-GPT-4o)
- [ ] Confirm LoRA fine-tuning requirement vs inference-only
