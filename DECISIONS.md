# CL-ANM Architectural and Technical Decisions (DECISIONS.md)

This log records engineering decisions made during Phase 3 of Closed-Loop Autonomous Network Management (CL-ANM) according to the Phase 2 specification and runtime environment constraints.

## D-01: Runtime Environment & Execution Architecture
- **Decision:** Execute Mininet, OpenFlow switches (OVS 3.7+), network namespaces, and network traffic generation natively within WSL2 Ubuntu with root privileges; keep project files on `/mnt/h/CNET Project` shared with Windows host.
- **Rationale:** Mininet requires Linux kernel network namespaces, Open vSwitch kernel/userspace datapath, and root privileges. WSL2 provides native Linux kernel capabilities while integrating with Windows developer tooling.
- **Alternatives Considered:** Bare Windows (unsupported by Mininet), VirtualBox VM (excessive virtualization overhead, slower).

## D-02: GPU & ML Inference Backend Selection
- **Decision:** For PyTorch, execute on CPU / DirectML (`torch_directml`). For LLM inference, support:
  1. Local on-prem GGUF models via llama.cpp Vulkan backend (`C:\Users\Sultan\Downloads\llama-b12255-bin-win-vulkan-x64\llama-cli.exe` or `llama-server.exe`) utilizing AMD Radeon RX 580 2048SP GPU (`-ngl 99 -c 2048`).
  2. Cloud API endpoints via Groq (`GROQ_API_KEY`, e.g. `llama-3.3-70b-versatile` or `qwen-2.5-32b`) and OpenAI (`OPENAI_API_KEY`).
  3. Deterministic local rule/heuristic offline fallback when offline or no API tokens available.
- **Rationale:** Complies strictly with the user's hardware constraint (AMD Radeon RX 580, no CUDA) and user requirement to prioritize free open-source models (Groq / local GGUF) while preventing OOM.

## D-03: Disk-Backed LLM Generation Cache & Checkpointing
- **Decision:** All LLM prompts and completions are cryptographically hashed (SHA-256) and cached in `results/checkpoints/llm_cache.json`.
- **Rationale:** Protects against token quota exhaustion, network glitches, or rate limits. If a run is interrupted or tokens run out, switching accounts or resuming uses the exact cached outputs without token consumption or variance.

## D-04: Episode Length & Time Scaling
- **Decision:** Episode duration is set to 90s (warm-up 20s, fault onset $t_0=20$s, observation $T_v=10$s, fault duration 50s, post-clearance 20s) as specified in Phase 3 Scope.
- **Rationale:** Shortened from Phase 2's 5 minutes to fit realistic compute budgets while providing ample cycles (1s telemetry polling $\times$ 90 intervals) for detection, debate, verification, and recovery validation.

## D-05: Dialect & Controller Architecture
- **Decision:** Primary controller execution utilizes OVS OpenFlow 1.3 flow-table modification via `ovs-ofctl -O OpenFlow13` and REST API abstractions mimicking Ryu REST controller. FRR vtysh is supported as a verified dialect with grammar and semantic validation for intent datasets (syntax/semantic checks).
- **Rationale:** Ensures direct, rock-solid execution in Mininet without fragile external controller RPC timeouts while maintaining the exact multi-dialect abstraction required by Phase 2.

## D-06: Verifier Architecture (Syn $\land$ Inv $\land$ Sim $\land$ $a \in \mathcal{A}_{perm}$)
- **Decision:** Implement a three-layer verifier:
  - **L1 (Syntax & Whitelist):** Pydantic schema validation + dialect grammar parser + command whitelist.
  - **L2 (Invariants):** NetworkX topological invariants: no forwarding loops, no blackholes, reachability predicates, ACL conflict checks, protected flows protection, and device/interface existence checks.
  - **L3 (Simulation / Shadow Twin):** Flow-level dry-run simulation against service objectives (latency bound, throughput capacity).
  - **Hallucination Detection:** Action references to nonexistent devices/interfaces or invalid dialect syntax are detected and logged as hallucinations before verification.

## D-07: PPO Action Space & Fluid Environment Pretraining
- **Decision:** Action space is path-weight and split-ratio selection over precomputed $k=3$ loop-free shortest paths between ingress-egress pairs. Pretraining executes on a flow-level NetworkX simulator (`fluid_env.py`) for sample efficiency, followed by fine-tuning on Mininet.
- **Rationale:** Discrete path selection prevents invalid flow loops by construction while enabling GraphSAGE / GCN topological embeddings to generalize across topologies.

