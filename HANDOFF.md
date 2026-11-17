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
