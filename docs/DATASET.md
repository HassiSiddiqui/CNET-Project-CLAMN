# CL-ANM Telemetry & Episode Dataset Schema (DATASET.md)

This document formalizes the data schemas for telemetry time series, ground-truth logs, intent pairs, and closed-loop execution outcomes.

## 1. Telemetry Time Series Schema (`results/raw/telemetry_episodes.jsonl`)

Each line represents a 1-second synchronous polling interval across all network devices:

| Field | Type | Description |
|---|---|---|
| `timestamp` | `float` | Unix epoch timestamp (seconds) |
| `step` | `int` | Discrete episode time-step ($t \in [0, 90]$) |
| `switch_id` | `string` | Target OpenFlow datapath identifier (`s1`, `s2`, ...) |
| `port_no` | `int` | Physical port number on switch (1, 2, ...) |
| `rx_packets` | `int` | Cumulative received packet counter |
| `tx_packets` | `int` | Cumulative transmitted packet counter |
| `rx_bytes` | `int` | Cumulative received bytes counter |
| `tx_bytes` | `int` | Cumulative transmitted bytes counter |
| `rx_drops` | `int` | Cumulative dropped packet counter on ingress |
| `tx_drops` | `int` | Cumulative dropped packet counter on egress |
| `rx_pps` | `float` | Packets per second rate (reset-handled) |
| `tx_pps` | `float` | Packets per second rate (reset-handled) |
| `rx_bps` | `float` | Throughput rate in bits per second |
| `tx_bps` | `float` | Throughput rate in bits per second |
| `loss_rate` | `float` | Ratio of dropped packets to total packets ($\in [0, 1]$) |
| `utilization` | `float` | Port link utilization ratio ($\in [0, 1]$) |
| `latency_ms` | `float` | Active round-trip probe latency in milliseconds |
| `syn_ack_ratio`| `float` | Ratio of TCP SYN flags to ACK flags |
| `src_entropy` | `float` | Shannon entropy of source IP addresses |
| `unique_dsts` | `int` | Count of distinct destination IPs observed |
| `cpu_percent` | `float` | Controller host CPU utilization percentage |
| `mem_percent` | `float` | Controller host memory utilization percentage |
| `staleness_mask`| `int` | `0` = fresh telemetry reading; `1` = missing / interpolated |

---

## 2. Ground-Truth Injected Faults Schema (`results/raw/ground_truth.jsonl`)

| Field | Type | Description |
|---|---|---|
| `timestamp` | `float` | Unix epoch timestamp of injection / clearance |
| `scenario` | `string` | Fault scenario ID (`S1_link_failure`, `S2_ddos`, ...) |
| `mode` | `string` | Scenario sub-mode (`hard`, `flapping`, `syn_flood`, ...) |
| `target` | `string` | Injected interface or device (`s1-eth1`, `10.0.0.1`, `s2`) |
| `onset_s` | `float` | Planned fault onset time in episode ($t_0=20$s) |
| `clearance_s`| `float` | Fault restoration time ($t=70$s) |
| `severity` | `dict` | Scenario-specific parameters (e.g. rate, loss %) |
| `seed` | `int` | Per-run deterministic pseudo-random seed |
| `status` | `string` | Event state (`active`, `cleared`, `scheduled`) |

---

## 3. Intent Dataset Schema (`data/intent_{split}.jsonl`)

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Canonical unique identifier (`INTENT-000001`) |
| `topology` | `string` | Topology context (`T1_tree`, `T2_abilene`, `T3_fattree`) |
| `intent_text` | `string` | Natural language operator intent |
| `intent_class`| `string` | Class: `reachability`, `qos`, `security`, `routing_policy`, `rate_limit_isolation` |
| `predicate` | `dict` | Machine-checkable invariant predicate for formal verification |
| `dialect` | `string` | Target network syntax (`ovs-ofctl`, `ryu-rest`, `frr-vtysh`) |
| `telemetry_summary` | `string` | Contextual telemetry snippet provided to model |
| `diagnosis` | `string` | Synthesized diagnosis / anomaly root-cause description |
| `config` | `dict` | Ground-truth executable configuration dictionary |
| `template_id` | `int` | Source template generator identifier |
| `is_negative` | `bool` | `true` if negative/adversarial intent; `false` otherwise |

---

## 4. Benchmark Runs Schema (`results/raw/runs.csv`)

Appended per evaluated episode:

```csv
run_id,topology,scenario,config,seed,status,mttr_s,recovery_time_s,p50_latency_ms,p95_latency_ms,packet_loss_pct,availability_pct,hallucination_rate,cpu_mean_pct,cpu_peak_pct,gpu_util_pct,gpu_mem_mb,tokens_used,decision_latency_ms,diagnosis_acc,rollback_count,rollback_success,unsafe_blocked,debate_invocation_rate,outcome_code
```

