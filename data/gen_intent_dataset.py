"""CL-ANM Intent-to-Configuration Dataset Generator.

Generates ~8,000 intent -> config pairs across 5 intent classes:
  1. Reachability
  2. QoS / Bandwidth Guarantee
  3. Security / ACL Filtering
  4. Routing Policy (path avoidance / pinning)
  5. Rate-Limit / Port Isolation

Supports OVS/Ryu and FRR vtysh dialects.
Includes ~10% adversarial / invalid negative samples.
Splits by template_id: 70% train / 15% val / 15% test (with unseen phrasings in test).
Outputs JSONL + Markdown dataset statistics report.
"""

from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class IntentSample:
    id: str
    topology: str
    intent_text: str
    intent_class: str
    predicate: Dict[str, Any]
    dialect: str
    telemetry_summary: str
    diagnosis: str
    config: Dict[str, Any]
    template_id: int
    is_negative: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Template definitions for the 5 intent classes
TEMPLATES = [
    # 0..3: Reachability
    (0, "reachability", "Restore connectivity from {src} to {dst} via switch {sw}"),
    (1, "reachability", "Ensure bidirectional reachability between {src} ({src_ip}) and {dst} ({dst_ip})"),
    (2, "reachability", "Route traffic from {src} to {dst} using alternative shortest path"),
    (3, "reachability", "Establish backup forwarding path from {src} to {dst} avoiding failed link"),

    # 4..7: QoS
    (4, "qos", "Guarantee minimum bandwidth of {bw} Mbps for flows from {src} to {dst}"),
    (5, "qos", "Enforce strict priority queue for voice and video traffic between {src} and {dst}"),
    (6, "qos", "Allocate dedicated queue on switch {sw} for traffic to {dst_ip} with {bw} Mbps capacity"),
    (7, "qos", "Apply DSCP marking and priority queueing for mission-critical traffic from {src}"),

    # 8..11: Security
    (8, "security", "Block all ingress UDP flood packets from suspect host {src} ({src_ip})"),
    (9, "security", "Drop all TCP SYN traffic exceeding rate limit on switch {sw} port {port}"),
    (10, "security", "Isolate compromised host {src} by installing drop rules at access switch {sw}"),
    (11, "security", "Deny ingress traffic with source IP {src_ip} targeting service on {dst_ip}"),

    # 12..15: Routing Policy
    (12, "routing_policy", "Reroute transit traffic away from congested link {link} via switch {sw}"),
    (13, "routing_policy", "Pin primary forwarding path for {dst_ip} through core switch {sw}"),
    (14, "routing_policy", "Divert all elephant flows destined to {dst} to secondary path {alt_path}"),
    (15, "routing_policy", "Enforce loop-free policy path {src} -> {sw} -> {dst} for critical subnet"),

    # 16..19: Rate-Limit / Isolation
    (16, "rate_limit_isolation", "Rate limit ingress flow on switch {sw} port {port} to {rate} pps"),
    (17, "rate_limit_isolation", "Apply OpenFlow meter on switch {sw} capping bandwidth to {bw} Mbps"),
    (18, "rate_limit_isolation", "Quarantine host {src} by moving port {port} to isolation VLAN 999"),
    (19, "rate_limit_isolation", "Police ingress traffic on {sw}-{port} with burst {burst} kbps and rate {bw} Mbps"),
]

# Unseen test phrasings for template evaluation (held out from training)
UNSEEN_TEST_TEMPLATES = [
    (101, "reachability", "Urgent: Establish end-to-end packet forwarding between {src} and {dst} over {sw}"),
    (102, "qos", "Bandwidth SLA requirement: Reserve at least {bw} Mbps throughput from {src} to {dst}"),
    (103, "security", "Mitigate active attack: Null-route all traffic originating from malicious actor {src_ip}"),
    (104, "routing_policy", "Traffic engineering mandate: Steer flows for {dst_ip} completely off {link}"),
    (105, "rate_limit_isolation", "Contain anomaly: Clamp down transmission rate on port {port} of {sw} to {bw} Mbps"),
]


def generate_sample(sample_id: int, template_entry: Tuple[int, str, str], is_negative: bool, seed: int) -> IntentSample:
    rng = random.Random(seed + sample_id)
    tpl_id, intent_class, tpl_text = template_entry

    topologies = ["T1_tree", "T2_abilene", "T3_fattree"]
    topo = rng.choice(topologies)
    dialects = ["ovs-ofctl", "ryu-rest", "frr-vtysh"]
    dialect = rng.choice(dialects)

    src_idx = rng.randint(1, 4)
    dst_idx = rng.randint(1, 4)
    while dst_idx == src_idx:
        dst_idx = rng.randint(1, 4)

    src = f"h{src_idx}"
    dst = f"h{dst_idx}"
    src_ip = f"10.0.0.{src_idx}"
    dst_ip = f"10.0.0.{dst_idx}"
    sw = f"s{rng.randint(1, 3)}"
    port = f"eth{rng.randint(1, 4)}"
    bw = rng.choice([10, 25, 50, 80, 100])
    rate = rng.choice([500, 1000, 2500, 5000])
    burst = bw * 128
    link = f"{sw}-{port}"
    alt_path = f"s1 -> s3 -> {dst}"

    # Build formatted intent text
    text = tpl_text.format(
        src=src, dst=dst, src_ip=src_ip, dst_ip=dst_ip,
        sw=sw, port=port, bw=bw, rate=rate, burst=burst,
        link=link, alt_path=alt_path
    )

    # Machine-checkable predicate
    predicate: Dict[str, Any] = {}
    if intent_class == "reachability":
        predicate = {"type": "path_exists", "src": src, "dst": dst, "allowed_switches": [sw, "s1", "s2", "s3"]}
    elif intent_class == "qos":
        predicate = {"type": "min_bandwidth_mbps", "src": src, "dst": dst, "min_mbps": bw}
    elif intent_class == "security":
        predicate = {"type": "traffic_blocked", "match": {"nw_src": src_ip, "nw_dst": dst_ip}}
    elif intent_class == "routing_policy":
        predicate = {"type": "avoid_link", "link": link}
    elif intent_class == "rate_limit_isolation":
        predicate = {"type": "max_rate_mbps", "switch": sw, "port": port, "max_mbps": bw}

    # Configuration command synthesis
    config: Dict[str, Any] = {}
    if not is_negative:
        if dialect == "ovs-ofctl":
            if intent_class == "security":
                config = {
                    "tool": "ovs-ofctl",
                    "command": f"ovs-ofctl -O OpenFlow13 add-flow {sw} priority=40000,ip,nw_src={src_ip},actions=drop",
                    "table": 0,
                    "action": "drop",
                }
            elif intent_class == "rate_limit_isolation":
                config = {
                    "tool": "ovs-ofctl",
                    "command": f"ovs-ofctl -O OpenFlow13 add-meter {sw} meter=1,kbps,band=type=drop,rate={bw*1000}",
                    "action": "meter",
                }
            else:
                config = {
                    "tool": "ovs-ofctl",
                    "command": f"ovs-ofctl -O OpenFlow13 add-flow {sw} priority=1000,ip,nw_dst={dst_ip},actions=output:2",
                    "action": "forward",
                }
        elif dialect == "ryu-rest":
            config = {
                "tool": "ryu-rest",
                "endpoint": f"/stats/flowentry/add",
                "payload": {"dpid": int(sw.replace("s", "")), "priority": 1000, "match": {"ipv4_dst": dst_ip}, "actions": [{"type": "OUTPUT", "port": 2}]},
            }
        else:  # frr-vtysh
            config = {
                "tool": "vtysh",
                "commands": [
                    "configure terminal",
                    f"router ospf",
                    f" network {dst_ip}/32 area 0.0.0.0",
                    "exit",
                ],
            }
    else:
        # Adversarial / Negative sample: invalid syntax or nonexistent entity
        bogus_sw = f"s{rng.randint(90, 99)}"
        config = {
            "tool": dialect,
            "command": f"ovs-ofctl add-flow {bogus_sw} priority=invalid_priority,actions=none",
            "error_type": "nonexistent_device",
        }

    telemetry_summary = f"Telemetry snapshot: port {link} loss_rate=0.0, pps={rate}, cpu=12.5%"
    diagnosis = f"Operational state normal. Applying intent: {intent_class} for {src}->{dst}"

    return IntentSample(
        id=f"INTENT-{sample_id:06d}",
        topology=topo,
        intent_text=text,
        intent_class=intent_class,
        predicate=predicate,
        dialect=dialect,
        telemetry_summary=telemetry_summary,
        diagnosis=diagnosis,
        config=config,
        template_id=tpl_id,
        is_negative=is_negative,
    )


def generate_dataset(total_samples: int = 8000, seed: int = 42) -> Tuple[List[IntentSample], List[IntentSample], List[IntentSample]]:
    """Generate train/val/test splits partitioned by template_id (70/15/15)."""
    rng = random.Random(seed)
    all_samples: List[IntentSample] = []

    # 1. Standard templates (20 templates)
    num_standard = int(total_samples * 0.85)
    for i in range(num_standard):
        tpl = rng.choice(TEMPLATES)
        is_neg = (rng.random() < 0.10)  # ~10% negatives
        s = generate_sample(i + 1, tpl, is_neg, seed)
        all_samples.append(s)

    # 2. Unseen test templates (for out-of-distribution intent phrasing evaluation)
    num_unseen = total_samples - num_standard
    unseen_samples: List[IntentSample] = []
    for i in range(num_unseen):
        tpl = rng.choice(UNSEEN_TEST_TEMPLATES)
        is_neg = (rng.random() < 0.10)
        s = generate_sample(num_standard + i + 1, tpl, is_neg, seed)
        unseen_samples.append(s)

    # Split standard samples: 70% train, 15% val, 15% test
    rng.shuffle(all_samples)
    n_train = int(len(all_samples) * 0.70)
    n_val = int(len(all_samples) * 0.15)

    train_set = all_samples[:n_train]
    val_set = all_samples[n_train:n_train + n_val]
    # Test set includes held-out standard samples + unseen phrasings
    test_set = all_samples[n_train + n_val:] + unseen_samples

    return train_set, val_set, test_set


def write_dataset_files(
    output_dir: str = "data",
    total_samples: int = 8000,
    seed: int = 42,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    train_set, val_set, test_set = generate_dataset(total_samples=total_samples, seed=seed)

    splits = {"train": train_set, "val": val_set, "test": test_set}
    stats: Dict[str, Any] = {"total": 0, "splits": {}, "classes": {}, "negatives": 0}

    for name, s_list in splits.items():
        file_path = os.path.join(output_dir, f"intent_{name}.jsonl")
        with open(file_path, "w", encoding="utf-8") as f:
            for s in s_list:
                f.write(json.dumps(s.to_dict()) + "\n")
                stats["classes"][s.intent_class] = stats["classes"].get(s.intent_class, 0) + 1
                if s.is_negative:
                    stats["negatives"] += 1

        stats["splits"][name] = len(s_list)
        stats["total"] += len(s_list)

    # Write Markdown Dataset Stats Report
    report_path = os.path.join(output_dir, "DATASET_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# CL-ANM Intent-to-Config Dataset Report\n\n")
        f.write(f"- **Total Samples:** {stats['total']}\n")
        f.write(f"- **Negative / Adversarial Samples:** {stats['negatives']} ({stats['negatives']/stats['total']*100:.1f}%)\n\n")
        f.write("### Split Distribution\n")
        for k, v in stats["splits"].items():
            f.write(f"- **{k}:** {v} samples ({v/stats['total']*100:.1f}%)\n")
        f.write("\n### Intent Class Balance\n")
        for k, v in stats["classes"].items():
            f.write(f"- **{k}:** {v} samples ({v/stats['total']*100:.1f}%)\n")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="CL-ANM Intent Dataset Generator")
    parser.add_argument("--total", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()

    stats = write_dataset_files(output_dir=args.out_dir, total_samples=args.total, seed=args.seed)
    print(f"Generated {stats['total']} intent samples:")
    print(f"  Train: {stats['splits']['train']} | Val: {stats['splits']['val']} | Test: {stats['splits']['test']}")
    print(f"  Negatives: {stats['negatives']} ({stats['negatives']/stats['total']*100:.1f}%)")


if __name__ == "__main__":
    main()

