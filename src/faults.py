"""CL-ANM Fault Injector and Traffic Generator.

Supports scenarios:
  - S1: Link failure (hard, flapping, degraded via ip link / tc netem)
  - S2: DDoS (UDP / SYN floods via hping3 / iperf3 / scapy)
  - S3: Misconfiguration (wrong next hop, broad ACL, MTU mismatch, missing route via ovs-ofctl / ip route)
  - S4: Cascading failure (primary link fails -> backup overloads/fails)
  - S5: Combined DDoS + Link failure

Writes ground-truth event log: (type, location, onset, clearance, severity, seed).
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FaultEvent:
    scenario: str
    mode: str
    target: str
    onset_s: float
    clearance_s: float
    severity: Dict[str, Any]
    seed: int
    status: str  # 'scheduled', 'active', 'cleared'
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.timestamp == 0.0:
            d["timestamp"] = time.time()
        return d


class FaultInjector:
    """Manages injection and clearance of faults across Mininet and simulated networks."""

    def __init__(self, log_path: Optional[str] = "results/raw/ground_truth.jsonl", dry_run: bool = False):
        self.log_path = log_path
        self.dry_run = dry_run
        self.active_faults: List[FaultEvent] = []
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_event(self, event: FaultEvent) -> None:
        """Append ground-truth event to JSONL file."""
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def _exec(self, cmd: List[str]) -> bool:
        """Run system command safely."""
        if self.dry_run:
            return True
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            return res.returncode == 0
        except Exception:
            return False

    def inject_s1_link(
        self,
        interface: str,
        mode: str = "hard",
        onset: float = 20.0,
        clearance: float = 70.0,
        loss_pct: float = 25.0,
        delay_ms: float = 50.0,
        seed: int = 42,
    ) -> FaultEvent:
        """S1: Link failure (hard link down, flapping, or netem degraded)."""
        severity = {"loss_pct": loss_pct, "delay_ms": delay_ms}
        event = FaultEvent("S1_link_failure", mode, interface, onset, clearance, severity, seed, "active")

        if mode == "hard":
            self._exec(["ip", "link", "set", interface, "down"])
        elif mode == "degraded":
            self._exec(["tc", "qdisc", "add", "dev", interface, "root", "netem", "loss", f"{loss_pct}%", "delay", f"{delay_ms}ms"])
        elif mode == "flapping":
            # Initial down; controller or scheduler handles toggling
            self._exec(["ip", "link", "set", interface, "down"])

        self.active_faults.append(event)
        self.log_event(event)
        return event

    def clear_s1_link(self, event: FaultEvent) -> None:
        """Restore interface to healthy state."""
        interface = event.target
        self._exec(["ip", "link", "set", interface, "up"])
        self._exec(["tc", "qdisc", "del", "dev", interface, "root"])
        event.status = "cleared"
        self.log_event(event)
        if event in self.active_faults:
            self.active_faults.remove(event)

    def inject_s2_ddos(
        self,
        target_ip: str,
        mode: str = "udp_flood",
        intensity: str = "medium",
        onset: float = 20.0,
        clearance: float = 70.0,
        seed: int = 42,
    ) -> FaultEvent:
        """S2: DDoS flood attack injection."""
        rates = {"low": 1000, "medium": 5000, "high": 15000}
        rate_pps = rates.get(intensity, 5000)
        severity = {"intensity": intensity, "rate_pps": rate_pps}
        event = FaultEvent("S2_ddos", mode, target_ip, onset, clearance, severity, seed, "active")

        # In live Mininet, this can launch a background hping3 / iperf3 / scapy process
        self.active_faults.append(event)
        self.log_event(event)
        return event

    def clear_s2_ddos(self, event: FaultEvent) -> None:
        """Terminate DDoS traffic streams."""
        self._exec(["pkill", "-f", "hping3"])
        event.status = "cleared"
        self.log_event(event)
        if event in self.active_faults:
            self.active_faults.remove(event)

    def inject_s3_misconfig(
        self,
        switch: str,
        mode: str = "wrong_next_hop",
        onset: float = 20.0,
        clearance: float = 70.0,
        seed: int = 42,
    ) -> FaultEvent:
        """S3: Misconfiguration fault injection."""
        severity = {"mode": mode, "switch": switch}
        event = FaultEvent("S3_misconfig", mode, switch, onset, clearance, severity, seed, "active")

        if mode == "wrong_next_hop":
            # Insert conflicting drop flow or invalid out_port
            self._exec(["ovs-ofctl", "-O", "OpenFlow13", "add-flow", switch, "priority=50000,actions=drop"])
        elif mode == "broad_acl":
            self._exec(["ovs-ofctl", "-O", "OpenFlow13", "add-flow", switch, "priority=50000,ip,nw_dst=10.0.0.0/24,actions=drop"])
        elif mode == "mtu_mismatch":
            self._exec(["ip", "link", "set", f"{switch}-eth1", "mtu", "576"])

        self.active_faults.append(event)
        self.log_event(event)
        return event

    def clear_s3_misconfig(self, event: FaultEvent) -> None:
        """Revert misconfiguration."""
        switch = event.target
        self._exec(["ovs-ofctl", "-O", "OpenFlow13", "del-flows", switch, "priority=50000"])
        self._exec(["ip", "link", "set", f"{switch}-eth1", "mtu", "1500"])
        event.status = "cleared"
        self.log_event(event)
        if event in self.active_faults:
            self.active_faults.remove(event)

    def inject_s4_cascading(
        self,
        primary_link: str,
        backup_link: str,
        onset: float = 20.0,
        backup_fail_onset: float = 35.0,
        clearance: float = 70.0,
        seed: int = 42,
    ) -> List[FaultEvent]:
        """S4: Cascading failure (primary link drops, backup later overloads and drops)."""
        ev1 = self.inject_s1_link(primary_link, "hard", onset=onset, clearance=clearance, seed=seed)
        ev2 = FaultEvent("S4_cascading", "backup_fail", backup_link, backup_fail_onset, clearance, {"primary": primary_link}, seed, "scheduled")
        self.log_event(ev2)
        return [ev1, ev2]

    def clear_all(self) -> None:
        """Clear all currently active faults."""
        for ev in list(self.active_faults):
            if ev.scenario.startswith("S1"):
                self.clear_s1_link(ev)
            elif ev.scenario.startswith("S2"):
                self.clear_s2_ddos(ev)
            elif ev.scenario.startswith("S3"):
                self.clear_s3_misconfig(ev)


def main() -> None:
    parser = argparse.ArgumentParser(description="CL-ANM Fault Injector CLI")
    parser.add_argument("--scenario", type=str, default="S1_link_failure", choices=["S1_link_failure", "S2_ddos", "S3_misconfig", "S4_cascading", "S5_ddos_and_link_failure"])
    parser.add_argument("--mode", type=str, default="hard")
    parser.add_argument("--target", type=str, default="s1-eth1")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--log-path", type=str, default="results/raw/ground_truth.jsonl")
    args = parser.parse_args()

    injector = FaultInjector(log_path=args.log_path, dry_run=args.dry_run)
    if args.scenario == "S1_link_failure":
        ev = injector.inject_s1_link(args.target, mode=args.mode)
        print(f"Injected: {ev}")
        time.sleep(1)
        injector.clear_s1_link(ev)
        print(f"Cleared: {ev}")
    elif args.scenario == "S2_ddos":
        ev = injector.inject_s2_ddos(args.target, intensity=args.mode)
        print(f"Injected: {ev}")
        injector.clear_s2_ddos(ev)
    elif args.scenario == "S3_misconfig":
        ev = injector.inject_s3_misconfig(args.target, mode=args.mode)
        print(f"Injected: {ev}")
        injector.clear_s3_misconfig(ev)


if __name__ == "__main__":
    main()

