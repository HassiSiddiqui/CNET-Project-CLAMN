"""Smoke tests for FaultInjector across S1-S5 scenarios."""

import os
import json
import pytest
from src.faults import FaultInjector, FaultEvent


def test_fault_injector_dry_run(tmp_path):
    log_file = str(tmp_path / "ground_truth.jsonl")
    injector = FaultInjector(log_path=log_file, dry_run=True)

    # Test S1 link failure
    ev1 = injector.inject_s1_link("s1-eth1", mode="hard", onset=20.0, clearance=70.0, seed=42)
    assert ev1.scenario == "S1_link_failure"
    assert ev1.status == "active"
    assert len(injector.active_faults) == 1

    injector.clear_s1_link(ev1)
    assert ev1.status == "cleared"
    assert len(injector.active_faults) == 0

    # Test S2 DDoS
    ev2 = injector.inject_s2_ddos("10.0.0.1", mode="udp_flood", intensity="high", seed=42)
    assert ev2.scenario == "S2_ddos"
    assert ev2.severity["rate_pps"] == 15000
    injector.clear_s2_ddos(ev2)

    # Test S3 misconfig
    ev3 = injector.inject_s3_misconfig("s2", mode="wrong_next_hop", seed=42)
    assert ev3.scenario == "S3_misconfig"
    injector.clear_s3_misconfig(ev3)

    # Test S4 cascading
    evs = injector.inject_s4_cascading("s1-eth1", "s1-eth2", onset=20.0, backup_fail_onset=35.0, seed=42)
    assert len(evs) == 2
    assert evs[0].scenario == "S1_link_failure"
    assert evs[1].scenario == "S4_cascading"

    # Verify log output was written
    assert os.path.exists(log_file)
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) >= 6

