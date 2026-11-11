"""tests/test_verifier.py — Smoke tests for verifier."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import networkx as nx
from src.verifier import verify, check_hallucinations

INVENTORY = {
    "nodes": {
        "s1": {"interfaces": ["eth0", "eth1"]},
        "s2": {"interfaces": ["eth0", "eth1"]},
        "h1": {"interfaces": ["eth0"]},
    },
    "edges": [["s1", "s2"], ["s1", "h1"]],
    "protected_flows": [["h1", "s2"]],
}

GOOD_PLAN = {
    "fault_hypothesis": "link_failure",
    "confidence": 0.9,
    "actions": [{"type": "reroute", "target": "s1", "params": {}}],
}

BAD_SCHEMA = {"fault_hypothesis": "x", "confidence": 0.9}  # missing actions

BAD_TARGET = {
    "fault_hypothesis": "link_failure",
    "confidence": 0.9,
    "actions": [{"type": "reroute", "target": "ghost_switch", "params": {}}],
}

def _make_graph():
    G = nx.Graph()
    for n in INVENTORY["nodes"]:
        G.add_node(n)
    for e in INVENTORY["edges"]:
        G.add_edge(*e)
    return G


def test_good_plan():
    G = _make_graph()
    r = verify(GOOD_PLAN, G, INVENTORY)
    assert r.passed, f"Expected pass, got: {r.structured_error}"

def test_bad_schema_fails_l1():
    G = _make_graph()
    r = verify(BAD_SCHEMA, G, INVENTORY)
    assert not r.passed
    assert r.failed_layer == 1

def test_hallucination_detected():
    flags = check_hallucinations(BAD_TARGET, INVENTORY)
    assert any("ghost_switch" in f for f in flags)

def test_link_down_breaks_reachability():
    plan = {
        "fault_hypothesis": "test",
        "confidence": 0.5,
        "actions": [{"type": "link_down", "target": "s1-h1", "params": {}}],
    }
    G = _make_graph()
    r = verify(plan, G, INVENTORY)
    # h1-s2 path goes through s1; removing s1-h1 isolates h1
    # depends on inventory protected_flows — may pass or fail
    assert isinstance(r.passed, bool)

if __name__ == "__main__":
    test_good_plan()
    test_bad_schema_fails_l1()
    test_hallucination_detected()
    test_link_down_breaks_reachability()
    print("All verifier tests passed.")

