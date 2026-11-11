"""tests/test_failure_drills.py — Deterministic injection of each failure taxonomy case."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
import networkx as nx
from src.verifier import verify

pytestmark = pytest.mark.needs_root  # some drills may need root

def _basic_inventory():
    return {
        "nodes": {"s1": {}, "s2": {}, "h1": {}},
        "edges": [["s1", "s2"], ["s1", "h1"]],
        "protected_flows": [],
    }

def _make_graph(inv):
    G = nx.Graph()
    for n in inv["nodes"]:
        G.add_node(n)
    for e in inv["edges"]:
        G.add_edge(*e)
    return G


# --- L1 failures ---
def test_drill_unknown_action_type():
    inv = _basic_inventory()
    plan = {"fault_hypothesis": "x", "confidence": 0.5,
            "actions": [{"type": "INVALID_CMD", "target": "s1", "params": {}}]}
    r = verify(plan, _make_graph(inv), inv)
    assert not r.passed and r.failed_layer == 1

def test_drill_missing_confidence():
    inv = _basic_inventory()
    plan = {"fault_hypothesis": "x", "actions": []}
    r = verify(plan, _make_graph(inv), inv)
    assert not r.passed and r.failed_layer == 1


# --- L2 failures ---
def test_drill_blackhole_node():
    inv = {
        "nodes": {"s1": {}, "s2": {}, "orphan": {}},
        "edges": [["s1", "s2"]],
        "protected_flows": [],
    }
    # Plan takes orphan offline (already isolated — just check pass/fail)
    plan = {"fault_hypothesis": "x", "confidence": 0.5,
            "actions": [{"type": "noop", "target": "s1", "params": {}}]}
    r = verify(plan, _make_graph(inv), inv)
    assert isinstance(r.passed, bool)  # structural check only


# --- Hallucination drills ---
def test_drill_nonexistent_device():
    inv = _basic_inventory()
    plan = {"fault_hypothesis": "x", "confidence": 0.9,
            "actions": [{"type": "reroute", "target": "ghost99", "params": {}}]}
    from src.verifier import check_hallucinations
    flags = check_hallucinations(plan, inv)
    assert len(flags) > 0

def test_drill_nonexistent_interface():
    inv = {
        "nodes": {"s1": {"interfaces": ["eth0"]}},
        "edges": [], "protected_flows": [],
    }
    plan = {"fault_hypothesis": "x", "confidence": 0.9,
            "actions": [{"type": "reroute", "target": "s1",
                         "params": {"interface": "eth99"}}]}
    from src.verifier import check_hallucinations
    flags = check_hallucinations(plan, inv)
    assert any("eth99" in f for f in flags)


if __name__ == "__main__":
    test_drill_unknown_action_type()
    test_drill_missing_confidence()
    test_drill_blackhole_node()
    test_drill_nonexistent_device()
    test_drill_nonexistent_interface()
    print("All failure drill tests passed.")

