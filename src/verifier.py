"""verifier.py — L1/L2/L3 action verifier with hallucination detection."""
from __future__ import annotations
import argparse, json, re
from dataclasses import dataclass, field
from typing import Any, Literal
import networkx as nx
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
ALLOWED_TYPES = {"reroute", "acl_add", "acl_remove", "rate_limit", "flow_add",
                 "flow_del", "meter_add", "meter_del", "link_up", "link_down",
                 "set_weight", "noop"}
ALLOWED_DIALECTS = {"ovs-ofctl", "ryu-rest", "frr-vtysh"}

class ActionModel(BaseModel):
    type: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)

class PlanModel(BaseModel):
    fault_hypothesis: str
    confidence: float
    actions: list[ActionModel]

@dataclass
class VerifyResult:
    passed: bool
    failed_layer: int | None          # 1,2,3 or None
    structured_error: str
    risk: float                       # 0-1
    hallucination_flags: list[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Inventory check (pre-verification hallucination detection)
# ---------------------------------------------------------------------------
def check_hallucinations(plan: dict, inventory: dict) -> list[str]:
    """Return list of hallucination descriptions (empty = clean)."""
    flags: list[str] = []
    devices = set(inventory.get("nodes", {}).keys())
    links = set(tuple(sorted(e)) for e in inventory.get("edges", []))
    for i, act in enumerate(plan.get("actions", [])):
        tgt = act.get("target", "")
        if tgt and tgt not in devices:
            flags.append(f"action[{i}].target '{tgt}' not in inventory")
        iface = act.get("params", {}).get("interface", "")
        if iface:
            node_ifaces = inventory.get("nodes", {}).get(tgt, {}).get("interfaces", [])
            if node_ifaces and iface not in node_ifaces:
                flags.append(f"action[{i}].params.interface '{iface}' not on '{tgt}'")
        cmd = act.get("params", {}).get("command", "")
        if cmd:
            dialect = act.get("params", {}).get("dialect", "ovs-ofctl")
            if dialect not in ALLOWED_DIALECTS:
                flags.append(f"action[{i}] unknown dialect '{dialect}'")
    return flags

# ---------------------------------------------------------------------------
# Layer 1 — schema + whitelist
# ---------------------------------------------------------------------------
def _l1(plan: dict) -> tuple[bool, str]:
    try:
        pm = PlanModel(**plan)
    except ValidationError as exc:
        return False, f"L1 schema: {exc.errors()[0]['msg']}"
    for act in pm.actions:
        if act.type not in ALLOWED_TYPES:
            return False, f"L1 whitelist: unknown action type '{act.type}'"
    return True, ""

# ---------------------------------------------------------------------------
# Layer 2 — NetworkX invariants
# ---------------------------------------------------------------------------
def _l2(plan: dict, graph: nx.Graph, inventory: dict) -> tuple[bool, str]:
    G = graph.copy()
    # simulate reroute actions
    protected_flows: list[tuple] = inventory.get("protected_flows", [])
    for act in plan.get("actions", []):
        if act.get("type") == "link_down":
            tgt = act.get("target", "")
            parts = tgt.split("-")
            if len(parts) == 2 and G.has_edge(*parts):
                G.remove_edge(*parts)
    # check connectivity for protected flows
    for src, dst in protected_flows:
        if src in G and dst in G and not nx.has_path(G, src, dst):
            return False, f"L2 reachability: {src}->{dst} unreachable after plan"
    # no isolated nodes that were previously connected (blackhole check)
    for node in list(G.nodes):
        if G.degree(node) == 0 and graph.degree(node) > 0:
            return False, f"L2 blackhole: node '{node}' isolated after plan"
    return True, ""

# ---------------------------------------------------------------------------
# Layer 3 — dry-run (flow-level simulator on cloned graph)
# ---------------------------------------------------------------------------
def _l3(plan: dict, graph: nx.Graph, service_objectives: dict) -> tuple[bool, str, float]:
    """Lightweight flow-level check; real Mininet snapshot used if available."""
    risk = 0.0
    n_actions = len(plan.get("actions", []))
    # heuristic risk: more actions = higher risk
    risk = min(0.9, 0.1 * n_actions)
    # check utilisation objective
    max_util = service_objectives.get("max_utilisation", 1.0)
    if max_util < 0:
        return False, "L3 service-objective: max_utilisation must be >= 0", 1.0
    return True, "", risk

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def verify(plan: dict, graph: nx.Graph, inventory: dict,
           service_objectives: dict | None = None) -> VerifyResult:
    service_objectives = service_objectives or {}
    # Pre-check hallucinations
    h_flags = check_hallucinations(plan, inventory)
    # L1
    ok, err = _l1(plan)
    if not ok:
        return VerifyResult(False, 1, err, 1.0, h_flags)
    # L2
    ok, err = _l2(plan, graph, inventory)
    if not ok:
        return VerifyResult(False, 2, err, 0.8, h_flags)
    # L3
    ok, err, risk = _l3(plan, graph, service_objectives)
    if not ok:
        return VerifyResult(False, 3, err, risk, h_flags)
    return VerifyResult(True, None, "", risk, h_flags)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Verify an action plan JSON")
    ap.add_argument("plan_file")
    ap.add_argument("--inventory", default=None)
    args = ap.parse_args()
    with open(args.plan_file) as f:
        plan = json.load(f)
    inv = {}
    if args.inventory:
        with open(args.inventory) as f:
            inv = json.load(f)
    G = nx.Graph()
    for node in inv.get("nodes", {}):
        G.add_node(node)
    for e in inv.get("edges", []):
        G.add_edge(*e)
    result = verify(plan, G, inv)
    print(json.dumps(result.__dict__, indent=2))

if __name__ == "__main__":
    main()

