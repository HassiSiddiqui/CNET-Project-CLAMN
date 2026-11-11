"""debate.py — Multi-role debate: Proposer / Critic / Security / Safety."""
from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, field
from typing import Any

@dataclass
class DebateResult:
    final_plan: dict | None
    risk: float
    rounds_used: int
    invoked: bool
    tokens_used: int
    latency_s: float
    ranked_proposals: list[dict] = field(default_factory=list)

ROLES = ["Proposer", "Critic", "Security", "Safety"]

ROLE_PROMPTS = {
    "Proposer": "Propose a corrective action plan for the fault. Be specific.",
    "Critic":   "Critique the proposed plan. Identify flaws and suggest improvements.",
    "Security": "Review the plan for security risks. Flag any ACL/firewall issues.",
    "Safety":   "Review the plan for service-safety risks. Flag any actions that could cause outages.",
}

def _should_debate(confidence: float, verifier_risk: float,
                   evidence_conflict: bool,
                   tau_c: float = 0.6, tau_r: float = 0.4,
                   debate_always: bool = False, no_debate: bool = False) -> bool:
    if no_debate:
        return False
    if debate_always:
        return True
    return confidence < tau_c or verifier_risk > tau_r or evidence_conflict

def _role_query(role: str, context: str, prev_response: str, slm_fn) -> tuple[str, int]:
    prompt = (f"[{role}]\n{ROLE_PROMPTS[role]}\n\n"
              f"Context:\n{context}\n\nPrevious response:\n{prev_response}")
    result, _, _ = slm_fn(prompt)
    text = json.dumps(result) if isinstance(result, dict) else str(result)
    tokens = len(text.split())
    return text, tokens

def _confidence_weighted_vote(proposals: list[dict]) -> tuple[dict | None, float]:
    if not proposals:
        return None, 1.0
    best = max(proposals, key=lambda p: p.get("confidence", 0.0))
    risk = 1.0 - best.get("confidence", 0.0)
    return best, risk

def run_debate(
    context: str,
    initial_plan: dict,
    slm_fn,
    confidence: float = 0.5,
    verifier_risk: float = 0.5,
    evidence_conflict: bool = False,
    tau_c: float = 0.6,
    tau_r: float = 0.4,
    max_rounds: int = 2,
    debate_always: bool = False,
    no_debate: bool = False,
) -> DebateResult:
    t0 = time.time()
    if not _should_debate(confidence, verifier_risk, evidence_conflict,
                          tau_c, tau_r, debate_always, no_debate):
        return DebateResult(initial_plan, verifier_risk, 0, False, 0, 0.0,
                            [initial_plan] if initial_plan else [])

    proposals = [initial_plan] if initial_plan else []
    tokens = 0
    rounds_done = 0
    prev = json.dumps(initial_plan, indent=2)

    for rnd in range(max_rounds):
        rounds_done += 1
        round_outputs = []
        for role in ROLES:
            resp, tok = _role_query(role, context, prev, slm_fn)
            tokens += tok
            round_outputs.append(resp)
            # Try parse structured plan from Critic/Safety (they may refine)
            try:
                parsed = json.loads(resp)
                if "fault_hypothesis" in parsed and "actions" in parsed:
                    proposals.append(parsed)
            except Exception:
                pass
        prev = "\n".join(round_outputs)

    final, risk = _confidence_weighted_vote(proposals)
    return DebateResult(
        final_plan=final,
        risk=risk,
        rounds_used=rounds_done,
        invoked=True,
        tokens_used=tokens,
        latency_s=time.time() - t0,
        ranked_proposals=proposals,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-file", required=True)
    ap.add_argument("--plan-file", required=True)
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--verifier-risk", type=float, default=0.5)
    ap.add_argument("--no-debate", action="store_true")
    ap.add_argument("--debate-always", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.context_file) as f:
        ctx = f.read()
    with open(args.plan_file) as f:
        plan = json.load(f)
    # Use a stub slm_fn for CLI testing
    def stub_slm(prompt):
        return plan, 1, []
    result = run_debate(ctx, plan, stub_slm,
                        confidence=args.confidence,
                        verifier_risk=args.verifier_risk,
                        no_debate=args.no_debate,
                        debate_always=args.debate_always)
    out = json.dumps(result.__dict__, indent=2)
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out)
    else:
        print(out)

if __name__ == "__main__":
    main()

