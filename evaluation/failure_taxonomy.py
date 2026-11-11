"""failure_taxonomy.py — Categorize failed episodes from audit logs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict
import pandas as pd

CATEGORIES = {
    "verifier_l1": "Schema/whitelist rejection",
    "verifier_l2": "NetworkX invariant violation",
    "verifier_l3": "Dry-run service-objective failure",
    "executor_apply": "Action application failure",
    "executor_recovery": "Recovery check failed",
    "circuit_open": "Circuit breaker opened",
    "safe_mode": "Safe mode engaged",
    "slm_parse": "SLM JSON parse failure",
    "slm_regen_exhausted": "SLM max regeneration attempts",
    "escalated_human": "Escalated to human operator",
    "unknown": "Uncategorized failure",
}

def classify_event(event: dict) -> str:
    ev = event.get("event", "")
    if ev == "circuit_open":
        return "circuit_open"
    if ev == "safe_mode_engaged":
        return "safe_mode"
    if ev == "action_failed":
        return "executor_apply"
    if ev == "rollback" and not event.get("restored", True):
        return "executor_apply"
    err = str(event.get("error", ""))
    if "L1" in err:
        return "verifier_l1"
    if "L2" in err:
        return "verifier_l2"
    if "L3" in err:
        return "verifier_l3"
    if "recovery check" in err.lower():
        return "executor_recovery"
    if "parse" in err.lower():
        return "slm_parse"
    if "regen" in err.lower():
        return "slm_regen_exhausted"
    if "escalat" in err.lower():
        return "escalated_human"
    return "unknown"

def run_taxonomy(audit_log: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    if Path(audit_log).exists():
        with open(audit_log) as f:
            for line in f:
                try:
                    event = json.loads(line)
                    cat = classify_event(event)
                    counts[cat] += 1
                except Exception:
                    pass
    else:
        counts["unknown"] = 0  # empty log → no failures

    rows = [{"category": k, "description": CATEGORIES.get(k, k), "count": v}
            for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    df = pd.DataFrame(rows)
    df.to_csv(out / "failure_taxonomy.csv", index=False)
    latex = df.to_latex(index=False,
                        caption="Failure taxonomy from audit log.",
                        label="tab:failure_taxonomy")
    (out / "failure_taxonomy.tex").write_text(latex)
    print(df.to_string())
    print(f"\nWritten to {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-log", default="results/logs/audit.jsonl")
    ap.add_argument("--output-dir", default="results/tables/")
    args = ap.parse_args()
    run_taxonomy(args.audit_log, args.output_dir)

if __name__ == "__main__":
    main()

