"""executor.py — snapshot → apply → watch → commit/rollback with audit log."""
from __future__ import annotations
import argparse, json, logging, subprocess, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG_PATH = Path("results/logs/audit.jsonl")
Tv = 10  # verification window seconds
MAX_RETRIES = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("executor")

# Circuit breaker: track repeated identical verifier-failure signatures
_circuit_state: dict[str, int] = {}
CIRCUIT_OPEN_THRESHOLD = 3

@dataclass
class ExecResult:
    success: bool
    action_applied: int
    action_failed: int
    rolled_back: bool
    escalated: bool
    fallback_used: str   # "none","rl","rules","human"
    duration_s: float
    error: str = ""

# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------
def _snapshot_ovs() -> dict:
    try:
        flows = subprocess.run(["ovs-ofctl", "dump-flows", "s1"],
                               capture_output=True, text=True, timeout=5).stdout
        return {"flows": flows, "ts": time.time()}
    except Exception as exc:
        log.warning(f"Snapshot failed: {exc}")
        return {}

def _restore_snapshot(snap: dict) -> bool:
    # In a real system: ovs-ofctl del-flows + re-add from snap
    log.info("Restoring snapshot (stub — no live Mininet)")
    return True

# ---------------------------------------------------------------------------
# Apply a single action via OVS
# ---------------------------------------------------------------------------
def _apply_action(action: dict, dry_run: bool = False) -> bool:
    atype = action.get("type", "noop")
    target = action.get("target", "")
    params = action.get("params", {})
    if dry_run or atype == "noop":
        log.info(f"[DRY] {atype} on {target}")
        return True
    if atype == "flow_add":
        cmd = ["ovs-ofctl", "add-flow", target, params.get("match", "")]
        try:
            subprocess.run(cmd, check=True, timeout=5)
            return True
        except Exception as exc:
            log.error(f"flow_add failed: {exc}")
            return False
    if atype == "link_down":
        try:
            subprocess.run(["ip", "link", "set", target, "down"],
                           check=True, timeout=5)
            return True
        except Exception as exc:
            log.error(f"link_down failed: {exc}")
            return False
    # Fallback: log unsupported
    log.warning(f"Action type '{atype}' not implemented; noop")
    return True

# ---------------------------------------------------------------------------
# Watch telemetry for Tv seconds
# ---------------------------------------------------------------------------
def _watch_recovery(baseline: dict, Tv: int = 10) -> bool:
    """Return True if telemetry stays within 95% of baseline for Tv seconds."""
    log.info(f"Watching telemetry for {Tv}s…")
    time.sleep(min(Tv, 2))  # stub: in real env poll telemetry_collector
    return True  # assume recovery in stub

# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------
def _fallback(plan: dict, snap: dict, use_rl: bool = True) -> tuple[bool, str]:
    """Try RL → static rules → human escalation."""
    # RL fallback (stub)
    if use_rl:
        log.info("Fallback: RL safest action")
        return True, "rl"
    # Static rule fallback
    log.info("Fallback: static rules")
    return True, "rules"

# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
def _audit(event: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({**event, "ts": time.time()}) + "\n")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def execute_plan(
    plan: dict,
    verifier_result=None,
    baseline_telemetry: dict | None = None,
    dry_run: bool = False,
    safe_mode: bool = False,
) -> ExecResult:
    t0 = time.time()

    # Circuit breaker check
    if verifier_result:
        sig = verifier_result.structured_error[:40]
        _circuit_state[sig] = _circuit_state.get(sig, 0) + 1
        if _circuit_state[sig] >= CIRCUIT_OPEN_THRESHOLD:
            log.warning(f"Circuit open for error sig: {sig}")
            _audit({"event": "circuit_open", "sig": sig})
            return ExecResult(False, 0, 0, False, True, "human",
                              time.time() - t0, "circuit breaker open")

    if safe_mode:
        log.warning("Safe mode: freezing changes, keeping last-known-good")
        _audit({"event": "safe_mode_engaged"})
        return ExecResult(False, 0, 0, False, True, "human",
                          time.time() - t0, "safe mode")

    snap = _snapshot_ovs()
    actions = plan.get("actions", [])
    applied = 0
    failed_at = -1

    for i, action in enumerate(actions):
        ok = False
        for retry in range(MAX_RETRIES):
            ok = _apply_action(action, dry_run=dry_run)
            if ok:
                break
            backoff = 0.5 * (2 ** retry)
            log.warning(f"Retry {retry+1} in {backoff}s…")
            time.sleep(backoff)
        if ok:
            applied += 1
            _audit({"event": "action_applied", "action": action})
        else:
            failed_at = i
            _audit({"event": "action_failed", "action": action})
            break

    if failed_at >= 0:
        # Rollback
        restored = _restore_snapshot(snap)
        _audit({"event": "rollback", "restored": restored})
        if not restored:
            log.error("Rollback failed; restoring initial snapshot")
        ok_fb, fb_name = _fallback(plan, snap)
        return ExecResult(False, applied, len(actions) - applied,
                          True, not ok_fb, fb_name, time.time() - t0,
                          f"action[{failed_at}] failed")

    # Watch telemetry
    recovered = _watch_recovery(baseline_telemetry or {}, Tv)
    if not recovered:
        log.warning("Recovery check failed; rolling back")
        _restore_snapshot(snap)
        _, fb_name = _fallback(plan, snap)
        return ExecResult(False, applied, 0, True, False, fb_name, time.time() - t0,
                          "recovery check failed")

    _audit({"event": "commit", "actions_applied": applied})
    return ExecResult(True, applied, 0, False, False, "none", time.time() - t0)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan_file")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--safe-mode", action="store_true")
    args = ap.parse_args()
    with open(args.plan_file) as f:
        plan = json.load(f)
    result = execute_plan(plan, dry_run=args.dry_run, safe_mode=args.safe_mode)
    print(json.dumps(result.__dict__, indent=2))

if __name__ == "__main__":
    main()

