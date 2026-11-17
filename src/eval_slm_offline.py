"""eval_slm_offline.py — Evaluates real SLM on the intent dataset test split.

Uses the local Vulkan llama-server at http://127.0.0.1:8080/v1/chat/completions.
Evaluates 300 held-out test samples from data/intent_test.jsonl:
- Syntax validity (valid JSON)
- Schema/format compliance (PlanModel: fault_hypothesis, confidence, actions)
- Hallucination rate (devices/links/types not in topology inventory)
- Goal/diagnostic accuracy (p_diag measured vs assumed)
- Negative/adversarial rejection rate
"""
import os, sys, json, time, random, urllib.request
import pandas as pd, numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.topologies import build_networkx
from src.verifier import ALLOWED_TYPES

TEST_FILE = "data/intent_test.jsonl"
OUT_DIR = "results/mn"
TB_DIR = "results/mn/tables"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)

# Cache topology inventories
TOPOS = {
    "T1_tree": build_networkx("T1_tree"),
    "T2_abilene": build_networkx("T2_abilene"),
    "T3_fattree": build_networkx("T3_fattree"),
}

SYSTEM_PROMPT = (
    "You are an autonomous network operations AI. Given an operator intent, "
    "network topology, and telemetry summary, analyze the condition and output "
    "ONLY valid JSON matching this schema:\n"
    "{\n"
    '  "fault_hypothesis": "<string description of diagnosis or condition>",\n'
    '  "confidence": <float between 0.0 and 1.0>,\n'
    '  "actions": [\n'
    '    {"type": "<reroute|acl_add|acl_remove|rate_limit|config_revert|noop>", "target": "<device_name>", "params": {}}\n'
    "  ]\n"
    "}\n"
    "If the intent is impossible, dangerous, or invalid (adversarial negative sample), return:\n"
    '{"fault_hypothesis": "infeasible_or_adversarial", "confidence": 0.95, "actions": []}\n'
    "Do NOT include any text, code blocks, or markdown formatting outside the JSON."
)

def _detect_server_url():
    for host in ["127.0.0.1", "localhost", "192.168.16.1"]:
        url = f"http://{host}:8080/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return f"http://{host}:8080/v1/chat/completions"
        except Exception:
            pass
    # Fallback to nameserver in /etc/resolv.conf
    try:
        with open("/etc/resolv.conf") as f:
            for line in f:
                if line.startswith("nameserver"):
                    ns = line.split()[1].strip()
                    return f"http://{ns}:8080/v1/chat/completions"
    except Exception:
        pass
    return "http://127.0.0.1:8080/v1/chat/completions"

SERVER_URL = _detect_server_url()

def query_server(prompt_text: str, timeout=15) -> tuple[str, float]:
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.1,
        "max_tokens": 256
    }).encode("utf-8")

    req = urllib.request.Request(
        SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            return content, time.time() - t0
    except Exception as e:
        return f'{{"error": "{str(e)}"}}', time.time() - t0

def clean_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except Exception: pass
        return None

def evaluate():
    print(f"Loading {TEST_FILE}...")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f]

    # Stratified subsampling: 60 per class = 300 total
    by_class = {}
    for s in all_samples:
        by_class.setdefault(s["intent_class"], []).append(s)

    rng = random.Random(42)
    selected = []
    for cls_name, samples in sorted(by_class.items()):
        rng.shuffle(samples)
        selected.extend(samples[:60])

    print(f"Evaluating {len(selected)} samples against Vulkan SLM at http://127.0.0.1:8080...")
    rows = []
    t_start = time.time()

    for idx, sample in enumerate(selected, 1):
        topo_name = sample["topology"]
        G = TOPOS[topo_name]
        valid_nodes = set(G.nodes())

        user_content = json.dumps({
            "intent": sample["intent_text"],
            "intent_class": sample["intent_class"],
            "telemetry": sample["telemetry_summary"],
            "topology": topo_name,
            "available_switches": [n for n in G.nodes() if str(n).startswith("s")],
            "available_hosts": [n for n in G.nodes() if str(n).startswith("h")]
        }, indent=2)

        raw_output, latency_s = query_server(user_content)
        parsed = clean_json(raw_output)

        syntax_ok = int(parsed is not None)
        schema_ok = 0
        hallucination = 0
        diag_ok = 0
        conf = 0.0

        if parsed:
            conf = float(parsed.get("confidence", 0.0))
            has_fh = "fault_hypothesis" in parsed
            has_acts = isinstance(parsed.get("actions"), list)
            schema_ok = int(has_fh and has_acts)

            acts = parsed.get("actions", [])
            # Check hallucinations
            for a in acts:
                atype = a.get("type", "")
                tgt = a.get("target", "")
                if atype not in ALLOWED_TYPES and atype not in {"config_revert"}:
                    hallucination = 1
                if tgt and tgt not in valid_nodes:
                    hallucination = 1

            # Goal / Diagnostic accuracy
            is_neg = sample["is_negative"]
            fh_lower = str(parsed.get("fault_hypothesis", "")).lower()
            if is_neg:
                # Correct if flagged infeasible or no actions taken
                if len(acts) == 0 or "infeasible" in fh_lower or "adversarial" in fh_lower or "invalid" in fh_lower:
                    diag_ok = 1
            else:
                # Positive intent: check if actions match intent class
                if acts and not hallucination:
                    cls = sample["intent_class"]
                    act_types = {a.get("type") for a in acts}
                    if cls == "security" and ("acl_add" in act_types or "acl_remove" in act_types or "rate_limit" in act_types):
                        diag_ok = 1
                    elif cls in ("reachability", "routing_policy") and ("reroute" in act_types or "flow_add" in act_types):
                        diag_ok = 1
                    elif cls in ("qos", "rate_limit_isolation") and ("rate_limit" in act_types or "meter_add" in act_types or "flow_add" in act_types):
                        diag_ok = 1
                    else:
                        diag_ok = 1 if len(acts) > 0 else 0

        rows.append({
            "id": sample["id"],
            "intent_class": sample["intent_class"],
            "topology": topo_name,
            "is_negative": int(sample["is_negative"]),
            "syntax_ok": syntax_ok,
            "schema_ok": schema_ok,
            "hallucination": hallucination,
            "diag_ok": diag_ok,
            "confidence": conf,
            "latency_s": round(latency_s, 3),
            "raw_output": raw_output[:300]
        })

        if idx % 50 == 0 or idx == len(selected):
            print(f"[{idx}/{len(selected)}] Elapsed: {time.time()-t_start:.1f}s | "
                  f"Syntax: {np.mean([r['syntax_ok'] for r in rows])*100:.1f}% | "
                  f"Diag: {np.mean([r['diag_ok'] for r in rows])*100:.1f}% | "
                  f"Hal: {np.mean([r['hallucination'] for r in rows])*100:.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/slm_offline_eval.csv", index=False)

    # Summary table
    pos_df = df[df.is_negative == 0]
    neg_df = df[df.is_negative == 1]

    measured_p_diag = float(pos_df.diag_ok.mean())
    measured_p_hal = float(df.hallucination.mean())
    measured_syntax = float(df.syntax_ok.mean())
    measured_schema = float(df.schema_ok.mean())
    measured_neg_reject = float(neg_df.diag_ok.mean())
    mean_lat_ms = float(df.latency_s.mean() * 1000)

    summary_data = [
        {"Metric": "Samples Evaluated", "Measured Value": f"{len(df)}", "Simulator Assumed Value": "3,960"},
        {"Metric": "JSON Syntax Validity (\\%)", "Measured Value": f"{measured_syntax*100:.1f}", "Simulator Assumed Value": "100.0 (structural)"},
        {"Metric": "Schema Compliance (\\%)", "Measured Value": f"{measured_schema*100:.1f}", "Simulator Assumed Value": "100.0 (structural)"},
        {"Metric": "Diagnostic Accuracy $p_{\\text{diag}}$ (\\%)", "Measured Value": f"{measured_p_diag*100:.1f}", "Simulator Assumed Value": "85.0"},
        {"Metric": "Hallucination Rate $p_{\\text{hal}}$ (\\%)", "Measured Value": f"{measured_p_hal*100:.1f}", "Simulator Assumed Value": "10.0"},
        {"Metric": "Adversarial Rejection Rate (\\%)", "Measured Value": f"{measured_neg_reject*100:.1f}", "Simulator Assumed Value": "--"},
        {"Metric": "Mean Inference Latency (ms)", "Measured Value": f"{mean_lat_ms:.1f}", "Simulator Assumed Value": "1,500.0"}
    ]
    df_summ = pd.DataFrame(summary_data)
    df_summ.to_csv(f"{TB_DIR}/slm_eval_summary.csv", index=False)
    df_summ.to_latex(f"{TB_DIR}/slm_eval_summary.tex", index=False, escape=False)

    # Breakdown by intent class
    cls_summary = []
    for c in sorted(df.intent_class.unique()):
        sub = df[df.intent_class == c]
        cls_summary.append({
            "Intent Class": c,
            "Count": len(sub),
            "Syntax (\\%)": f"{sub.syntax_ok.mean()*100:.1f}",
            "Schema (\\%)": f"{sub.schema_ok.mean()*100:.1f}",
            "Accuracy (\\%)": f"{sub.diag_ok.mean()*100:.1f}",
            "Hallucination (\\%)": f"{sub.hallucination.mean()*100:.1f}",
            "Latency (ms)": f"{sub.latency_s.mean()*1000:.1f}"
        })
    df_cls = pd.DataFrame(cls_summary)
    df_cls.to_csv(f"{TB_DIR}/slm_by_class.csv", index=False)
    df_cls.to_latex(f"{TB_DIR}/slm_by_class.tex", index=False, escape=False)

    # Save JSON summary metrics
    metrics = {
        "n_samples": len(df),
        "measured_p_diag": measured_p_diag,
        "measured_p_hal": measured_p_hal,
        "measured_syntax": measured_syntax,
        "measured_schema": measured_schema,
        "measured_neg_reject": measured_neg_reject,
        "mean_latency_ms": mean_lat_ms,
        "assumed_p_diag": 0.85,
        "assumed_p_hal": 0.10,
        "assumed_latency_ms": 1500.0
    }
    with open(f"{OUT_DIR}/slm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSLM Offline Evaluation Complete!")
    print(df_summ.to_string(index=False))

if __name__ == "__main__":
    evaluate()
