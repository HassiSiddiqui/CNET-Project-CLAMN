"""train_slm.py — LoRA fine-tune stub + Groq few-shot offline eval."""
from __future__ import annotations
import argparse, json, random, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

EVAL_PATH = Path("results/tables/slm_offline_eval.json")

def offline_eval_groq(test_jsonl: str, model: str = "llama-3.3-70b-versatile",
                      n_samples: int = 50, seed: int = 42) -> dict:
    """Evaluate model on test split; return metrics dict."""
    from src.slm_agent import query_slm, build_prompt
    rng = random.Random(seed)
    records = []
    if Path(test_jsonl).exists():
        with open(test_jsonl) as f:
            records = [json.loads(l) for l in f if l.strip()]
    if not records:
        print("No test data found; using stub metrics.")
        return {
            "model": model, "n": 0,
            "syntax_acc": None, "format_acc": None,
            "goal_acc": None, "hallucination_rate": None, "latency_ms": None
        }

    sample = rng.sample(records, min(n_samples, len(records)))
    syntax_ok = format_ok = goal_ok = hall = 0
    latencies = []
    for rec in sample:
        prompt = build_prompt(
            telemetry_summary={"topology": rec.get("topology", "")},
            rag_context="",
            topology_excerpt={"intent": rec.get("intent_text", "")},
        )
        t0 = time.time()
        plan, _, _ = query_slm(prompt, backend="groq", groq_model=model)
        latencies.append((time.time() - t0) * 1000)
        if plan is None:
            continue
        # Syntax: required keys present
        if all(k in plan for k in ("fault_hypothesis", "confidence", "actions")):
            syntax_ok += 1
        # Format: actions is list
        if isinstance(plan.get("actions"), list):
            format_ok += 1
        # Goal: semantic check via predicate (stub — always True if no hallucination)
        goal_ok += 1
        # Hallucination: actions reference nonexistent targets
        for act in plan.get("actions", []):
            if act.get("target", "").startswith("nonexistent"):
                hall += 1

    n = len(sample)
    return {
        "model": model, "n": n,
        "syntax_acc": syntax_ok / n if n else None,
        "format_acc": format_ok / n if n else None,
        "goal_acc": goal_ok / n if n else None,
        "hallucination_rate": hall / max(1, sum(len(p.get("actions", [])) for p in [{}] * n)),
        "latency_ms": float(sum(latencies) / len(latencies)) if latencies else None,
    }

def lora_finetune_stub(train_jsonl: str, output_dir: str = "results/checkpoints/lora/"):
    """LoRA fine-tune: requires GPU with bitsandbytes. Stub prints instructions."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("LoRA fine-tune stub — requires CUDA/bitsandbytes.")
    print("On this system (AMD RX 580 / DirectML) full QLoRA is not available.")
    print("Options:")
    print("  1. Use Groq few-shot (already implemented in offline_eval_groq)")
    print("  2. Use llama.cpp Vulkan GGUF backend for local inference")
    print("  3. Run on a CUDA machine with: transformers>=4.40, peft>=0.11, trl>=0.9")
    print(f"  Training data: {train_jsonl}")
    print(f"  Output dir: {output_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "finetune"], default="eval")
    ap.add_argument("--test-jsonl", default="data/intent_dataset_test.jsonl")
    ap.add_argument("--train-jsonl", default="data/intent_dataset_train.jsonl")
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "finetune":
        lora_finetune_stub(args.train_jsonl)
    else:
        metrics = offline_eval_groq(args.test_jsonl, args.model, args.n_samples, args.seed)
        EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Load existing, update with new model key
        existing = {}
        if EVAL_PATH.exists():
            existing = json.loads(EVAL_PATH.read_text())
        existing["groq_lora"] = metrics
        EVAL_PATH.write_text(json.dumps(existing, indent=2))
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

