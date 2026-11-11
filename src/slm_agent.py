"""slm_agent.py — Groq-primary / GGUF-fallback SLM with schema-constrained JSON output."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from typing import Any

CACHE_PATH = Path("results/checkpoints/llm_cache.json")
MAX_REGEN = 3

SYSTEM_PROMPT = (
    "You are a network operations AI. Given a network telemetry summary and "
    "optional RAG context, return ONLY valid JSON matching exactly:\n"
    '{"fault_hypothesis":"<str>","confidence":<0-1>,'
    '"actions":[{"type":"<str>","target":"<str>","params":{}}]}\n'
    "No prose outside the JSON."
)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))

# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _call_groq(prompt: str, model: str, temperature: float) -> str:
    from groq import Groq  # type: ignore
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()

def _call_gguf(prompt: str, model_path: str) -> str:
    """Call llama.cpp Vulkan binary via subprocess."""
    import subprocess, shlex
    llama_exe = os.environ.get(
        "LLAMA_EXE",
        r"C:\Users\Sultan\Downloads\llama-b12255-bin-win-vulkan-x64\llama-cli.exe"
    )
    full_prompt = SYSTEM_PROMPT + "\n\nUser: " + prompt + "\nAssistant:"
    cmd = [llama_exe, "-m", model_path, "-ngl", "99", "-c", "2048",
           "--temp", "0.1", "-n", "512", "--prompt", full_prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout.strip()

def _call_openai(prompt: str, model: str, temperature: float) -> str:
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=512,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Parse JSON from raw response
# ---------------------------------------------------------------------------
def _parse_plan(raw: str) -> dict | None:
    try:
        # Try direct parse
        return json.loads(raw)
    except Exception:
        pass
    # Extract JSON block
    import re
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_prompt(telemetry_summary: dict, rag_context: str, topology_excerpt: dict) -> str:
    return json.dumps({
        "telemetry": telemetry_summary,
        "rag_context": rag_context,
        "topology_excerpt": topology_excerpt,
    }, indent=2)

def query_slm(
    prompt: str,
    verifier_fn=None,
    graph=None,
    inventory: dict | None = None,
    backend: str = "groq",
    groq_model: str = "llama-3.3-70b-versatile",
    gguf_model_path: str = "",
    openai_model: str = "gpt-4o",
    temperature: float = 0.1,
) -> tuple[dict | None, int, list[str]]:
    """Return (plan_dict, attempts_used, error_list)."""
    cache = _load_cache()
    key = _cache_key(prompt)
    errors: list[str] = []

    for attempt in range(1, MAX_REGEN + 1):
        cache_k = f"{key}_{attempt}"
        if cache_k in cache:
            raw = cache[cache_k]
        else:
            t0 = time.time()
            try:
                if backend == "groq":
                    raw = _call_groq(prompt, groq_model, temperature)
                elif backend == "gguf":
                    raw = _call_gguf(prompt, gguf_model_path)
                elif backend == "openai":
                    raw = _call_openai(prompt, openai_model, temperature)
                else:
                    raise ValueError(f"Unknown backend: {backend}")
            except Exception as exc:
                errors.append(f"attempt {attempt} backend error: {exc}")
                break
            latency = time.time() - t0
            cache[cache_k] = raw
            _save_cache(cache)

        plan = _parse_plan(raw)
        if plan is None:
            errors.append(f"attempt {attempt}: JSON parse failed")
            continue

        if verifier_fn and graph is not None and inventory is not None:
            result = verifier_fn(plan, graph, inventory or {})
            if result.passed:
                return plan, attempt, errors
            else:
                errors.append(f"attempt {attempt} verifier: {result.structured_error}")
                # Append error feedback to prompt for next attempt
                prompt = prompt + f"\n\n[VERIFIER ERROR attempt {attempt}]: {result.structured_error}\nCorrect the JSON and retry."
        else:
            return plan, attempt, errors

    return None, MAX_REGEN, errors

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--backend", default="groq", choices=["groq", "gguf", "openai"])
    ap.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    ap.add_argument("--gguf-model", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.prompt_file) as f:
        prompt = f.read()
    plan, attempts, errors = query_slm(
        prompt, backend=args.backend,
        groq_model=args.groq_model, gguf_model_path=args.gguf_model
    )
    result = {"plan": plan, "attempts": attempts, "errors": errors}
    out = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out)
    else:
        print(out)

if __name__ == "__main__":
    main()

