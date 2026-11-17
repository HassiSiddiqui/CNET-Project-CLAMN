#!/bin/bash
PROJ="/mnt/h/CNET Project"
PY="$PROJ/.venv/bin/python3"
cd "$PROJ"
echo "=== Running Offline SLM Evaluation (Qwen2.5-Coder-7B) ==="
"$PY" -u src/eval_slm_offline.py --n-per-class 20 --model-name "Qwen2.5-Coder-7B"
echo "=== Offline SLM Evaluation Done ==="
