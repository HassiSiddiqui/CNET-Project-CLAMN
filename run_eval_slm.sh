#!/bin/bash
PROJ="/mnt/h/CNET Project"
PY="$PROJ/.venv/bin/python3"
cd "$PROJ"
echo "=== Running Offline SLM Evaluation ==="
"$PY" -u src/eval_slm_offline.py
echo "=== Offline SLM Evaluation Done ==="
