#!/bin/bash
PROJ="/mnt/h/CNET Project"
PY="$PROJ/.venv/bin/python3"
cd "$PROJ"
echo "=== Running Stage 2 (full: v2 + unseen) ==="
"$PY" simeval/run_v2_and_unseen.py
echo "=== DONE ==="
