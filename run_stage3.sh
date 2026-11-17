#!/bin/bash
PROJ="/mnt/h/CNET Project"
PY="$PROJ/.venv/bin/python3"
cd "$PROJ"
# Clean up any stale Mininet state
mn -c 2>/dev/null
ovs-vsctl del-br mn 2>/dev/null || true
echo "=== Stage 3: Real Mininet Tier M1 (N=5) ==="
"$PY" simeval_mn/run_mn.py --n 5 --topos T1_tree T2_abilene 2>&1
echo "=== Stage 3 complete ==="
