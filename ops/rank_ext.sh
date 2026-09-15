#!/bin/bash
# Q1 rank seed extension for one rung (2026-09-14 amendment): three more
# seeds for direct50-A and adaptive2, then the descriptive pooled report.
# No verdict is recomputed and no threshold is re-derived.
#   nohup ops/rank_ext.sh 3 mps > runs/rank_ext_r3.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUNG="${1:?usage: rank_ext.sh RUNG [DEVICE]}"
DEVICE="${2:-mps}"
export PYTHONUNBUFFERED=1
if [ "$DEVICE" = "cpu" ]; then
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=16
fi
mkdir -p runs
echo $$ > runs/rank_ext_r${RUNG}.pid
trap 'rm -f runs/rank_ext_r'"${RUNG}"'.pid' EXIT
echo "[rank-ext r$RUNG] start $(date) pid=$$ device=$DEVICE"
$PY scripts/ladder.py extend --rung "$RUNG" --device "$DEVICE"
echo "[rank-ext r$RUNG] pooled report"
$PY scripts/ladder.py rankpooled --rung "$RUNG" --device "$DEVICE"
echo "[rank-ext r$RUNG] COMPLETE $(date)"
