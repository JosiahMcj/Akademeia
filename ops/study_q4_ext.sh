#!/bin/bash
# Q4 seed extension for one rung (2026-09-11 amendment): three more seeds,
# then the descriptive pooled report. Thresholds are untouched by design.
#   nohup ops/study_q4_ext.sh 1 cpu > runs/q4_ext_r1.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUNG="${1:?usage: study_q4_ext.sh RUNG [DEVICE]}"
DEVICE="${2:-cpu}"
export PYTHONUNBUFFERED=1
if [ "$DEVICE" = "cpu" ]; then
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=16
fi
mkdir -p runs
echo $$ > runs/q4_ext_r${RUNG}.pid
trap 'rm -f runs/q4_ext_r'"${RUNG}"'.pid' EXIT
echo "[Q4-ext r$RUNG] start $(date) pid=$$ device=$DEVICE"
$PY scripts/study_q4.py extend --rung "$RUNG" --device "$DEVICE"
echo "[Q4-ext r$RUNG] pooled report"
$PY scripts/study_q4.py pooled --rung "$RUNG" --device "$DEVICE"
echo "[Q4-ext r$RUNG] COMPLETE $(date)"
