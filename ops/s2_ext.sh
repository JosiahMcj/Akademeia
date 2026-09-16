#!/bin/bash
# S2 seed extension for one rung (2026-09-15 amendment): three more seeds,
# then the descriptive pooled report. The registered verdict is untouched.
#   nohup ops/s2_ext.sh 3 mps > runs/s2_ext_r3.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUNG="${1:?usage: s2_ext.sh RUNG [DEVICE]}"
DEVICE="${2:-mps}"
export PYTHONUNBUFFERED=1
if [ "$DEVICE" = "cpu" ]; then
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=16
fi
mkdir -p runs
echo $$ > runs/s2_ext_r${RUNG}.pid
trap 'rm -f runs/s2_ext_r'"${RUNG}"'.pid' EXIT
echo "[S2-ext r$RUNG] start $(date) pid=$$ device=$DEVICE"
$PY scripts/study_s2.py extend --rung "$RUNG" --device "$DEVICE"
echo "[S2-ext r$RUNG] pooled report"
$PY scripts/study_s2.py pooled --rung "$RUNG" --device "$DEVICE"
echo "[S2-ext r$RUNG] COMPLETE $(date)"
