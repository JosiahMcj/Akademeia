#!/bin/bash
# Study Q4 (scratchpad) for one rung. CPU lane by design: this box's GPU has
# other tenants and this study has no need of it.
#   nohup ops/study_q4.sh 1 cpu > runs/q4_r1.log 2>&1 < /dev/null & disown
# Preflight exits 7 if the task cannot discriminate at this rung; that is a
# recorded finding, not a failure to retry.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUNG="${1:?usage: study_q4.sh RUNG [DEVICE]}"
DEVICE="${2:-cpu}"
export PYTHONUNBUFFERED=1
if [ "$DEVICE" = "cpu" ]; then
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=16      # the gated cpu instrument's thread count
fi
mkdir -p runs
echo $$ > runs/q4_r${RUNG}.pid
trap 'rm -f runs/q4_r'"${RUNG}"'.pid' EXIT
echo "[Q4 r$RUNG] start $(date) pid=$$ device=$DEVICE"
echo "[Q4 r$RUNG] preflight (exit 7 = cannot discriminate, recorded)"
$PY scripts/study_q4.py preflight --rung "$RUNG" --device "$DEVICE"
echo "[Q4 r$RUNG] arms A/A'"
$PY scripts/study_q4.py arms --rung "$RUNG" --device "$DEVICE"
echo "[Q4 r$RUNG] verdict"
$PY scripts/study_q4.py verdict --rung "$RUNG" --device "$DEVICE"
echo "[Q4 r$RUNG] COMPLETE $(date)"
