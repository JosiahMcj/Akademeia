#!/bin/bash
# CPU lane: tier-1 science on the host CPU cores while the GPU lane waits for
# memory windows beside the resident model server. CUDA is hidden, so there
# is no context to create and no contention -- measured ~50 min/run of pure
# training (100-step bench, 20 threads), ~70 min with the 20 checkpoint evals.
#
# Gate order is law: P0 smoke on cpu -> parity vs MLX on cpu -> the
# cpu<->cuda pair gate. Only a fully gated instrument runs campaigns; a pair
# FAIL stops this lane before any campaign work (set -e).
#   nohup ops/cpu_lane.sh > runs/cpu_lane.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""    # no context beside the resident server
export OMP_NUM_THREADS=16         # leave cores for the box's own tenants
export AK_R1_FLOOR_ADVISORY=1     # the dated 2026-08-22 amendment:
                                  # rung-1 mixed floor advisory on this
                                  # instrument (diagnostic: 12 discordant)
mkdir -p runs
echo $$ > runs/cpu_lane.pid
trap 'rm -f runs/cpu_lane.pid' EXIT

echo "[cpu-lane] start $(date) pid=$$"

echo "[cpu-lane] P0 determinism smoke on cpu"
$PY scripts/doctor.py --smoke --device cpu

echo "[cpu-lane] golden parity vs the MLX reference on cpu (~4h: 3 runs)"
$PY scripts/parity_check.py --device cpu

# The pair gate exists to license mixing devices on the SAME box; a box with
# no cuda device has nothing to pair (2026-08-20 instrument-change amendment).
if command -v nvidia-smi > /dev/null 2>&1; then
  echo "[cpu-lane] cpu<->cuda device-pair gate"
  $PY scripts/device_pair_check.py --devices cpu cuda
else
  echo "[cpu-lane] pair gate n/a: no cuda device on this host"
fi

echo "[cpu-lane] rung 1 end to end on cpu"
ops/ladder_rung.sh 1 cpu

echo "[cpu-lane] study S2 (Q2) rung 1 on cpu"
ops/study_s2.sh 1 cpu

echo "[cpu-lane] study S1 (Q3) rung 1 on cpu"
ops/study_s1.sh 1 cpu

echo "[cpu-lane] LANE COMPLETE $(date) -- grep verdict runs/journal.jsonl"
