#!/bin/bash
# Q3 lottery study at one rung (12 runs: 6 seeds x A/A'), then the cross-rung
# verdict over everything measured so far.
#   tmux new -d -s s1-r3 'ops/study_s1.sh 3 cuda'
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUNG="${1:?usage: study_s1.sh RUNG [DEVICE]}"
DEVICE="${2:-cuda}"
if [ "$DEVICE" = "cpu" ]; then
  export CUDA_VISIBLE_DEVICES=""   # no context beside the resident server
  export OMP_NUM_THREADS=16        # leave cores for the box's own tenants
fi
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # determinism on CUDA
export CUDA_MODULE_LOADING=LAZY          # eager loads GBs of sm_121 cubins
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Transient-contention retry: the resident model server's KV cache spikes
# while it serves, and a CUDA alloc that lands in a spike dies with OOM even
# though the same run fits minutes later (measured: attempt 1 trained fine at
# ~7GB free; attempt 2 OOMed at the same nominal headroom). Delay, never die
# -- but ONLY for unregistered crashes: registered exits (3 parity fail,
# 6 determinism fail, 7 discrimination fail, 8 probe abort) surface at once.
run_gpu() {
  local tries=0 rc
  while true; do
    set +e; "$@"; rc=$?; set -e
    [ $rc -eq 0 ] && return 0
    case $rc in 3|6|7|8) echo "[chain] registered exit $rc from: $*"; exit $rc;; esac
    tries=$((tries+1))
    [ $tries -ge 12 ] && { echo "[chain] giving up after $tries attempts (rc=$rc)"; exit $rc; }
    echo "[chain] rc=$rc, likely transient box contention; retry $tries/12 in 300s"
    sleep 300
  done
}

mkdir -p runs
echo $$ > "runs/study_s1_r${RUNG}.pid"
trap 'rm -f "runs/study_s1_r'"${RUNG}"'.pid"' EXIT

echo "[S1 r$RUNG] start $(date) pid=$$ device=$DEVICE"
run_gpu $PY scripts/doctor.py --device "$DEVICE"
run_gpu $PY scripts/study_s1.py run --rung "$RUNG" --device "$DEVICE"
run_gpu $PY scripts/study_s1.py verdict
echo "[S1 r$RUNG] COMPLETE $(date)"
