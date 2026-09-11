#!/bin/bash
# Q2 composition-wall study at one rung: S2 pool (idempotent build), A/A'
# runs, verdict vs the rung baseline, then the D1 per-template diagnostic.
# Requires the rung's ladder baseline to exist (the measuring stick).
#   tmux new -d -s s2-r3 'ops/study_s2.sh 3 cuda'
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUNG="${1:?usage: study_s2.sh RUNG [DEVICE]}"
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
echo $$ > "runs/study_s2_r${RUNG}.pid"
trap 'rm -f "runs/study_s2_r'"${RUNG}"'.pid"' EXIT

echo "[S2 r$RUNG] start $(date) pid=$$ device=$DEVICE"
run_gpu $PY scripts/doctor.py --device "$DEVICE"
if [ ! -f data/pool-s2.jsonl ]; then
  echo "[S2 r$RUNG] building the S2 pool (hash must match v2)"
  run_gpu $PY scripts/study_s2.py build
fi
run_gpu $PY scripts/study_s2.py runs --rung "$RUNG" --device "$DEVICE"
run_gpu $PY scripts/study_s2.py verdict --rung "$RUNG" --device "$DEVICE"
run_gpu $PY scripts/diag_d1.py --rung "$RUNG" --device "$DEVICE"
echo "[S2 r$RUNG] COMPLETE $(date)"
