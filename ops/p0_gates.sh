#!/bin/bash
# Day one on a campaign box, in order: P0 determinism on the campaign device, then
# golden parity vs the MLX reference. Nothing counts before both are journaled.
#   tmux new -d -s p0 'ops/p0_gates.sh cuda'
# or: nohup ops/p0_gates.sh cuda > runs/p0_gates.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEVICE="${1:-cuda}"
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
echo $$ > runs/p0_gates.pid
trap 'rm -f runs/p0_gates.pid' EXIT

echo "[p0] start $(date) pid=$$ device=$DEVICE"

echo "[p0] law-layer suite"
$PY -m pytest -q

echo "[p0] P0 determinism smoke on $DEVICE"
run_gpu $PY scripts/doctor.py --smoke --device "$DEVICE"

echo "[p0] golden parity vs the MLX tier-1 reference on $DEVICE"
run_gpu $PY scripts/parity_check.py --device "$DEVICE"

echo "[p0] GATES COMPLETE $(date) -- grep parity runs/journal.jsonl"
