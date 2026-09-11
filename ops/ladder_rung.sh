#!/bin/bash
# One Q1 ladder rung, end to end. Launch detached on a campaign box:
#   tmux new -d -s ladder-r3 'ops/ladder_rung.sh 3 cuda'
# or: nohup ops/ladder_rung.sh 3 cuda > runs/ladder_r3.log 2>&1 < /dev/null & disown
# Stages queue politely (python-side wait_for_load + gpulock) and are
# RESUMABLE: arms with matching frozen hashes AND device are reused, so a
# relaunch after a death costs nothing. Aborts LOUDLY on a real gate failure;
# exit 8 from the probe means the budget decision returns to the operator.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUNG="${1:?usage: ladder_rung.sh RUNG [DEVICE]}"
DEVICE="${2:-cuda}"
if [ "$DEVICE" = "cpu" ]; then
  export CUDA_VISIBLE_DEVICES=""   # no context beside the resident server
  export OMP_NUM_THREADS=16        # leave cores for the box's own tenants
fi

# Unbuffered: nohup makes stdout a file, so Python block-buffers and the
# box-guard's "waiting" lines sit invisible for hours -- a silent log reads
# exactly like a dead chain.
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


# Liveness must be checked by PID, never by process-name matching -- a waiter
# whose command line mentions the script self-matches and reports it alive
# forever (that blind spot hid a dead chain for 46h in v2).
mkdir -p runs
echo $$ > "runs/ladder_r${RUNG}.pid"
trap 'rm -f "runs/ladder_r'"${RUNG}"'.pid"' EXIT

echo "[ladder r$RUNG] start $(date) pid=$$ device=$DEVICE"

echo "[ladder r$RUNG] smoke"
run_gpu $PY scripts/doctor.py --smoke --device "$DEVICE"

echo "[ladder r$RUNG] timing probe (1/3 budget; exit 8 = operator decision)"
run_gpu $PY scripts/ladder.py probe --rung "$RUNG" --device "$DEVICE"

echo "[ladder r$RUNG] baseline direct50 A/A' (resumes)"
run_gpu $PY scripts/ladder.py baseline --rung "$RUNG" --device "$DEVICE"

echo "[ladder r$RUNG] calibrate (mechanical, freezes one way)"
# AK_R1_FLOOR_ADVISORY=1 applies the dated 2026-08-22 amendment
# (rung 1 only): mixed-items floor advisory after the discrimination
# diagnostic passed on this instrument. Default strict.
CAL_FLAGS=""
[ "${AK_R1_FLOOR_ADVISORY:-0}" = "1" ] && [ "$RUNG" = "1" ] && CAL_FLAGS="--floor-advisory"
run_gpu $PY scripts/ladder.py calibrate --rung "$RUNG" $CAL_FLAGS

echo "[ladder r$RUNG] arms: core + branch + adaptive2 (resumes)"
run_gpu $PY scripts/ladder.py arms --rung "$RUNG" --device "$DEVICE"

echo "[ladder r$RUNG] verdict"
run_gpu $PY scripts/ladder.py verdict --rung "$RUNG"

echo "[ladder r$RUNG] CHAIN COMPLETE $(date) -- grep rung runs/journal.jsonl"
