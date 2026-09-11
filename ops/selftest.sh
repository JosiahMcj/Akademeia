#!/bin/bash
# Nightly substrate self-test for a campaign box (GitHub runners have no
# accelerator, so
# CI covers the pure-Python laws and THIS covers the CUDA substrate). Wire it
# to a timer once the box exists; it must stay quiet on success and loud on
# failure.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEVICE="${1:-cuda}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # determinism on CUDA
export CUDA_MODULE_LOADING=LAZY          # eager loads GBs of sm_121 cubins
{
  echo "[selftest] $(date) device=$DEVICE"
  $PY -m pytest -q
  $PY scripts/doctor.py --smoke --device "$DEVICE"
  echo "[selftest] OK $(date)"
} >> runs/selftest.log 2>&1
