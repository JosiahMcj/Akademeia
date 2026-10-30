#!/bin/bash
# Run the suite and the ledger audit, and FAIL LOUDLY if either does.
#
# Exists because piping pytest into `tail` hides its exit code, so a
# `pytest | tail -1 && git commit` chain will happily commit over a failing
# test -- which it did once. Pipes are set to fail on any stage here.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== test suite =="
.venv/bin/python -m pytest tests/ -q
echo
echo "== ledger audit =="
.venv/bin/python scripts/audit_ledger.py
