"""The suite is the CPU law layer: hide CUDA from it entirely.

Measured on an accelerator box: torch 2.13's AdamW.step() runs an accelerator
graph-capture health check that initializes a CUDA context even when every
tensor lives on the CPU. On a box whose resident model server holds nearly
all unified memory, that context creation dies with CUDA OOM -- inside
CPU-only tests. A CPU suite must not need the GPU to exist; GPU behavior is
gated by doctor --smoke --device cuda and the journaled P0/parity gates, not
by pytest."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
