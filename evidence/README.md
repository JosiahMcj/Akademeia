# Evidence

The run ledger for the v3 campaign: the append-only `journal.jsonl` (every
run, gate, verdict, amendment and annotation) and the per-arm result files
every verdict was computed from. Each arm file carries the frozen hashes,
the device and the host it was measured on, so any number in RESULTS.md can
be traced to the run that produced it.

`runs/` itself stays untracked -- it also holds checkpoints and per-seed
resume caches, which are large and are not evidence. This directory is the
part that cannot be regenerated without repeating the compute, so it is
tracked.

Machine identifiers in these records are mapped to stable anonymous labels
(`box-B` and so on), matching the instrument labels used in the results; the
device strings are the literal torch devices the runs used. The first
instrument's own ledger -- including its cuda parity attempts and the
cpu<->cuda pair gate cited in the results -- is held separately and is not
part of this repository.
