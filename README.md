# Akademeia

My curriculum machine, v3: a pre-registered scale ladder that asks whether
what you learn about training curricula at small scale still holds an order
of magnitude up.

Two machines came before it. Paideia (v1) asked whether curricula buy speed.
Ephebeia (v2) asked what they buy at a fixed budget, and whether tiny-scale
rankings survive a 3.4x scale-up -- they did, exactly. Akademeia asks the
four questions neither could reach at small scale:

- **Q1** Does the curriculum ranking survive an ORDER OF MAGNITUDE?
- **Q2** Where does the chained-multiplication depth wall break?
- **Q3** Does the structural-generalization lottery ever narrow?
- **Q4** What does a scratchpad buy?

**[RESULTS.md](RESULTS.md) has the answers**, with the numbers and the
caveats that belong next to them. The short version: the ranking that had
real statistical separation in it survived, while an ordinal rank statistic
flipped a pair no test ever separated; the depth wall finally moved at the
third rung without clearing its margin; the lottery stayed flat three times;
and writing the intermediate steps down proved learnable while buying
nothing at all for length.

## How it is built to be trusted

[PROTOCOL.md](PROTOCOL.md) was the first commit, before any code, and every
amendment in it is dated and registered BEFORE the run it governs -- including
the ones that went against me. Thresholds are derived from same-seed A/A'
pairs and frozen one way. The benchmark instruments are content-hashed and
pinned, so a drifted pool or tokenizer fails the suite rather than quietly
changing a result. Each arm records the frozen hashes, the device and the
host it ran on, because a comparison across instruments is void unless a
parity gate covered that pair.

The substrate is PyTorch, and it had to earn the right to be compared against
the MLX reference the earlier machines used: per-device determinism, a golden
parity gate against the stored reference band, and a pair gate before any
result crosses two devices. Five instruments across four substrates land
inside the same band.

[evidence/](evidence/) carries the append-only run ledger and every per-arm
result file the verdicts were computed from, so any number in the results can
be traced to the run that produced it.

```
make setup && make test
```

The figures in [figures/](figures/) are generated from that ledger, never
typed by hand.
