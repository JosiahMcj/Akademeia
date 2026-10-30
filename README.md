# A Core-then-Branch Curriculum's Retention Cost Survives 47x Scale While the Ordinal Ranking Never Settles: A Pre-registered Ladder from 3.2M to 151.4M Parameters

*Akademeia, the third machine in my curriculum series.*

Does the order in which a small language model sees its training data change
what it can do, and does that difference survive an order of magnitude of
scale? I trained models at four sizes, from 3.2M to 151.4M parameters on 30M
to 450M tokens, and asked four pre-registered questions.

**Read the paper.** [paper/PAPER.pdf](paper/PAPER.pdf) has the figures inline,
and [paper/PAPER.md](paper/PAPER.md) is the same text in Markdown.

## What I found

- **The ranking that has real separation in it survives 47x of scale.** A
  core-then-branch curriculum keeps 0.02 of the basics at rung 4 against the
  baseline's 0.85, and its retention deficit does not soften as the model
  grows. The ordinal rank statistic flipped twice, both times on the one pair
  that no gate ever separated.
- **The composition wall moves last and barely.** Transfer on unseen
  three-variable products sat at the floor for two rungs, then reached
  0.198 +- 0.035 at rung 3, far under the margin it would need to clear.
- **The per-run lottery does not narrow.** Spread was FLAT at both steps, at
  twelve pairs per rung, under an exact Brown-Forsythe permutation test.
- **Length extrapolation is a hard zero, and the model never attempts the
  extra step.** One factor longer than training scored 0.000 in 36 of 36
  runs. Decoding the outputs showed the model wrote the third partial product
  in 0 of 120 items. Training on longer chains helps the 3.2M model and not
  the 39.5M one.

![The scale ladder](figures/v3-ladder.png)

![The length wall](figures/v3-d2-length.png)

## How the study is built

[PROTOCOL.md](PROTOCOL.md) was the first commit, before any code. Every
threshold was frozen from same-seed calibration pairs before the arm it
judges ran, and every post-data change is a dated amendment.
[RESULTS.md](RESULTS.md) holds the verdicts, and `evidence/` holds the
append-only journal and the per-arm result files every number is read from.

This is the third machine in a series. Paideia (v1) asked whether curricula
buy speed. Ephebeia (v2, coming soon) asked what they buy at a fixed budget.

## Reproducing

```
make setup && make test
```

Running the campaign, one stage at a time:

```
ops/p0_gates.sh <device>            # determinism, then golden parity
ops/ladder_rung.sh <rung> <device>  # probe, baseline, calibrate, arms, verdict
ops/study_s2.sh <rung> <device>     # Q2
ops/study_s1.sh <rung> <device>     # Q3
scripts/verdict_report.py --arm <tag> --base <tag> --rung <n>
```

Arms resume from hash-, device- and host-stamped caches, so an interrupted run
costs at most one seed. `runs/journal.jsonl` is the append-only record of every
run, gate, verdict and amendment. The published copy is `evidence/journal.jsonl`,
with the per-arm result files in `evidence/arms/`.

Figures and the paper:

```
python scripts/make_figures.py          # Q1-Q4 figures, from a live runs/ tree
python scripts/make_length_figures.py   # D2, D3 and D3-ops figures, from evidence/
paper/build_pdf.sh                      # the paper as PDF (needs pandoc and Chrome)
```
