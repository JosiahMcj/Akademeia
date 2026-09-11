# Akademeia results

v3 of my curriculum machine. The question line is the same one v1 and v2
asked -- does the order you feed a small model its training data change
what it can do -- and v3 asks whether the answers survive an order of
magnitude of scale. Every threshold below was frozen from same-seed A/A'
pairs before the arm it judges ever ran; every number is read out of the
run ledgers in `runs/`, never retyped.

Three rungs completed: 4L/256d (3.2M params, 30M tokens), 6L/384d (10.7M,
45M tokens), 8L/640d (39.5M, 135M tokens). That is a 12x span in
parameters and 4.5x in tokens.

## The instrument agrees with itself across four substrates

The torch port had to reproduce the MLX reference it inherited before any
result counted. Golden parity, tier-1, band 0.1 around a reference mean of
0.5889:

| instrument | parity mean | diff |
|---|---|---|
| the MLX reference | 0.5889 | -- |
| torch, accelerator A | 0.5778 | -0.011 |
| torch, processor A | 0.5833 | -0.006 |
| torch, processor B | 0.5778 | -0.011 |
| torch, accelerator B | 0.5556 | -0.033 |

Determinism is gated per device (same seed trained twice, curves compared
at 1e-6) and device pairs are gated before any comparison crosses them: the
two pairs that a result ever crossed differed by 0.006 and 0.022. Parity failed on the
first attempt at -0.18; the cause was three port-spec mismatches found by
measuring the MLX substrate directly (embedding init scale, attention
projection shape and init, and the optimizer's uncorrected update with
pre-step decoupled decay), not by tuning toward the gate.

## Q1: the separated ranking survives; the ordinal measure does not

Two arms are judged against the mixed 50/50 baseline at each rung, with
thresholds re-derived per rung.

| rung | branch vs direct50 | adaptive2 vs direct50 |
|---|---|---|
| 1 | 8W/18L, not promoted, retention -0.55 | 7W/3L, non-inferior |
| 2 | 5W/19L, not promoted, retention -0.59 | 6W/3L, non-inferior |
| 3 | 4W/21L, not promoted, retention -0.81, P(worse)=0.0005 | 9W/13L, non-inferior |

**Every gate-level verdict replicated at all three rungs.** Core-then-branch
loses at every scale, and its cost is not subtle: the branch arm's basics
retention sits at 0.03-0.04 at every rung, against 0.58-0.85 for the
baseline. Training a niche on top of a frozen basics core destroyed the
basics, and three scale steps did not soften that.

The ordinal measure tells a different story, and both belong in the record.
Pairwise rank agreement was 1.0 from rung 1 to rung 2 and **0.667 from rung
2 to rung 3**, with the top-1 arm changing from adaptive2 to direct50. By
the registered rule that is the rung where stability breaks, and the
pre-registered prediction that rankings scramble beyond tier 2 is nominally
borne out.

But the single discordant pair is (adaptive2, direct50), separated at rung
3 by 0.022 -- against that rung's own noise band of 0.15, a sign test of
P=0.26, and a measured difference of 0.022 between the two instruments
involved. It is the
one pair that no gate separated at any rung. The two concordant pairs both
involve branch, which sits last at every rung with power. So the honest
reading is that a ranking with real separation in it survived an order of
magnitude, while an ordinal statistic with no noise floor flipped a pair
that was never distinguishable in the first place. A rank-agreement number
is only as meaningful as the gaps underneath it.

## Q2: the composition wall is the last thing scale moves

Unseen three-variable product templates, trained against a product-heavy
variety pool, scored against the frozen bench:

| rung | trained 3-var (in-distribution) | unseen 3-var (transfer) | margin | verdict |
|---|---|---|---|---|
| 1 | 0.29 | 0.048 | 0.150 | depth-limited |
| 2 | 0.69 | 0.063 | 0.429 | depth-limited |
| 3 | 0.96 | 0.270 | 0.714 | depth-limited |

In-distribution execution of chained multiplication is essentially solved
by rung 3, and unrelated unseen templates sit at 0.86-0.96 throughout. The
transfer number is the interesting one: flat at the floor for two rungs,
then 4x at rung 3 -- off the floor for the first time, and still nowhere
near the margin it would have to clear. "The wall persists" survives as the
verdict; "the wall is absolute" would have been wrong, and one more rung is
the experiment that would settle it.

## Q3: the lottery does not narrow

Structural generalization on the vm niche, six same-seed pair means per
rung, exact Brown-Forsythe permutation on the spread:

| step | spread | statistic | exact p | verdict |
|---|---|---|---|---|
| rung 1 -> 2 | 0.0665 -> 0.0636 | 0.0125 | 0.712 | flat |
| rung 2 -> 3 | 0.0636 -> 0.0164 | 0.0306 | 0.390 | flat |

Means rise steadily (0.168 -> 0.306 -> 0.414) while the registered variance
question stays unanswered in the same direction v2 left it. The rung-3
spread is nominally 4x smaller, which is suggestive and does not clear
alpha at n=6; by its own registered rule this study is underpowered against
a narrowing of that size, and saying so is the result rather than a
footnote to it.

## Q4: writing the intermediates down is learnable, and buys nothing for length

The scratchpad format gives the model a task in which it must emit each
partial product before the answer (`E=a*b*c;a=3;b=5;c=2` -> `15;30`), graded
FULL-TRACE: a wrong intermediate with a lucky final answer is wrong. This is
a different task from the plain format, and it is registered as such -- what
follows is a statement about the scratchpad task on its own terms, never a
comparison against the plain one.

| rung | novel combinations (6 runs) | one step longer than training | derived noise band |
|---|---|---|---|
| 1 | 0.325 | 0.000 | 0.40 |
| 2 | 0.411 | 0.000 | 0.60 |
| 3 | 0.556 | 0.000 | 0.40 |

**Length extrapolation is a hard zero: 0.000 in 18 runs out of 18, across all
three rungs.** One more factor in the chain than training ever showed, using
only symbols the model knows, in a format it has already demonstrated on
novel combinations -- and every run, at every scale, scores nothing. Twelve
times the parameters moved it not at all. Writing the intermediates down did
not buy the step that the depth wall in Study S2 is made of. This diagnostic
is reported and never gated, exactly as registered, so nothing is promoted or
condemned on it. It also needs no noise band: an exact floor across 18
independent runs is not a question of statistical resolution.

**The format itself is learnable, and the trend is suggestive rather than
established.** Novel combinations of trained symbols rise monotonically,
0.325 -> 0.411 -> 0.556. But the derived noise bands are 0.40, 0.60 and 0.40,
and the whole three-rung rise is 0.231 -- so every individual step, and the
total, sit inside the band. Three independent rungs each moving the same
direction is a pattern worth reporting; at six runs per rung it is not a
powered claim, and I do not make one. What IS established here is the floor.

**Why the resolution is poor, stated plainly.** One arm at rung 2 spans 0.217
to 0.617 across its three seeds. This task is a per-seed lottery of the same
kind Study S1 measures on the vm niche. Any further Q4 work should buy more
SEEDS before more scale: another rung would sharpen nothing that matters,
while a wider seed set would turn the suggestive trend into a real answer.

## What this does not cover

- **Rung 4** (12L/1024d, 151.4M params, 450M tokens) was never run. Rung 3
  cost 2.3h per run on the fastest instrument available to me, and rung 4
  projects to several times that -- it belongs at greater scale than I
  could reach here. Q2's transfer number and Q1's rank question both
  point at rung 4 as the experiment that matters next.
- **Q4 at more seeds.** All three rungs are measured, but the study is
  noise-limited rather than scale-limited (see its bands above); a wider
  seed set, not a fourth rung, is the experiment that would sharpen the
  trend. The floor result does not need it.
- Rungs 1-2 and rung 3 were measured on two different instruments. Every
  cell is single-instrument by construction, that pair is gated, and no raw
  cross-instrument score comparison is made outside the gated band.
  Absolute scores dropped slightly at rung 3 across all three arms while
  basics rose sharply; instrument change and scale change are confounded
  there, and with no third measurement to separate them I make no claim
  about the cause.

## Reproducing

```bash
ops/p0_gates.sh <device>          # determinism, then golden parity
ops/ladder_rung.sh <rung> <device>  # probe, baseline, calibrate, arms, verdict
ops/study_s2.sh <rung> <device>     # Q2
ops/study_s1.sh <rung> <device>     # Q3
scripts/verdict_report.py --arm <tag> --base <tag> --rung <n>
scripts/make_figures.py
```

Arms resume from hash-, device- and host-stamped caches, so an interrupted
run costs at most one seed. `runs/journal.jsonl` is the append-only record
of every run, gate, verdict and amendment.
