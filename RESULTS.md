# Akademeia results

v3 of my curriculum machine. The question line is the same one v1 and v2
asked -- does the order you feed a small model its training data change
what it can do -- and v3 asks whether the answers survive an order of
magnitude of scale. Every threshold below was frozen from same-seed A/A'
pairs before the arm it judges ever ran; every number is read out of the
run ledgers in `runs/`, never retyped.

Four rungs completed: 4L/256d (3.2M params, 30M tokens), 6L/384d (10.7M,
45M tokens), 8L/640d (39.5M, 135M tokens) and 12L/1024d (151.4M, 450M
tokens). That is a 47x span in parameters and 15x in tokens.

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
| torch, accelerator C, strict fp32 | 0.5611 | -0.028 |
| torch, accelerator C, TF32 | 0.5833 | -0.006 |

Accelerator C is a cluster node class. I registered both of its precision
settings as separate candidates before running either, and both passed; the
rung-4 and length-wall runs used TF32. Rather than assume its nodes were
interchangeable, I trained the same seed on four GPUs across three nodes: the
loss curves were bitwise identical, so the class is one instrument.

Determinism is gated per device (same seed trained twice, curves compared
at 1e-6) and device pairs are gated before any comparison crosses them: the
two gated pairs differed by 0.006 and 0.022. Accelerator C has no direct pair
gate with the earlier instruments. Two comparisons cross that boundary, and I
flag both where they appear: the rung 3 -> rung 4 rank comparison in Q1, and
the length-wall studies, whose zero baseline was measured on accelerator B
and processor B. Parity failed on the
first attempt at -0.18; the cause was three port-spec mismatches found by
measuring the MLX substrate directly (embedding init scale, attention
projection shape and init, and the optimizer's uncorrected update with
pre-step decoupled decay), not by tuning toward the gate.

## Q1: the separated ranking survives; the ordinal measure never settles

Two arms are judged against the mixed 50/50 baseline at each rung, with
thresholds re-derived per rung. Four rungs were measured, spanning 3.2M to
151.4M parameters -- a 47x range.

| rung | branch vs direct50 | adaptive2 vs direct50 |
|---|---|---|
| 1 | 8W/18L, not promoted, retention -0.55 | 7W/3L, non-inferior |
| 2 | 5W/19L, not promoted, retention -0.59 | 6W/3L, non-inferior |
| 3 | 4W/21L, not promoted, P(worse)=0.0005, retention -0.81 | 9W/13L, non-inferior |
| 4 | 5W/11L, non-inferior, P(worse)=0.105, retention -0.83 | 8W/5L, non-inferior |

**The finding that does not move is the cost, not the score.** Core-then-branch
retains 0.02 of the basics at rung 4 against the baseline's 0.85, and its
retention delta is -0.55, -0.59, -0.81, -0.83 across the four rungs. Training
a niche on top of a frozen basics core destroys the basics, at 3.2M parameters
and still at 151M, and 47x of scale does not soften it at all. That is the
clearest result in the study and it rests on a quantity no gate had to
adjudicate.

**Branch's niche disadvantage, by contrast, stopped being significant at rung
4, and the honest reading is ambiguous.** Its win rate improved (5 of 16
discordant items, against 4 of 25 at rung 3) but the number of discordant
items fell from 25 to 16, so the test lost power at the same time the arm
looked better. Whether the arm genuinely caught up or the instrument simply
stopped being able to tell cannot be separated from these data, and no claim
is made either way.

**The ordinal rank measure flipped twice, both times on the same pair.** Rank
agreement was 1.0 from rung 1 to 2, then 0.667 from 2 to 3, then 0.667 again
from 3 to 4 -- and the single discordant pair was (adaptive2, direct50) every
time. Their separation at rung 3 was 0.022 and at rung 4 is 0.017. That pair
was measured directly with six runs per arm at rung 3: a difference of 0.008
against an uncertainty of 0.034, a quarter of its own error bar, head-to-head
3-3. (Rung 4 ran on accelerator C and rung 3 on accelerator B, with no direct
pair gate between them, so the rung 3 -> 4 step also crosses instruments.) It is a coin flip, and across four rungs it has landed adaptive2, then
adaptive2, then direct50, then adaptive2.

So the four-rung answer to the flagship question separates cleanly into two
statements. A ranking with real separation in it -- branch against the field,
on the quantity that actually differs -- survived 47x of scale intact. The
ordinal statistic layered on top of it never settled, because it consumes only
the SIGN of a difference and one of its three pairs has no real difference to
take the sign of. The pre-registered prediction that rankings would scramble
beyond tier 2 is technically borne out, and is an artifact of the measure
rather than a fact about curricula.

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
then clearly off it at rung 3 -- and still nowhere near the margin it would
have to clear.

Because that was the one surprising positive number in the study, I measured
it again with three more seeds per arm before publishing it. **It shrank.**
At twelve runs the rung-3 transfer estimate is 0.198 +- 0.035, against the
0.270 that three seeds reported -- about three quarters of it. The movement
is still real: it sits 0.143 above the rung-1 and rung-2 floor, about four
times the standard error, and the twelve runs range from 0.00 to 0.33 with
two of them at exactly zero. So the wall moved at rung 3, by less than the
first look suggested, and nowhere near enough to clear its margin. "The wall
persists" survives as the verdict; "the wall is absolute" would have been
wrong; and "the wall broke" would have been an overstatement I would have
published without the extra seeds.

## Q3: the lottery does not narrow

Structural generalization on the vm niche, same-seed pair means per rung,
exact Brown-Forsythe permutation on the spread:

| step | pairs | instrument | spread | exact p | verdict |
|---|---|---|---|---|---|
| rung 1 -> 2 | 6 | processor B | 0.0665 -> 0.0636 | 0.712 | flat |
| rung 2 -> 3 | 6 | processor B, then accelerator B | 0.0636 -> 0.0164 | 0.390 | flat |
| rung 1 -> 2 | 12 | accelerator C | 0.0557 -> 0.0962 | 0.283 | flat |
| rung 2 -> 3 | 12 | accelerator C | 0.0962 -> 0.0432 | 0.097 | flat |

The registered design used six pairs per rung and stated in its own results
that it was underpowered at that size. So the pairs were doubled to twelve,
registered in advance, with the test, the alpha and the original verdicts all
left untouched. **FLAT survives at both steps.** That is a materially stronger
claim than the first one: scale-invariance of this lottery has now survived a
doubling of power rather than merely failing to reach significance once. The
permutation test remains exact at twelve pairs (2,704,156 relabelings), so
nothing about the method was relaxed to afford the larger design.

The twelve-pair cells are single-instrument: all twelve seeds of every rung,
including the original six, were trained on the same accelerator, because the
extension's new tag meant the original seeds were measured again rather than
reused. The six-pair rows came from a different machine. So the two halves of
this table are not a "more seeds" comparison, and an earlier draft of this
document wrongly read them as one: it said rung 3's spread "grew from 0.0164
to 0.0432 once the seeds doubled." It did not. The same six seeds, measured
again on the accelerator, already give a rung-3 spread of 0.0474. The change
came from the instrument, not from the extra seeds. Seeds that passed the same
parity gate still land at different per-seed values (0 of 6 identical at
rung 3), which is what a mean-level gate permits and a spread estimate can
feel.

What the extension does establish is the verdict: on one instrument, with
twelve pairs per rung, FLAT holds at both steps. Means rise with scale on that
instrument (0.124 -> 0.287 -> 0.396): the lottery's centre moves up and its
width does not shrink.

## Q4: writing the intermediates down is learnable, and buys nothing for length

The scratchpad format gives the model a task in which it must emit each
partial product before the answer (`E=a*b*c;a=3;b=5;c=2` -> `15;30`), graded
FULL-TRACE: a wrong intermediate with a lucky final answer is wrong. This is
a different task from the plain format, and it is registered as such -- what
follows is a statement about the scratchpad task on its own terms, never a
comparison against the plain one.

| rung | novel combinations | one step longer than training |
|---|---|---|
| 1 | 0.336 +- 0.015 (n=12) | 0.000 |
| 2 | 0.389 +- 0.033 (n=12) | 0.000 |
| 3 | 0.544 +- 0.028 (n=12) | 0.000 |

**Length extrapolation is a hard zero: 0.000 in 36 runs out of 36, across
all three rungs.** One more factor in the chain than training ever showed,
using only symbols the model knows, in a format it has already demonstrated
on novel combinations -- and every run, at every scale, scores nothing.
Twelve times the parameters moved it not at all. Writing the intermediates
down did not buy the step that the depth wall in Study S2 is made of. This
diagnostic is reported and never gated, exactly as registered, so nothing is
promoted or condemned on it. It also needs no uncertainty interval: an exact
floor across 36 independent runs is not a question of statistical resolution.

**The format itself is learnable, and improves with scale unevenly.** Each
rung was first measured at three seeds and then extended by three more,
registered in advance, for twelve runs per rung; the thresholds frozen from
the original pairs were left exactly as they were, and what follows is
descriptive. Rung 1 to rung 3 moves +0.208 against a combined uncertainty of
0.031 -- about six and a half times its own uncertainty. The intermediate
steps are not equal: rung 1 to rung 2 is +0.053 against 0.036, barely larger
than the uncertainty itself, while rung 2 to rung 3 is +0.156 against 0.043.
Most of the gain arrives in the last step, and the first step is not
distinguishable from noise at this sample size.

The extension is also a caution worth recording. At three seeds per rung the
rung 1 to rung 2 gap read 0.086; at twelve it reads 0.053. Doubling the seeds
shrank the apparent effect rather than confirming it, which is what small
samples usually do to a promising-looking step.

Rung 3 was measured on a different instrument from rungs 1 and 2. The gated
difference between those two instruments is 0.022, against a rung 2 to rung 3
step of 0.156, so the step is not an artifact of the change -- but it is a
confound the reader should hold, and no cross-instrument claim is made
outside the gated band.

**On the per-seed lottery.** Individual runs still scatter widely -- one
rung-2 arm spanned 0.217 to 0.617 across its three original seeds -- so this
task behaves like the lottery Study S1 measures on the vm niche. Twelve runs
per rung is enough to place the means; it is not enough to make any claim
about single-run behaviour, and none is made.

## The length wall, taken apart (D2 and D3)

Q4's firmest number was an exact zero: chains one factor longer than training
scored 0.000 in 36 of 36 runs. Three follow-up studies, each registered before
its data existed, asked what the model actually does there.

**It never attempts the extra step (D2).** I decoded the model's output on those
chains and classified every trace. One step deeper than training, the model
wrote the required three partial products in 0 of 60 items at rung 1 and 0 of
60 at rung 3: all 120 stopped at two. It is not failing at the third
multiplication. It never writes one.

All D2 and D3 runs used accelerator C, while Q4's zero came from accelerators
and processors on another machine. D2's 0 of 120 is a count on its own
instrument and needs no cross-machine comparison. D3's comparison is within
itself, against a fixed threshold.

**Training on longer chains helps the small model and not the large one (D3).**
I added length-4 chains to training and tested (a,b,c,d), the exact template
that had scored zero, now a new combination at a length the model had seen.

| rung | held-out (a,b,c,d), 6 runs | attempts 3 partials | length 5 |
|---|---|---|---|
| 1 (3.2M) | 0.267 +- 0.069 | 222 of 360 | 0 of 6 runs |
| 3 (39.5M) | 0.025 +- 0.014 | 17 of 360 | 0 of 6 runs |

At rung 1 the model carried length-4 over to the new combination. At rung 3,
trained on the same chains, it stopped at two steps in 343 of 360 outputs. The
larger model learned length 4 for the chains it saw, not as a rule it could
reuse. Both rungs held length 5 at exactly zero, as I predicted beforehand:
the boundary sits wherever training stopped. Two cautions. The rung-1 score
swings from 0.02 to 0.47 across its six runs, so its "exposure helped"
reading rests on a noisy mean. And the gap runs the opposite way to what
scale usually buys.

**What the large model does instead is a tendency, not a mechanism (D3-ops).**
In the rung-3 two-step outputs, step 1 is a*b 98% of the time, and step 2
most often multiplies by d, the last number, skipping c: 55%, against 19% for
c. That falls short of the 60% I registered for calling it a mechanism, so I
report it as a plurality. Six hand-picked examples had suggested 4 of 6; the
full count is lower, the same shrinkage the note on three seeds describes.

## A note on three seeds

Three numbers in this study stood out enough that I went back and measured
them again with double the seeds, each time registered in advance and each
time leaving the original verdict untouched. All three shrank:

| quantity | at 3 seeds | at 6 runs per arm |
|---|---|---|
| Q4 rung 1 to rung 2 step | 0.086 | 0.053 |
| Q1 rung-3 rank gap | 0.022 | 0.008 |
| Q2 rung-3 transfer | 0.270 | 0.198 |

That is not bad luck, and it is not evidence that the machine is broken. It
is what selection does: I chose to re-measure the estimates that looked most
notable, and standing out is exactly the property a small-sample estimate
acquires by being lucky. Every one of them moved toward something duller when
asked again.

Q3's rung-3 spread is not a fourth example, though an earlier draft of this
document counted it as one. Its change between the six-pair and twelve-pair
tables comes from a change of instrument (see Q3), and all three examples
above were re-measured on the instrument that produced the original.

The practical consequence for a reader: any three-seed number in this study
that has NOT been extended should be read as an upper bound on its own
effect, not as a point estimate. The ones that have been extended are
labelled where they appear. The results that do not depend on a small-sample
mean at all -- the item-level sign tests, which run on 9 to 26 discordant
items per comparison, every one above the registered floor of 8, and the exact zeros, which are counts rather than estimates -- are
untouched by this concern, and they carry the study's firmest claims.

## What this does not cover

- **Rung 5 and beyond.** Rung 4 is the top of this ladder. Its own
  calibration is the reason to be careful about going further with three
  seeds: the rung-4 noise band came out at 0.55, wider than any score it
  judges, because the registered estimator is three times the WORST of three
  same-seed pairs and one pair happened to differ by 0.184. At that width the
  margin test is nearly vacuous and the verdicts rest on the item-level sign
  test alone. A fifth rung would want more seeds per arm before more scale.
- **Q4 at more than twelve seeds.** All three rungs are measured at twelve
  runs each. That resolves the overall trend and leaves the first step
  (rung 1 to rung 2) unresolved; more seeds would settle it, though what it
  would settle is a detail of the curve rather than the finding.
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
