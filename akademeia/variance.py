"""Q3 registered variance machinery (PROTOCOL amendment 2026-08-11).

The lottery question is about SPREAD, not means: v2 measured means up,
spread unchanged, and was underpowered at n=3 by its own rule. v3 runs n=6
same-seed A/A' pairs per rung. Laws:

  - A seed's outcome is the pair mean (A+A')/2, which averages out one
    realization draw; the realization band stays reported alongside so the
    spread is never mistaken for pure structure.
  - Spread = sample sd across seed outcomes.
  - Adjacent rungs are compared by an EXACT Brown-Forsythe permutation test
    (all C(12,6)=924 relabelings at n=6+6; group medians recomputed per
    relabeling). Exact tests, never asymptotics, at these n.
"""
from itertools import combinations
from statistics import median, stdev

def seed_outcomes(a: dict, ap: dict) -> list:
    """Per-seed pair means, ordered by seed. a/ap are arm dicts sharing seeds."""
    assert sorted(a["final_items"]) == sorted(ap["final_items"]), \
        "A/A' seed mismatch"
    return [(x + y) / 2 for x, y in zip(a["final"], ap["final"])]

def realization_band(a: dict, ap: dict) -> float:
    return max(abs(x - y) for x, y in zip(a["final"], ap["final"]))

def spread(outcomes) -> float:
    return stdev(outcomes)

def _bf_stat(x, y) -> float:
    """Brown-Forsythe distance: |mean|x-med(x)| - mean|y-med(y)||."""
    zx = [abs(v - median(x)) for v in x]
    zy = [abs(v - median(y)) for v in y]
    return abs(sum(zx) / len(zx) - sum(zy) / len(zy))

def bf_exact_perm(x, y) -> tuple:
    """(stat, p): exact permutation p-value for a spread difference between
    two groups. Enumerates every split of the pooled outcomes into the two
    group sizes; p = fraction of relabelings with a stat at least as large."""
    observed = _bf_stat(x, y)
    pool = list(x) + list(y)
    n = len(x)
    count = total = 0
    for idx in combinations(range(len(pool)), n):
        chosen = set(idx)
        gx = [pool[i] for i in idx]
        gy = [pool[i] for i in range(len(pool)) if i not in chosen]
        if _bf_stat(gx, gy) >= observed - 1e-12:
            count += 1
        total += 1
    return observed, count / total
