"""Phase C (scale ladder) verdict statistics, registered BEFORE any tier-2 run.

The question V2-Q1 asks is ordinal: do tier-1 curriculum rankings survive a
scale-up? With 3-4 arms there are only 3-6 pairs, so the honest instrument is
exact pair counting, not a correlation coefficient pretending to a large n:

- concordant fraction over UNTIED pairs (ties carry no ordinal information --
  the discordant-pairs law), reported with the raw pair counts
- top-1 agreement (did the best tier-1 arm stay best?)
- and per-arm H-gates at tier-2 against tier-2-calibrated thresholds; tier-1
  thresholds are NEVER reused because paired noise must be re-measured at the
  new substrate.
"""

def rank_agreement(tier1: dict, tier2: dict) -> dict:
    """tier1/tier2: arm name -> mean final score at that tier."""
    names = sorted(tier1)
    if sorted(tier2) != names:
        raise ValueError(f"arm sets differ: {names} vs {sorted(tier2)}")
    if len(names) < 2:
        raise ValueError("rank agreement needs at least 2 arms")
    concordant = discordant = ties = 0
    disagreements = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d1 = tier1[a] - tier1[b]
            d2 = tier2[a] - tier2[b]
            if d1 == 0 or d2 == 0:
                ties += 1
            elif (d1 > 0) == (d2 > 0):
                concordant += 1
            else:
                discordant += 1
                disagreements.append((a, b))
    untied = concordant + discordant
    top1_t1 = max(names, key=lambda n: tier1[n])
    top1_t2 = max(names, key=lambda n: tier2[n])
    return {"concordant": concordant, "discordant": discordant, "ties": ties,
            "untied_pairs": untied,
            "agreement": (concordant / untied) if untied else None,
            "top1_tier1": top1_t1, "top1_tier2": top1_t2,
            "top1_match": top1_t1 == top1_t2,
            "disagreements": disagreements}
