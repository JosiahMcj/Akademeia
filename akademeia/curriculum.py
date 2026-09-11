"""Curriculum genome (spec section 6). Small + attributable: mutations touch
one gene at a time and carry a label."""
from dataclasses import dataclass, replace
import random
from akademeia.data.families import FAMILIES, BASIC_FAMILIES

DIRECT = "direct"

@dataclass(frozen=True)
class Segment:
    tokens: int
    weights: dict            # family -> float, positive, normalized at stream time
    difficulty: dict         # family -> (lo, hi), 1 <= lo <= hi <= 3
    replay_frac: float = 0.0

@dataclass(frozen=True)
class Curriculum:
    name: str
    segments: tuple

def validate(c: Curriculum, budget: int, extra_families=()) -> None:
    """extra_families: family names that live in a VARIANT pool rather than
    the frozen FAMILIES registry (see build_pool's extra_examples). Empty by
    default, so the registry check is unchanged for every existing caller."""
    if not c.segments or len(c.segments) > 4:
        raise ValueError("1..4 segments")
    if sum(s.tokens for s in c.segments) != budget:
        raise ValueError("segment tokens must sum to budget")
    for i, s in enumerate(c.segments):
        if s.tokens <= 0: raise ValueError("segment tokens must be positive")
        if not s.weights: raise ValueError("empty weights")
        for fam, w in s.weights.items():
            if fam not in FAMILIES and fam not in extra_families:
                raise ValueError(f"unknown family {fam}")
            if w <= 0: raise ValueError("weights must be positive")
        for fam, (lo, hi) in s.difficulty.items():
            if fam not in s.weights: raise ValueError("difficulty for unweighted family")
            if not (1 <= lo <= hi <= 3): raise ValueError("bad difficulty band")
        if set(s.difficulty) != set(s.weights):
            raise ValueError("every weighted family needs a difficulty band")
        if not (0 <= s.replay_frac < 1): raise ValueError("replay_frac in [0,1)")
        if i == 0 and s.replay_frac != 0: raise ValueError("first segment cannot replay")

def _uni(fams, band=(1, 3)):
    return {f: 1.0 for f in fams}, {f: band for f in fams}

ALL = tuple(BASIC_FAMILIES) + ("physics",)

def seeded_curricula(budget: int) -> dict:
    w_all, d_all = _uni(ALL)
    w_bas, d_bas = _uni(BASIC_FAMILIES)
    w_phy, d_phy = _uni(("physics",))
    third, half = budget // 3, budget // 2
    return {
        "direct": Curriculum(DIRECT, (Segment(budget, w_all, d_all),)),
        "easy2hard": Curriculum("easy2hard", (
            Segment(third, w_all, {f: (1, 1) for f in ALL}),
            Segment(third, w_all, {f: (1, 2) for f in ALL}),
            Segment(budget - 2 * third, w_all, {f: (2, 3) for f in ALL}))),
        "basics_then_niche": Curriculum("basics_then_niche", (
            Segment(half, w_bas, d_bas),
            Segment(budget - half, w_phy, d_phy))),
        "basics_then_niche_replay": Curriculum("basics_then_niche_replay", (
            Segment(half, w_bas, d_bas),
            Segment(budget - half, w_phy, d_phy, replay_frac=0.25))),
        "interleave_ramp": Curriculum("interleave_ramp", (
            Segment(third, {**{f: 1.0 for f in BASIC_FAMILIES}, "physics": 0.2},
                    {**{f: (1, 2) for f in BASIC_FAMILIES}, "physics": (1, 1)}),
            Segment(third, {**{f: 1.0 for f in BASIC_FAMILIES}, "physics": 1.0},
                    {**{f: (1, 3) for f in BASIC_FAMILIES}, "physics": (1, 2)}),
            Segment(budget - 2 * third, {**{f: 0.5 for f in BASIC_FAMILIES}, "physics": 2.0},
                    {**{f: (2, 3) for f in BASIC_FAMILIES}, "physics": (2, 3)}))),
    }

def curriculum_to_json(c: Curriculum) -> dict:
    return {"name": c.name, "segments": [
        {"tokens": s.tokens, "weights": s.weights,
         "difficulty": {f: list(b) for f, b in s.difficulty.items()},
         "replay_frac": s.replay_frac} for s in c.segments]}

def curriculum_from_json(d: dict) -> Curriculum:
    return Curriculum(d["name"], tuple(
        Segment(s["tokens"], dict(s["weights"]),
                {f: tuple(b) for f, b in s["difficulty"].items()},
                s.get("replay_frac", 0.0)) for s in d["segments"]))

def mutate(parent: Curriculum, r: random.Random, budget: int, max_tries: int = 200):
    """One-gene mutation; returns (child, gene_label). Retries until valid."""
    for _ in range(max_tries):
        segs = list(parent.segments)
        i = r.randrange(len(segs)); s = segs[i]
        op = r.choice(["boundary", "weight", "band", "replay"])
        try:
            if op == "boundary" and len(segs) >= 2:
                j = i if i + 1 < len(segs) else i - 1   # neighbor to trade with
                a, b = min(i, j), min(i, j) + 1
                delta = int(segs[a].tokens * r.choice([-0.1, 0.1]))
                segs[a] = replace(segs[a], tokens=segs[a].tokens - delta)
                segs[b] = replace(segs[b], tokens=segs[b].tokens + delta)
                gene = f"boundary:{a}|{b}:{delta:+d}"
            elif op == "weight":
                fam = r.choice(list(s.weights))
                factor = r.choice([0.5, 0.75, 1.5, 2.0])
                w = dict(s.weights); w[fam] = w[fam] * factor
                segs[i] = replace(s, weights=w)
                gene = f"weight:{i}:{fam}:x{factor}"
            elif op == "band":
                fam = r.choice(list(s.difficulty))
                lo, hi = s.difficulty[fam]
                lo2, hi2 = r.choice([(max(1, lo - 1), hi), (min(lo + 1, hi), hi),
                                     (lo, max(lo, hi - 1)), (lo, min(3, hi + 1))])
                d = dict(s.difficulty); d[fam] = (lo2, hi2)
                segs[i] = replace(s, difficulty=d)
                gene = f"band:{i}:{fam}:{lo2}-{hi2}"
            elif op == "replay" and i > 0:
                rf = round(min(0.9, max(0.0, s.replay_frac + r.choice([-0.1, 0.1]))), 2)
                segs[i] = replace(s, replay_frac=rf)
                gene = f"replay:{i}:{rf}"
            else:
                continue
            child = Curriculum(f"{parent.name}~{gene}", tuple(segs))
            validate(child, budget)
            return child, gene
        except ValueError:
            continue
    raise RuntimeError("mutation failed to produce a valid child")
