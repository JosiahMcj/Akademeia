"""Curriculum -> ordered token stream. Deterministic given (curriculum, pool,
seed). Every example's provenance is recorded and hard-asserted against the
active segment (spec: training only reads the curriculum-ordered stream)."""
from dataclasses import dataclass
import random

@dataclass
class Stream:
    tokens: list          # flat token ids, >= budget
    provenance: list      # one (family, difficulty, segment_idx) per example, in order

def _draw(r, pool, weights, difficulty):
    fams = sorted(weights)
    total = sum(weights[f] for f in fams)
    x = r.random() * total
    for f in fams:
        x -= weights[f]
        if x <= 0:
            fam = f; break
    else:
        fam = fams[-1]
    lo, hi = difficulty[fam]
    d = r.randint(lo, hi)
    idxs = pool.index[(fam, d)]
    ex = pool.examples[r.choice(idxs)]
    assert (ex.family, ex.difficulty) == (fam, d), "provenance violation"
    return ex

def materialize(curriculum, pool, tokenizer, seed: int) -> Stream:
    from akademeia.data.families import render
    r = random.Random(seed)
    tokens, prov = [], []
    for si, seg in enumerate(curriculum.segments):
        target = sum(s.tokens for s in curriculum.segments[:si + 1])
        prior = curriculum.segments[:si]
        while len(tokens) < target:
            if prior and seg.replay_frac > 0 and r.random() < seg.replay_frac:
                p = prior[r.randrange(len(prior))]
                ex = _draw(r, pool, p.weights, p.difficulty)
            else:
                ex = _draw(r, pool, seg.weights, seg.difficulty)
            tokens.extend(tokenizer.encode(render(ex)))
            prov.append((ex.family, ex.difficulty, si))
    return Stream(tokens, prov)
