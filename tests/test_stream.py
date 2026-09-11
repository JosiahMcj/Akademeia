import random
from collections import Counter
from akademeia.curriculum import Segment, Curriculum
from akademeia.stream import materialize
from akademeia.data.pool import build_pool
from akademeia.tokenizer import CharTokenizer
from akademeia.data.families import render

def setup():
    pool = build_pool(seed=1, per_cell=50)
    tok = CharTokenizer.fit([render(e) for e in pool.examples])
    return pool, tok

def test_budget_and_determinism():
    pool, tok = setup()
    c = Curriculum("t", (Segment(4000, {"copy": 1.0}, {"copy": (1, 2)}),))
    s1 = materialize(c, pool, tok, seed=5)
    s2 = materialize(c, pool, tok, seed=5)
    assert s1.tokens == s2.tokens and len(s1.tokens) >= 4000
    assert s1.provenance == s2.provenance

def test_provenance_respects_segment():
    pool, tok = setup()
    c = Curriculum("t", (Segment(3000, {"copy": 1.0}, {"copy": (1, 1)}),
                         Segment(3000, {"physics": 1.0}, {"physics": (2, 2)})))
    s = materialize(c, pool, tok, seed=5)
    for fam, dif, seg in s.provenance:
        if seg == 0: assert (fam, dif) == ("copy", 1)
        else: assert fam == "physics" and dif == 2

def test_replay_draws_prior_distribution():
    pool, tok = setup()
    c = Curriculum("t", (Segment(3000, {"copy": 1.0}, {"copy": (1, 1)}),
                         Segment(6000, {"physics": 1.0}, {"physics": (2, 2)},
                                 replay_frac=0.4)))
    s = materialize(c, pool, tok, seed=5)
    fams = Counter(f for f, d, seg in s.provenance if seg == 1)
    frac = fams["copy"] / (fams["copy"] + fams["physics"])
    assert 0.25 < frac < 0.55  # ~0.4 with sampling noise

def test_mixture_weights_respected():
    pool, tok = setup()
    c = Curriculum("t", (Segment(20000, {"copy": 3.0, "reverse": 1.0},
                                 {"copy": (1, 1), "reverse": (1, 1)}),))
    s = materialize(c, pool, tok, seed=5)
    fams = Counter(f for f, d, seg in s.provenance)
    ratio = fams["copy"] / fams["reverse"]
    assert 2.2 < ratio < 4.0
