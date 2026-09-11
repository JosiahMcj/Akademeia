import random, pytest
from akademeia.curriculum import (Segment, Curriculum, validate, seeded_curricula,
                                mutate, DIRECT)
from akademeia.data.families import FAMILIES

B = 1_000_000

def test_seeded_curricula_valid():
    cs = seeded_curricula(B)
    assert set(cs) == {"direct", "easy2hard", "basics_then_niche",
                       "basics_then_niche_replay", "interleave_ramp"}
    for c in cs.values():
        validate(c, B)          # raises on any violation
    assert cs["direct"].name == DIRECT and len(cs["direct"].segments) == 1

def test_validate_rejects():
    seg = Segment(tokens=B, weights={"copy": 1.0}, difficulty={"copy": (1, 3)})
    validate(Curriculum("ok", (seg,)), B)
    bad_budget = Curriculum("b1", (Segment(B - 1, {"copy": 1.0}, {"copy": (1, 3)}),))
    with pytest.raises(ValueError): validate(bad_budget, B)
    bad_family = Curriculum("b2", (Segment(B, {"nope": 1.0}, {"nope": (1, 3)}),))
    with pytest.raises(ValueError): validate(bad_family, B)
    bad_replay = Curriculum("b3", (Segment(B, {"copy": 1.0}, {"copy": (1, 3)}, replay_frac=0.5),))
    with pytest.raises(ValueError): validate(bad_replay, B)  # first segment can't replay

def test_mutate_one_gene_and_valid():
    r = random.Random(9)
    base = seeded_curricula(B)["interleave_ramp"]
    for _ in range(100):
        child, gene = mutate(base, r, B)
        validate(child, B)
        assert isinstance(gene, str) and gene  # attributable label
        assert child.name != base.name
