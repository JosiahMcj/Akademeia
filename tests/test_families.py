import random, re
import pytest
from akademeia.data.families import (
    FAMILIES, Example, render, prompt, physics_eval_independent, PHYSICS_TEMPLATES)

def rng(): return random.Random(1234)

def test_registry_shape():
    assert set(f.kind for f in FAMILIES.values()) == {"basic", "niche"}
    assert "physics" in FAMILIES and FAMILIES["physics"].kind == "niche"
    assert len([f for f in FAMILIES.values() if f.kind == "basic"]) == 10

@pytest.mark.parametrize("name", list(FAMILIES))
@pytest.mark.parametrize("difficulty", [1, 2, 3])
def test_examples_wellformed(name, difficulty):
    fam = FAMILIES[name]
    r = rng()
    for _ in range(50):
        ex = fam.gen(r, difficulty)
        assert isinstance(ex, Example) and ex.family == name
        assert ex.difficulty == difficulty
        assert "$" not in ex.text_in and "$" not in ex.text_out
        assert render(ex) == f"I:{ex.text_in};O:{ex.text_out}$"
        assert prompt(ex) == f"I:{ex.text_in};O:"

def test_reverse_gold_independent():
    fam = FAMILIES["reverse"]; r = rng()
    for _ in range(50):
        ex = fam.gen(r, 2)
        payload = ex.text_in.split("rev ", 1)[1]
        assert ex.text_out == payload[::-1]

def test_addchain_gold_independent():
    fam = FAMILIES["addchain"]; r = rng()
    for _ in range(50):
        ex = fam.gen(r, 3)
        assert int(ex.text_out) == eval(ex.text_in)  # input is a pure +/- chain

def test_physics_gold_independent_and_integer():
    fam = FAMILIES["physics"]; r = rng()
    for _ in range(200):
        ex = fam.gen(r, 2)
        got = physics_eval_independent(ex.text_in)
        assert str(got) == ex.text_out
        assert re.fullmatch(r"-?\d+", ex.text_out)

def test_physics_templates_split_disjoint():
    train = {t.template_id for t in PHYSICS_TEMPLATES if t.split == "train"}
    bench = {t.template_id for t in PHYSICS_TEMPLATES if t.split == "bench"}
    assert train and bench and not (train & bench)
    assert len(bench) >= 5

def test_physics_gen_respects_split():
    fam = FAMILIES["physics"]; r = rng()
    train_ids = {t.template_id for t in PHYSICS_TEMPLATES if t.split == "train"}
    for _ in range(100):
        assert fam.gen(r, 2).template_id in train_ids
    bench_ids = {t.template_id for t in PHYSICS_TEMPLATES if t.split == "bench"}
    for _ in range(50):
        ex = fam.gen(r, 2, split="bench")
        assert ex.template_id in bench_ids
