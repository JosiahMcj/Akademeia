import random, re
import pytest
from akademeia.data.families import (FAMILIES, NICHE_FAMILIES, _vm_shape,
                                    _vm_split, vm_eval_independent, _VM_LADDER)

def rng(): return random.Random(77)

def test_registered_as_niche():
    assert NICHE_FAMILIES == ("physics", "vm")
    assert FAMILIES["vm"].kind == "niche"

@pytest.mark.parametrize("d", [1, 2, 3])
def test_vm_gold_independent(d):
    fam = FAMILIES["vm"]; r = rng()
    for _ in range(100):
        ex = fam.gen(r, d)
        assert re.fullmatch(r"-?\d+", ex.text_out)
        assert vm_eval_independent(ex.text_in) == int(ex.text_out)
        assert ex.text_in.endswith(";out")

def test_vm_shapes_stack_legal():
    r = rng()
    for d in (1, 2, 3):
        for _ in range(50):
            ops = _vm_shape(r, d)
            depth = _VM_LADDER[d][0]        # read the ladder, never hardcode it
            for op in ops:
                assert depth >= (1 if op == "dup" else 2)
                depth += 1 if op == "dup" else (0 if op == "swap" else -1)
            assert depth == 1

def test_vm_split_deterministic_and_disjoint():
    r = rng()
    shapes = {_vm_shape(r, 3) for _ in range(200)}
    splits = {s: _vm_split(s) for s in shapes}
    assert set(splits.values()) == {"train", "bench"}   # both exist
    for s, sp in splits.items():
        assert _vm_split(s) == sp                        # stable

def test_vm_gen_respects_split():
    fam = FAMILIES["vm"]; r = rng()
    for _ in range(50):
        assert _vm_split(tuple(
            fam.gen(r, 2).template_id[3:].split(","))) == "train"
    for _ in range(20):
        ex = fam.gen(r, 2, split="bench")
        assert _vm_split(tuple(ex.template_id[3:].split(","))) == "bench"
