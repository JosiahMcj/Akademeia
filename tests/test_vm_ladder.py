"""Amendment 4: vm must be arithmetically comparable to physics, and its
held-out split must cover every complexity level."""
import random
from akademeia.data.families import (FAMILIES, _VM_LADDER, _vm_shape_space,
                                    _vm_bench_shapes, _vm_split, _vm_legal,
                                    vm_eval_independent)

def rng(): return random.Random(4242)

def test_every_combination_stratum_has_held_out_shapes():
    """Amendment 5: COMBINATION strata (>=2 ops) must each contribute held-out
    shapes -- but 1-op strata never do, because holding out a primitive makes
    that operation unlearnable (that bug scored 0.03 against 1.00
    in-distribution)."""
    bench = _vm_bench_shapes()
    for pushes, lo, hi in _VM_LADDER.values():
        for n_ops in range(lo, hi + 1):
            space = _vm_shape_space(pushes, n_ops)
            if not space:
                continue
            held = [s for s in space if s in bench]
            if n_ops == 1:
                assert not held, f"primitive stratum pushes={pushes} is held out"
            elif len(space) >= 2:
                assert held, f"stratum pushes={pushes} nops={n_ops} has no bench shape"
                assert len(held) < len(space), "a stratum must keep train shapes too"

def test_split_is_deterministic_and_total():
    for pushes, lo, hi in _VM_LADDER.values():
        for n_ops in range(lo, hi + 1):
            for s in _vm_shape_space(pushes, n_ops):
                assert _vm_split(s) in ("train", "bench")
                assert _vm_split(s) == _vm_split(s)

def test_d1_d2_carry_one_real_arithmetic_op():
    """swap is depth-neutral, so d1/d2 are single-operation programs -- the
    arithmetic load physics d1/d2 has."""
    real = {"add", "sub", "mul"}
    for d in (1, 2):
        pushes, lo, hi = _VM_LADDER[d]
        for n_ops in range(lo, hi + 1):
            for shape in _vm_shape_space(pushes, n_ops):
                assert sum(1 for o in shape if o in real) == 1

def test_all_generated_shapes_are_legal():
    r = rng()
    for d in (1, 2, 3):
        pushes = _VM_LADDER[d][0]
        for _ in range(50):
            ex = FAMILIES["vm"].gen(r, d)
            ops = tuple(ex.template_id[3:].split(","))
            assert _vm_legal(pushes, ops)
            assert vm_eval_independent(ex.text_in) == int(ex.text_out)

def _easy_fraction(family: str, n: int = 900) -> float:
    """Share of golds with <= 2 digits -- the items a tiny model can actually
    get right, and therefore the items that let a bench discriminate."""
    r = rng()
    easy = 0
    for i in range(n):
        ex = FAMILIES[family].gen(r, (i % 3) + 1)
        easy += len(ex.text_out.lstrip("-")) <= 2
    return easy / n

def test_vm_easy_fraction_is_comparable_to_physics():
    """The bench can only measure where the model can sometimes succeed. The
    first vm design chained multiplications (golds like 74646) and scored
    0.033; physics scored 0.450 with ~half its golds at <= 2 digits."""
    vm, physics = _easy_fraction("vm"), _easy_fraction("physics")
    assert vm >= 0.30, f"vm easy fraction {vm:.2f} too low to discriminate"
    assert abs(vm - physics) < 0.30, f"vm {vm:.2f} vs physics {physics:.2f}"

def test_d1_golds_are_single_or_double_digit():
    r = rng()
    for _ in range(200):
        assert len(FAMILIES["vm"].gen(r, 1).text_out.lstrip("-")) <= 2

def test_bench_split_generation_works_at_combination_difficulties():
    """d1 is deliberately all-trainable, so bench draws come from d2-3."""
    r = rng()
    for d in (2, 3):
        ex = FAMILIES["vm"].gen(r, d, split="bench")
        assert _vm_split(tuple(ex.template_id[3:].split(","))) == "bench"
