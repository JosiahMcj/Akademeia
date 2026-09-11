"""Amendment 5: a held-out group must be a novel COMBINATION of primitives the
model can learn elsewhere -- never a primitive itself.

Holding out ('mul',) made multiplication unlearnable at every tractable level
while the bench stayed full of it. The model scored 1.00 in-distribution at d1
and 0.03 on the bench. Nothing errored; only the split was wrong.
"""
from akademeia.data.families import (_vm_bench_shapes, _vm_shape_space,
                                    _VM_LADDER, _ARITH, PHYSICS_TEMPLATES)

def _trainable_primitives():
    bench = _vm_bench_shapes()
    prims = set()
    for pushes, lo, hi in _VM_LADDER.values():
        for shape in _vm_shape_space(pushes, 1):
            if shape not in bench:
                prims.update(o for o in shape if o in _ARITH)
    return prims

def test_every_primitive_is_trainable():
    assert _trainable_primitives() == set(_ARITH)

def test_no_single_op_shape_is_held_out():
    bench = _vm_bench_shapes()
    for pushes, lo, hi in _VM_LADDER.values():
        for shape in _vm_shape_space(pushes, 1):
            assert shape not in bench, f"{shape} is a primitive, not a combination"

def test_held_out_combinations_use_only_trainable_primitives():
    prims = _trainable_primitives()
    for shape in _vm_bench_shapes():
        assert len(shape) >= 2
        for op in shape:
            if op in _ARITH:
                assert op in prims

def test_every_held_out_stratum_keeps_training_shapes():
    bench = _vm_bench_shapes()
    for pushes, lo, hi in _VM_LADDER.values():
        for n_ops in range(lo, hi + 1):
            space = _vm_shape_space(pushes, n_ops)
            held = [s for s in space if s in bench]
            if held:
                assert len(held) < len(space)

def test_physics_heldout_templates_reuse_trained_operators():
    """The physics analogue of the same law: every held-out formula's operators
    appear in trained formulas, so unseen means new combination, not new op."""
    train_ops, bench_ops = set(), set()
    for t in PHYSICS_TEMPLATES:
        ops = {c for c in t.expr if c in "*/+-"}
        (bench_ops if t.split == "bench" else train_ops).update(ops)
    assert bench_ops <= train_ops
