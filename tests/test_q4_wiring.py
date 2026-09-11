"""Q4 wiring laws (amendment of 2026-09-04). The seams that let a niche ride
in a VARIANT pool must not disturb the frozen registry paths, and the built
artifacts must satisfy every registered law -- including the ones the builder
checks, re-checked here against what actually shipped."""
import json, random, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from akademeia.curriculum import Curriculum, Segment, validate
from akademeia.data.families import Example, render
from akademeia.data.pool import build_pool, load_pool
from akademeia.data.bench import load_bench
from akademeia.scratchpad import trace_eval_independent, split_law_ok
from akademeia.tokenizer import CharTokenizer

MANIFEST = json.load(open(ROOT / "freeze" / "manifest.json"))

def test_extra_examples_default_path_is_byte_identical():
    """The seam must be invisible when unused: same hash as no seam at all."""
    a = build_pool(seed=1, per_cell=20)
    b = build_pool(seed=1, per_cell=20, extra_examples=())
    assert a.content_hash == b.content_hash == build_pool(seed=1, per_cell=20).content_hash

def test_extra_examples_are_indexed_and_change_the_hash():
    base = build_pool(seed=1, per_cell=20)
    extra = [Example("scratchpad", 2, "sp-abc", "E=a*b*c;a=3;b=5;c=2", "15;30")]
    var = build_pool(seed=1, per_cell=20, extra_examples=extra)
    assert var.content_hash != base.content_hash      # a variant, not the pool
    assert ("scratchpad", 2) in var.index             # the stream can draw it
    assert len(var.examples) == len(base.examples) + 1

def test_validate_rejects_an_unregistered_family_by_default():
    c = Curriculum("q4", (Segment(100, {"scratchpad": 1.0}, {"scratchpad": (1, 3)}),))
    with pytest.raises(ValueError):
        validate(c, 100)                              # default stays strict
    validate(c, 100, extra_families=("scratchpad",))  # only the named one passes

def test_validate_still_rejects_a_typo_when_extras_are_declared():
    c = Curriculum("q4", (Segment(100, {"scratchpd": 1.0}, {"scratchpd": (1, 3)}),))
    with pytest.raises(ValueError):
        validate(c, 100, extra_families=("scratchpad",))

@pytest.mark.skipif(not (ROOT / "freeze" / "bench-scratchpad.jsonl").exists(),
                    reason="q4 artifacts not built here")
def test_built_bench_obeys_every_registered_law():
    bench = load_bench(ROOT / "freeze" / "bench-scratchpad.jsonl")
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    assert split_law_ok()
    tok.assert_covers([f"I:{i.text_in};O:{i.gold}$" for i in bench])  # zero new chars
    chains = [i for i in bench if i.section != "basics"]
    assert {i.section for i in bench} == {"niche_unseen", "niche_heldrange",
                                          "basics"}
    for i in chains:
        assert trace_eval_independent(i.text_in, i.gold)       # grader accepts gold
    # and the grader must be able to FAIL: a perturbed trace is rejected
    bad = [i for i in chains if trace_eval_independent(i.text_in, i.gold + "0")]
    assert not bad
    # every legal trace fits the evaluator's generation cap
    assert max(len(i.gold) for i in chains) + 1 <= MANIFEST["q4_max_out"]

@pytest.mark.skipif(not (ROOT / "data" / "pool_q4.jsonl").exists(),
                    reason="q4 artifacts not built here")
def test_built_pool_has_no_bench_chain_and_matches_its_pin():
    pool = load_pool(ROOT / "data" / "pool_q4.jsonl")
    assert pool.content_hash == MANIFEST["pool_q4_hash"]
    bench = load_bench(ROOT / "freeze" / "bench-scratchpad.jsonl")
    pool_ins = {e.text_in for e in pool.examples}
    for i in bench:
        if i.section != "basics":
            assert i.text_in not in pool_ins            # held out by construction
    # the pool carries ONLY train-split chain templates
    ids = {e.template_id for e in pool.examples if e.family == "scratchpad"}
    assert ids == {"sp-ab", "sp-abc", "sp-mnc", "sp-bcd"}
