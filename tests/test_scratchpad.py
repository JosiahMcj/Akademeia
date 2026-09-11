import random
from pathlib import Path
from akademeia.scratchpad import (CHAIN_TEMPLATES, ChainTemplate, gold_trace,
                                  render_chain, gen_item,
                                  trace_eval_independent, split_law_ok)
from akademeia.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
ABC = next(t for t in CHAIN_TEMPLATES if t.template_id == "sp-abc")

def test_render_and_gold_trace():
    text_in, text_out = render_chain(ABC, {"a": 3, "b": 5, "c": 2})
    assert text_in == "E=a*b*c;a=3;b=5;c=2"
    assert text_out == "15;30"          # left-fold partials; last = answer

def test_independent_grader_accepts_only_the_full_correct_trace():
    text_in = "E=a*b*c;a=3;b=5;c=2"
    assert trace_eval_independent(text_in, "15;30")
    assert not trace_eval_independent(text_in, "16;30")   # wrong step, right answer
    assert not trace_eval_independent(text_in, "15;31")   # right step, wrong answer
    assert not trace_eval_independent(text_in, "30")      # answer without the trace
    assert not trace_eval_independent(text_in, "15;30;")  # trailing junk
    assert not trace_eval_independent("garbage", "15;30")

def test_every_gold_trace_passes_its_own_grader():
    r = random.Random(0)
    for t in CHAIN_TEMPLATES:
        for d in (1, 2, 3):
            for _ in range(20):
                text_in, text_out = gen_item(t, r, d)
                assert trace_eval_independent(text_in, text_out)

def test_zero_new_characters_against_the_frozen_tokenizer():
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    r = random.Random(1)
    texts = []
    for t in CHAIN_TEMPLATES:
        for d in (1, 2, 3):
            text_in, text_out = gen_item(t, r, d)
            texts.append(f"I:{text_in};O:{text_out}$")
    tok.assert_covers(texts)

def test_registered_split_law_holds():
    assert split_law_ok()

def test_split_law_detects_violations():
    bad_symbol = CHAIN_TEMPLATES + (
        ChainTemplate("sp-xyz", ("x", "y", "z"), "bench"),)   # new primitives
    assert not split_law_ok(bad_symbol)
    bad_len = CHAIN_TEMPLATES + (
        ChainTemplate("sp-4b", ("a", "b", "c", "m"), "bench"),)  # untrained length
    assert not split_law_ok(bad_len)
    dup = CHAIN_TEMPLATES + (
        ChainTemplate("sp-dup", ("a", "b", "c"), "bench"),)   # verbatim train tuple
    assert not split_law_ok(dup)

def test_generation_is_deterministic():
    a = [gen_item(ABC, random.Random(7), 2) for _ in range(5)]
    b = [gen_item(ABC, random.Random(7), 2) for _ in range(5)]
    assert a == b

def test_module_is_standalone_additive():
    """Nothing in the package may import scratchpad until the wiring
    amendment lands -- the live-code-edit-is-a-deploy law."""
    pkg = ROOT / "akademeia"
    for f in pkg.rglob("*.py"):
        if f.name == "scratchpad.py":
            continue
        assert "scratchpad" not in f.read_text(), f"{f} imports scratchpad"
