"""The D2 classifier must be proven on synthetic examples of every registered
class -- accepted as its own class and rejected as the others -- before it is
pointed at real model output (the known-positive law)."""
import pytest
from akademeia.lengthdiag import classify, parse_trace, summarise, CLASSES

GOLD = "15;30;60"          # a 4-variable chain: three partials

CASES = {
    "CORRECT":                  "15;30;60",
    "SHORT_PREFIX_CORRECT":     "15;30",        # right fold, stopped early
    "SHORT_PREFIX_WRONG":       "15;31",        # short AND wrong
    "RIGHT_LENGTH_WRONG_VALUE": "15;30;61",     # knew the depth, missed a value
    "LONG":                     "15;30;60;120",
    "MALFORMED":                "15;;x",
}

@pytest.mark.parametrize("expected,emitted", list(CASES.items()))
def test_each_class_is_recognised(expected, emitted):
    assert classify(emitted, GOLD) == expected

@pytest.mark.parametrize("expected,emitted", list(CASES.items()))
def test_each_class_excludes_every_other(expected, emitted):
    got = classify(emitted, GOLD)
    for other in CLASSES:
        if other != expected:
            assert got != other, f"{emitted!r} classified as {other}, want {expected}"

def test_empty_and_none_are_malformed():
    assert classify("", GOLD) == "MALFORMED"
    assert classify(None, GOLD) == "MALFORMED"

def test_a_single_correct_partial_is_a_correct_prefix():
    assert classify("15", GOLD) == "SHORT_PREFIX_CORRECT"

def test_a_single_wrong_partial_is_not():
    assert classify("16", GOLD) == "SHORT_PREFIX_WRONG"

def test_parse_rejects_non_integers():
    assert parse_trace("15;30.5") is None
    assert parse_trace("15;30") == [15, 30]

def test_majority_rule_needs_60_percent():
    rows = [("15;30", GOLD)] * 6 + [("15;30;61", GOLD)] * 4
    s = summarise(rows)
    assert s["counts"]["SHORT_PREFIX_CORRECT"] == 6
    assert s["majority"] == "SHORT_PREFIX_CORRECT"      # exactly 60%
    rows = [("15;30", GOLD)] * 5 + [("15;30;61", GOLD)] * 5
    assert summarise(rows)["majority"] is None          # a plurality is not one
