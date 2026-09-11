import pytest
from akademeia.ranks import rank_agreement

def test_perfect_agreement():
    t1 = {"a": 0.5, "b": 0.4, "c": 0.3}
    t2 = {"a": 0.7, "b": 0.6, "c": 0.2}
    r = rank_agreement(t1, t2)
    assert r["agreement"] == 1.0 and r["top1_match"]
    assert (r["concordant"], r["discordant"], r["ties"]) == (3, 0, 0)

def test_full_reversal():
    t1 = {"a": 0.5, "b": 0.4, "c": 0.3}
    t2 = {"a": 0.2, "b": 0.3, "c": 0.4}
    r = rank_agreement(t1, t2)
    assert r["agreement"] == 0.0 and not r["top1_match"]
    assert r["top1_tier2"] == "c"
    assert len(r["disagreements"]) == 3

def test_ties_carry_no_information():
    t1 = {"a": 0.5, "b": 0.5, "c": 0.3}
    t2 = {"a": 0.6, "b": 0.4, "c": 0.3}
    r = rank_agreement(t1, t2)
    assert r["ties"] == 1                      # (a,b) tied at tier 1
    assert r["untied_pairs"] == 2
    assert r["agreement"] == 1.0

def test_all_ties_reports_none_not_a_number():
    r = rank_agreement({"a": 0.5, "b": 0.5}, {"a": 0.4, "b": 0.6})
    assert r["untied_pairs"] == 0 and r["agreement"] is None

def test_mismatched_arms_refuse():
    with pytest.raises(ValueError):
        rank_agreement({"a": 1, "b": 2}, {"a": 1, "c": 2})

def test_single_arm_refuses():
    with pytest.raises(ValueError):
        rank_agreement({"a": 1}, {"a": 1})
