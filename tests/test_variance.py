import pytest
from akademeia.variance import (seed_outcomes, realization_band, spread,
                                _bf_stat, bf_exact_perm)

TIGHT = [0.50, 0.51, 0.49, 0.50, 0.52, 0.48]
WIDE = [0.10, 0.90, 0.30, 0.70, 0.05, 0.95]

def test_seed_outcomes_are_pair_means():
    a = {"final": [0.4, 0.6], "final_items": {"11": [], "22": []}}
    ap = {"final": [0.6, 0.6], "final_items": {"11": [], "22": []}}
    assert seed_outcomes(a, ap) == [0.5, 0.6]
    assert realization_band(a, ap) == pytest.approx(0.2)

def test_seed_mismatch_asserts():
    a = {"final": [0.4], "final_items": {"11": []}}
    ap = {"final": [0.4], "final_items": {"22": []}}
    with pytest.raises(AssertionError):
        seed_outcomes(a, ap)

def test_bf_stat_zero_for_identical_groups():
    assert _bf_stat(TIGHT, TIGHT) == 0.0

def test_exact_perm_identical_spread_is_insignificant():
    _, p = bf_exact_perm(TIGHT, TIGHT)
    assert p == 1.0

def test_exact_perm_detects_a_real_spread_difference():
    stat, p = bf_exact_perm(TIGHT, WIDE)
    assert stat > 0.2
    assert p < 0.02            # exact floor at n=6+6 is 1/924

def test_exact_perm_is_symmetric_and_deterministic():
    s1, p1 = bf_exact_perm(TIGHT, WIDE)
    s2, p2 = bf_exact_perm(WIDE, TIGHT)
    assert s1 == pytest.approx(s2) and p1 == pytest.approx(p2)
    assert bf_exact_perm(TIGHT, WIDE) == (s1, p1)

def test_exact_perm_enumerates_every_split():
    # C(12,6) = 924 relabelings; the exact p is a multiple of 1/924
    _, p = bf_exact_perm(TIGHT, WIDE)
    assert abs(p * 924 - round(p * 924)) < 1e-9

def test_spread_is_sample_sd():
    assert spread([0.4, 0.6]) == pytest.approx(0.1414213562, rel=1e-6)
