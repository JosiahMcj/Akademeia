import pytest
from akademeia.stats import (tokens_to_threshold, sign_test, paired_noise,
                           discordant, aulc, powered)

CURVE = [(10, 0.1), (20, 0.4), (30, 0.6), (40, 0.7)]

def test_tokens_to_threshold():
    assert tokens_to_threshold(CURVE, 0.55) == 30
    assert tokens_to_threshold(CURVE, 0.05) == 10
    assert tokens_to_threshold(CURVE, 0.9) is None   # never competent

def test_aulc():
    assert aulc([(10, 0.0), (20, 1.0)]) == pytest.approx(0.5)

def test_discordant_and_sign_test():
    arm =  [True, True, False, True, False]
    base = [False, True, True, True, False]
    wins, losses = discordant(arm, base)
    assert (wins, losses) == (1, 1)
    # 8 wins 1 loss: one-sided binomial p = (C(9,8)+C(9,9)) / 2^9 = 10/512
    p = sign_test(wins=8, losses=1)
    assert p == pytest.approx(10 / 512)
    assert sign_test(0, 0) == 1.0

def test_paired_noise():
    diffs = [0.01, -0.02, 0.015]
    assert paired_noise(diffs) == pytest.approx(0.02)

def test_powered():
    assert powered(5, 3, min_discordant=8)
    assert not powered(4, 3, min_discordant=8)
