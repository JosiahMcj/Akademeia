from akademeia.gates import gate, Thresholds

TH = Thresholds(margin_tokens=2_000_000, noise_final=0.02, noise_basics=0.02,
                T=0.6, min_discordant=8)
HASHES = {"bench": "b1", "pool": "p1", "tokenizer": "t1", "substrate": "s1"}

def mk(tt=(10_000_000, 14_000_000), final=(0.7, 0.65), basics=(0.8, 0.8),
       wins=9, losses=1):
    arm = {"tokens_to_T": [tt[0]] * 3, "final": [final[0]] * 3,
           "basics": [basics[0]] * 3, "wins": wins, "losses": losses,
           "hashes": dict(HASHES)}
    base = {"tokens_to_T": [tt[1]] * 3, "final": [final[1]] * 3,
            "basics": [basics[1]] * 3, "hashes": dict(HASHES)}
    return arm, base

def test_promotes_when_all_pass():
    arm, base = mk()
    v = gate(arm, base, TH)
    assert v["promote"] and all(v["gates"].values())

def test_g1_margin():
    arm, base = mk(tt=(13_500_000, 14_000_000))     # only 0.5M faster < margin
    assert not gate(arm, base, TH)["gates"]["G1"]

def test_g2_final_regression():
    arm, base = mk(final=(0.60, 0.65))
    assert not gate(arm, base, TH)["gates"]["G2"]

def test_g3_basics_regression():
    arm, base = mk(basics=(0.70, 0.80))
    assert not gate(arm, base, TH)["gates"]["G3"]

def test_g4_hash_mismatch_voids():
    arm, base = mk()
    arm["hashes"]["bench"] = "EVIL"
    v = gate(arm, base, TH)
    assert v["void"] and not v["promote"]

def test_g5_underpowered_is_not_a_negative():
    arm, base = mk(wins=3, losses=1)
    v = gate(arm, base, TH)
    assert not v["promote"] and v["underpowered"]

def test_never_competent_ranks_worst():
    arm, base = mk()
    arm["tokens_to_T"] = [None] * 3
    assert not gate(arm, base, TH)["gates"]["G1"]

def test_baseline_none_tts_voids_not_promotes():
    # the P3 rejudge bug: a sentinel baseline made G1 trivially true
    arm, base = mk()
    base["tokens_to_T"] = [None] * 3
    v = gate(arm, base, TH)
    assert v["void"] and not v["promote"]
    assert "baseline" in v["reason"]

def test_tts_from_curves_recomputes_at_registered_T():
    from akademeia.stats import tts_from_curves
    curves = {"11": [[10, 0.1, 0.9], [20, 0.4, 0.9], [30, 0.7, 0.9]],
              "22": [[10, 0.05, 0.9], [20, 0.2, 0.9], [30, 0.25, 0.9]]}
    assert tts_from_curves(curves, 0.4) == [20, None]
