from akademeia.gates import gate_v2, ThresholdsV2

TH = ThresholdsV2(noise_final=0.05, noise_basics=0.05, min_discordant=8)
HASHES = {"bench": "b", "pool": "p", "tokenizer": "t", "substrate": "s"}

def mk(arm_final=0.45, base_final=0.35, arm_basics=0.7, base_basics=0.7,
       wins=20, losses=5):
    arm = {"final": [arm_final] * 3, "basics": [arm_basics] * 3,
           "wins": wins, "losses": losses, "hashes": dict(HASHES)}
    base = {"final": [base_final] * 3, "basics": [base_basics] * 3,
            "hashes": dict(HASHES)}
    return arm, base

def test_superior_promotes():
    v = gate_v2(*mk(), TH, mode="superior")
    assert v["promote"] and all(v["gates"].values())

def test_superior_fails_h2_within_noise():
    v = gate_v2(*mk(arm_final=0.38), TH, mode="superior")   # +0.03 < 0.05
    assert not v["gates"]["H2"]

def test_superior_fails_h1_not_significant():
    v = gate_v2(*mk(wins=6, losses=4), TH, mode="superior")   # p=0.38, powered
    assert not v["gates"]["H1"] and not v["underpowered"]

def test_underpowered_flagged():
    v = gate_v2(*mk(wins=3, losses=2), TH, mode="superior")   # n=5 < 8
    assert v["underpowered"] and not v["gates"]["H1"]

def test_noninferior_passes_when_equal():
    v = gate_v2(*mk(arm_final=0.34, wins=10, losses=12), TH, mode="noninferior")
    assert v["promote"]                                     # within band, no sig loss

def test_noninferior_fails_when_significantly_worse():
    v = gate_v2(*mk(arm_final=0.34, wins=2, losses=20), TH, mode="noninferior")
    assert not v["gates"]["H1"]

def test_retention_reported_not_gated_for_specialist():
    v = gate_v2(*mk(arm_basics=0.05), TH, mode="superior", retention_gates=False)
    assert v["promote"] and v["retention_delta"] < -0.6

def test_retention_gates_generalist():
    v = gate_v2(*mk(arm_basics=0.05), TH, mode="superior", retention_gates=True)
    assert not v["gates"]["H4"]

def test_defective_input_voids():
    arm, base = mk()
    base["final"] = [None] * 3
    v = gate_v2(arm, base, TH)
    assert v["void"] and "base" in v["reason"]
