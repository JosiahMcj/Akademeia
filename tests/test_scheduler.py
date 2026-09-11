import pytest
from akademeia.scheduler import LearningProgressScheduler

FAMS = ["a", "b", "c", "d"]

def sched(**kw):
    return LearningProgressScheduler(FAMS, **kw)

def test_first_update_is_uniform():
    s = sched()
    w = s.update({f: 0.0 for f in FAMS})
    assert all(abs(v - 0.25) < 1e-9 for v in w.values())

def test_mass_moves_to_the_learning_family():
    s = sched(floor=0.02, cap=0.9)
    s.update({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0})
    w = s.update({"a": 0.4, "b": 0.0, "c": 0.0, "d": 0.0})   # only a improved
    assert w["a"] > 0.5
    assert all(w["a"] > w[f] for f in ("b", "c", "d"))
    assert abs(sum(w.values()) - 1.0) < 1e-9

def test_floor_keeps_saturated_families_alive():
    """v1 measured that starving a family destroys it (basics 0.73 -> 0.017)."""
    s = sched(floor=0.05, cap=0.9)
    s.update({f: 0.0 for f in FAMS})
    w = s.update({"a": 0.9, "b": 0.0, "c": 0.0, "d": 0.0})
    for f in ("b", "c", "d"):
        assert w[f] > 0.0

def test_cap_prevents_single_family_collapse():
    s = sched(floor=0.01, cap=0.35)
    s.update({f: 0.0 for f in FAMS})
    w = s.update({"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0})
    assert w["a"] <= 0.35 + 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9

def test_regression_is_not_rewarded():
    s = sched()
    s.update({"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5})
    w = s.update({"a": 0.1, "b": 0.5, "c": 0.5, "d": 0.5})   # a got WORSE
    assert w["a"] <= min(w[f] for f in ("b", "c", "d")) + 1e-9

def test_no_progress_anywhere_falls_back_to_uniform():
    s = sched()
    s.update({f: 0.3 for f in FAMS})
    w = s.update({f: 0.3 for f in FAMS})
    assert all(abs(v - 0.25) < 1e-9 for v in w.values())

def test_deterministic_and_weights_positive():
    a, b = sched(), sched()
    for scores in ({"a": 0.0, "b": 0.1, "c": 0.2, "d": 0.3},
                   {"a": 0.4, "b": 0.1, "c": 0.5, "d": 0.3}):
        wa, wb = a.update(scores), b.update(scores)
        assert wa == wb
        assert all(v > 0 for v in wa.values())

def test_missing_family_raises():
    s = sched()
    with pytest.raises(ValueError):
        s.update({"a": 0.1})

def test_bad_config_raises():
    with pytest.raises(ValueError):
        LearningProgressScheduler(FAMS, cap=0.1)     # cap*4 < 1, impossible
    with pytest.raises(ValueError):
        LearningProgressScheduler([])

# ---- fixed shares (Amendment 8, arm 2) ----

def test_fixed_share_held_exactly_every_update():
    s = LearningProgressScheduler(FAMS + ["niche"], floor=0.02, cap=0.35,
                                  fixed={"niche": 0.5})
    w0 = s.current()
    assert w0["niche"] == 0.5
    for scores in ({f: 0.0 for f in FAMS + ["niche"]},
                   {**{f: 0.0 for f in FAMS}, "a": 0.9, "niche": 0.7},
                   {**{f: 0.5 for f in FAMS}, "niche": 0.9}):
        w = s.update(scores)
        assert w["niche"] == 0.5                       # pinned, never steered
        assert sum(w.values()) == pytest.approx(1.0)
        assert sum(w[f] for f in FAMS) == pytest.approx(0.5)

def test_fixed_progress_moves_only_within_steered():
    s = LearningProgressScheduler(FAMS + ["niche"], floor=0.02, cap=0.35,
                                  fixed={"niche": 0.5})
    s.update({f: 0.0 for f in FAMS + ["niche"]})
    w = s.update({**{f: 0.0 for f in FAMS}, "a": 0.8, "niche": 0.9})
    assert w["a"] > w["b"]                             # a earned steered mass
    assert w["niche"] == 0.5                           # niche progress ignored
    assert w["a"] <= 0.35 + 1e-9

def test_fixed_validation():
    with pytest.raises(ValueError):
        LearningProgressScheduler(FAMS, fixed={"zzz": 0.5})
    with pytest.raises(ValueError):
        LearningProgressScheduler(FAMS, fixed={"a": 1.0})
    with pytest.raises(ValueError):
        LearningProgressScheduler(FAMS, fixed={"a": 0.4, "b": 0.3,
                                               "c": 0.2, "d": 0.15})
