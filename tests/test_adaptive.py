import torch
import pytest
import akademeia.adaptive as adaptive
from akademeia.adaptive import train_adaptive, probe_scores, _fingerprint
from akademeia.config import SubstrateConfig
from akademeia.model import TinyLM
from akademeia.data.pool import build_pool
from akademeia.tokenizer import CharTokenizer
from akademeia.data.families import render, FAMILIES
from akademeia.probes import build_probes
from akademeia.scheduler import LearningProgressScheduler

# sized so each run spans several 50-step loss windows (a run shorter than one
# window records a single loss and cannot show a trend)
MICRO = SubstrateConfig(n_layers=2, d_model=32, n_heads=4, ctx=64,
                        batch_size=8, total_tokens=120_000,
                        eval_every_tokens=40_000, lr_warmup_steps=5)

def setup():
    pool = build_pool(seed=1, per_cell=40)
    tok = CharTokenizer.fit([render(e) for e in pool.examples])
    probes = build_probes(seed=3, per_family=1)
    return pool, tok, probes

def test_probe_scores_are_per_family_and_pure():
    pool, tok, probes = setup()
    torch.manual_seed(0)
    model = TinyLM(MICRO, tok.vocab_size)
    before = _fingerprint(model)
    scores = probe_scores(model, tok, probes, max_out=6)
    assert set(scores) == set(FAMILIES)
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert _fingerprint(model) == before

def test_purity_assertion_is_armed(monkeypatch):
    """Arm the trap before trusting its silence: make probing mutate the model
    and confirm the purity assert actually fires."""
    pool, tok, probes = setup()
    torch.manual_seed(0)
    model = TinyLM(MICRO, tok.vocab_size)
    real_greedy = adaptive._greedy
    def sabotage(m, tokenizer, ids, max_out, stop_id, device):
        with torch.no_grad():
            m.head.weight += 1.0               # a "probe" that trains
        return real_greedy(m, tokenizer, ids, max_out, stop_id, device)
    monkeypatch.setattr(adaptive, "_greedy", sabotage)
    with pytest.raises(AssertionError, match="PURITY"):
        probe_scores(model, tok, probes, max_out=4)

def test_adaptive_run_records_weights_probes_and_checkpoints(tmp_path):
    pool, tok, probes = setup()
    sched = LearningProgressScheduler(sorted(FAMILIES))
    res = train_adaptive(MICRO, pool, tok, seed=7, out_dir=tmp_path,
                         probes=probes, scheduler=sched)
    n_chunks = MICRO.total_tokens // MICRO.eval_every_tokens
    assert len(res.checkpoints) == n_chunks
    assert len(res.weight_history) == n_chunks
    assert len(res.probe_history) == n_chunks
    for _, p in res.checkpoints:
        assert p.exists()
    for w in res.weight_history:
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert set(w) == set(FAMILIES)
    assert res.losses[-1] < res.losses[0]

def test_adaptive_same_seed_reproduces(tmp_path):
    """P0-determinism holds through the chunked path too (CPU, the dev
    device; the accelerator gets its own registered pass)."""
    pool, tok, probes = setup()
    r1 = train_adaptive(MICRO, pool, tok, seed=7, out_dir=tmp_path / "a",
                        probes=probes,
                        scheduler=LearningProgressScheduler(sorted(FAMILIES)))
    r2 = train_adaptive(MICRO, pool, tok, seed=7, out_dir=tmp_path / "b",
                        probes=probes,
                        scheduler=LearningProgressScheduler(sorted(FAMILIES)))
    assert r1.losses == r2.losses
    assert r1.probe_history == r2.probe_history
    assert r1.weight_history == r2.weight_history

def test_probe_informativeness_classifies_families():
    from akademeia.adaptive import probe_informativeness
    hist = [{"a": 0.0, "b": 1.0, "c": 0.2, "d": 0.5},
            {"a": 0.0, "b": 1.0, "c": 0.4, "d": 0.5}]
    r = probe_informativeness(hist)
    assert r["floored"] == ["a"] and r["ceilinged"] == ["b"]
    assert r["informative"] == ["c"]
    assert r["stuck"] == ["d"]              # constant mid-range, steers nothing
    assert r["n_informative"] == 1 and r["n_families"] == 4

def test_probe_informativeness_empty_history():
    from akademeia.adaptive import probe_informativeness
    assert probe_informativeness([])["n_informative"] == 0

def test_mixture_follows_the_scheduler(tmp_path, monkeypatch):
    """A family that keeps improving on probes must gain mass."""
    pool, tok, probes = setup()
    seq = [{f: 0.0 for f in FAMILIES},
           {**{f: 0.0 for f in FAMILIES}, "copy": 0.5},
           {**{f: 0.0 for f in FAMILIES}, "copy": 0.9}]
    calls = {"n": 0}
    def fake_scores(model, tokenizer, probes_, max_out=24, device="cpu"):
        s = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return dict(s)
    monkeypatch.setattr(adaptive, "probe_scores", fake_scores)
    sched = LearningProgressScheduler(sorted(FAMILIES), floor=0.01, cap=0.5)
    res = train_adaptive(MICRO, pool, tok, seed=7, out_dir=tmp_path,
                         probes=probes, scheduler=sched)
    first, last = res.weight_history[0], res.weight_history[-1]
    assert abs(first["copy"] - 1 / len(FAMILIES)) < 1e-9   # starts uniform
    assert last["copy"] > first["copy"]                     # then earns mass
    assert all(v > 0 for v in last.values())                # floor holds
