"""Per-seed curve cache: an external kill mid-arm costs at most one run.
Reuse is stamp-gated (hashes+device+salt+seed+warm) -- the reconcile-key
lesson: a cache keyed too loosely adopts the wrong record silently."""
import json, sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_arm
from akademeia.config import SubstrateConfig
from akademeia.curriculum import Segment, Curriculum
from akademeia.data.pool import build_pool, save_pool
from akademeia.data.bench import build_bench, save_bench, bench_hash
from akademeia.tokenizer import CharTokenizer
from akademeia.data.families import render

MICRO = SubstrateConfig(n_layers=2, d_model=32, n_heads=4, ctx=64,
                        batch_size=8, total_tokens=40_000,
                        eval_every_tokens=20_000, lr_warmup_steps=5)

@pytest.fixture
def lab(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir(); (tmp_path / "freeze").mkdir()
    (tmp_path / "data").mkdir()
    pool = build_pool(seed=1, per_cell=60)
    save_pool(pool, tmp_path / "data" / "pool.jsonl")
    tok = CharTokenizer.fit([render(e) for e in pool.examples])
    tok.save(tmp_path / "freeze" / "tokenizer.json")
    bench = build_bench(seed=7, per_section=5)
    save_bench(bench, tmp_path / "freeze" / "bench-physics.jsonl")
    manifest = {"pool_hash": pool.content_hash, "tokenizer_hash": tok.hash(),
                "bench_hash_physics": bench_hash(bench)}
    (tmp_path / "freeze" / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(run_arm, "ROOT", tmp_path)
    c = Curriculum("direct50-physics",
                   (Segment(40_000, {"copy": 1.0, "physics": 1.0},
                            {"copy": (1, 2), "physics": (1, 1)}),))
    return tmp_path, c

def journal_kinds(root):
    return [json.loads(l)["kind"]
            for l in open(root / "runs" / "journal.jsonl")]

def test_seed_curves_cached_and_reused(lab):
    root, c = lab
    a1 = run_arm.run_curriculum(MICRO, c, "physics", seeds=(3, 4), tag="t",
                                device="cpu")
    assert (root / "runs" / "curve-t-s3.json").exists()
    # simulate the mid-arm kill: arm file lost, one seed's curve survives
    (root / "runs" / "arm-t.json").unlink()
    (root / "runs" / "curve-t-s4.json").unlink()
    a2 = run_arm.run_curriculum(MICRO, c, "physics", seeds=(3, 4), tag="t",
                                device="cpu")
    kinds = journal_kinds(root)
    assert "resume_seed" in kinds          # seed 3 came from the cache
    assert a2["final"] == a1["final"]      # and reproduces exactly
    assert a2["basics"] == a1["basics"]

def test_foreign_host_curve_not_adopted(lab):
    # Same device string, different machine: "cpu" on one box and
    # on the Studio's M3 are different instruments. A runs/ tree copied
    # between hosts must re-measure, never adopt.
    root, c = lab
    run_arm.run_curriculum(MICRO, c, "physics", seeds=(3,), tag="t",
                           device="cpu")
    (root / "runs" / "arm-t.json").unlink()
    p = root / "runs" / "curve-t-s3.json"
    cached = json.loads(p.read_text())
    cached["stamp"]["host"] = "some-other-box.local"
    p.write_text(json.dumps(cached))
    run_arm.run_curriculum(MICRO, c, "physics", seeds=(3,), tag="t",
                           device="cpu")
    assert "resume_seed" not in journal_kinds(root)

def test_foreign_host_arm_not_adopted(lab):
    root, c = lab
    run_arm.run_curriculum(MICRO, c, "physics", seeds=(3,), tag="t",
                           device="cpu")
    ap = root / "runs" / "arm-t.json"
    arm = json.loads(ap.read_text())
    assert arm["host"] == run_arm.HOST     # arm carries its instrument's host
    arm["host"] = "some-other-box.local"
    ap.write_text(json.dumps(arm))
    run_arm.run_curriculum(MICRO, c, "physics", seeds=(3,), tag="t",
                           device="cpu")
    kinds = journal_kinds(root)
    assert "stale_arm" in kinds            # loudly re-measured
    assert "resume" not in kinds

def test_stale_stamp_is_not_adopted(lab):
    root, c = lab
    run_arm.run_curriculum(MICRO, c, "physics", seeds=(3,), tag="t",
                           device="cpu")
    (root / "runs" / "arm-t.json").unlink()
    p = root / "runs" / "curve-t-s3.json"
    cached = json.loads(p.read_text())
    cached["stamp"]["device"] = "cuda"     # a different instrument's curve
    p.write_text(json.dumps(cached))
    run_arm.run_curriculum(MICRO, c, "physics", seeds=(3,), tag="t",
                           device="cpu")
    kinds = journal_kinds(root)
    assert "resume_seed" not in kinds      # re-measured, never adopted
