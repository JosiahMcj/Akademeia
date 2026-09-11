import numpy as np
from akademeia.config import SubstrateConfig
from akademeia.trainer import train_run
from akademeia.curriculum import Segment, Curriculum
from akademeia.data.pool import build_pool
from akademeia.tokenizer import CharTokenizer
from akademeia.data.families import render

MICRO = SubstrateConfig(n_layers=2, d_model=64, n_heads=4, ctx=64,
                        batch_size=8, total_tokens=40_000,
                        eval_every_tokens=20_000, lr_warmup_steps=5)

def setup():
    pool = build_pool(seed=1, per_cell=50)
    tok = CharTokenizer.fit([render(e) for e in pool.examples])
    c = Curriculum("t", (Segment(40_000, {"copy": 1.0}, {"copy": (1, 1)}),))
    return pool, tok, c

def test_loss_decreases_and_checkpoints(tmp_path):
    pool, tok, c = setup()
    res = train_run(MICRO, c, pool, tok, seed=3, out_dir=tmp_path)
    assert res.losses[-1] < res.losses[0] * 0.9
    assert len(res.checkpoints) == 2
    for _, path in res.checkpoints:
        assert path.exists()

def test_same_seed_reproduces(tmp_path):
    """P0-determinism on CPU (the development device). The accelerator gets
    its own registered pass before any experiment counts (PROTOCOL)."""
    pool, tok, c = setup()
    r1 = train_run(MICRO, c, pool, tok, seed=3, out_dir=tmp_path / "a")
    r2 = train_run(MICRO, c, pool, tok, seed=3, out_dir=tmp_path / "b")
    assert np.allclose(r1.losses, r2.losses, rtol=1e-6)

def test_different_seed_differs(tmp_path):
    pool, tok, c = setup()
    r1 = train_run(MICRO, c, pool, tok, seed=3, out_dir=tmp_path / "a")
    r2 = train_run(MICRO, c, pool, tok, seed=4, out_dir=tmp_path / "b")
    assert abs(r1.losses[-1] - r2.losses[-1]) > 0

def test_warm_start_loads(tmp_path):
    pool, tok, c = setup()
    core = train_run(MICRO, c, pool, tok, seed=3, out_dir=tmp_path / "core")
    warm = train_run(MICRO, c, pool, tok, seed=4, out_dir=tmp_path / "warm",
                     init_from=core.checkpoints[-1][1])
    fresh = train_run(MICRO, c, pool, tok, seed=4, out_dir=tmp_path / "fresh")
    assert warm.losses[0] < fresh.losses[0] * 0.8
