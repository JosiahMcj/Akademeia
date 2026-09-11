from akademeia.tiers import RUNGS, TIER1, TIER4
from akademeia.model import TinyLM, param_count

VOCAB = 40   # the frozen tokenizer's size class; exact vocab shifts params ~0.1%

def counts():
    return {r: param_count(TinyLM(cfg, VOCAB)) for r, cfg in RUNGS.items()}

def test_rung_param_targets():
    n = counts()
    assert 2_000_000 <= n[1] <= 4_500_000
    assert 10_000_000 <= n[2] <= 11_500_000
    assert 38_000_000 <= n[3] <= 41_000_000
    assert 148_000_000 <= n[4] <= 155_000_000

def test_ladder_spans_an_order_of_magnitude():
    n = counts()
    for r in (2, 3, 4):
        assert n[r] / n[r - 1] >= 3.0          # every step is a real step
    assert n[4] / n[1] >= 40.0                 # the Q1 claim's span

def test_twenty_checkpoints_per_run_every_rung():
    for cfg in RUNGS.values():
        assert cfg.total_tokens % cfg.eval_every_tokens == 0
        assert cfg.total_tokens // cfg.eval_every_tokens == 20

def test_heads_divide_width_and_frame_is_constant():
    for cfg in RUNGS.values():
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.ctx == TIER1.ctx and cfg.batch_size == TIER1.batch_size

def test_budgets_scale_chinchilla_ish():
    for prev, cur in ((TIER1, RUNGS[2]), (RUNGS[2], RUNGS[3]),
                      (RUNGS[3], TIER4)):
        assert cur.total_tokens > prev.total_tokens
    assert TIER4.total_tokens / TIER1.total_tokens >= 10.0
