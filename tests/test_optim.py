"""MlxAdamW must implement MLX's measured update rule exactly:
p = p*(1 - lr*wd) - lr * m / (sqrt(v) + eps), NO bias correction."""
import math
import torch
import pytest
from akademeia.optim import MlxAdamW

def one_step(wd: float, g_val: float = 2.0, p_val: float = 1.0,
             lr: float = 0.1):
    p = torch.tensor([p_val], requires_grad=True)
    opt = MlxAdamW([p], lr=lr, weight_decay=wd)
    p.grad = torch.tensor([g_val])
    opt.step()
    return float(p)

def test_first_step_matches_the_mlx_formula_no_bias_correction():
    lr, g = 0.1, 2.0
    m = 0.1 * g                       # (1-b1)*g
    v = 0.001 * g * g                 # (1-b2)*g^2
    expected = 1.0 - lr * m / (math.sqrt(v) + 1e-8)   # wd=0
    assert one_step(wd=0.0) == pytest.approx(expected, rel=1e-6)

def test_decay_is_decoupled_and_applied_before_the_step():
    lr, g, wd = 0.1, 2.0, 0.01
    m = 0.1 * g
    v = 0.001 * g * g
    expected = 1.0 * (1 - lr * wd) - lr * m / (math.sqrt(v) + 1e-8)
    assert one_step(wd=wd) == pytest.approx(expected, rel=1e-6)

def test_no_bias_correction_means_larger_early_steps():
    """torch AdamW's corrected first step is ~lr; MLX's uncorrected first
    step is ~lr*(1-b1)/sqrt(1-b2) = 3.16x LARGER -- a real early-trajectory
    difference the substrates must share."""
    moved = abs(one_step(wd=0.0) - 1.0)
    assert moved == pytest.approx(0.1 * math.sqrt(10), rel=1e-4)

def test_two_optimizers_same_grads_identical():
    torch.manual_seed(0)
    ps = [torch.randn(4, requires_grad=True) for _ in range(2)]
    qs = [p.detach().clone().requires_grad_(True) for p in ps]
    a, b = MlxAdamW(ps, lr=0.01), MlxAdamW(qs, lr=0.01)
    for _ in range(3):
        for p, q in zip(ps, qs):
            g = torch.randn_like(p)
            p.grad, q.grad = g, g.clone()
        a.step(); b.step()
    for p, q in zip(ps, qs):
        assert torch.equal(p, q)

def test_respects_live_lr_updates():
    """The warmup schedule writes group['lr'] every step; the optimizer must
    read it live."""
    p = torch.tensor([1.0], requires_grad=True)
    opt = MlxAdamW([p], lr=1.0, weight_decay=0.0)
    opt.param_groups[0]["lr"] = 0.0
    p.grad = torch.tensor([5.0])
    opt.step()
    assert float(p) == 1.0
