"""AdamW with MLX's exact update semantics, MEASURED from
mlx.optimizers (2026-08-11): NO bias correction (mlx default
bias_correction=False), decoupled decay applied as p *= (1 - lr*wd) BEFORE
the Adam step, update p -= lr * m / (sqrt(v) + eps).

torch.optim.AdamW always bias-corrects; the uncorrected rule's early steps
are (1-b1)/sqrt(1-b2) = 3.16x LARGER than corrected ones, converging as t
grows -- a real early-trajectory difference at these budgets and one of the
three measured mismatches behind the first parity FAIL. The
substrates must share an update rule for the parity gate to compare like
with like. Pure elementwise ops: deterministic everywhere, and no
accelerator health-check side effects."""
import torch

class MlxAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps, wd = group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                p.mul_(1 - lr * wd)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr)
