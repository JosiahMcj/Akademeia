"""Deterministic training, PyTorch substrate (v3 port). Seed fixes init and
stream; deterministic algorithms are enabled; the DEVICE is journaled by the
caller and excluded from the substrate hash (PROTOCOL port gates: cross-device
comparisons are void unless the parity gate covered the pair).

Save-every checkpointing and the no-eval-in-trainer split carry from v2."""
from dataclasses import dataclass
import os
from pathlib import Path
# use_deterministic_algorithms on CUDA hard-errors at the first matmul unless
# cuBLAS gets a fixed workspace; must be set before the first cuBLAS call
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from akademeia.model import TinyLM
from akademeia.stream import materialize
from akademeia.optim import MlxAdamW

@dataclass
class RunResult:
    losses: list                 # mean loss per 50-step window
    checkpoints: list            # [(tokens_seen, Path)]

def _batches(tokens, ctx, bs, device):
    step = bs * (ctx + 1)
    for off in range(0, len(tokens) - step + 1, step):
        yield torch.tensor(tokens[off:off + step],
                           dtype=torch.long, device=device).view(bs, ctx + 1)

def tf32_enabled() -> bool:
    """Whether the TF32 tensor-core path is permitted. Set EXPLICITLY rather
    than inherited: an unset sampling/precision knob silently adopts whatever
    the platform defaults to, and on new hardware that default can flip a
    gate from pass to fail while every line of the code is unchanged. AK_TF32
    is recorded on every run row, so the instrument a number came from is
    always readable off the ledger."""
    return os.environ.get("AK_TF32", "0") == "1"

def apply_precision_policy() -> dict:
    """Pin the precision policy and return it for journalling."""
    tf32 = tf32_enabled()
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return {"tf32": tf32}

def train_run(cfg, curriculum, pool, tokenizer, seed: int, out_dir,
              stream_salt: int = 0, init_from=None,
              device: str = "cpu") -> RunResult:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    apply_precision_policy()
    torch.manual_seed(seed)
    model = TinyLM(cfg, tokenizer.vocab_size).to(device)
    if init_from is not None:
        model.load_state_dict(torch.load(init_from, map_location=device))
    opt = MlxAdamW(model.parameters(), lr=cfg.lr)
    stream_seed = (seed * 1_000_003 + stream_salt) % 2**31
    stream = materialize(curriculum, pool, tokenizer, seed=stream_seed)
    losses, window, checkpoints = [], [], []
    tokens_seen, next_ckpt, step = 0, cfg.eval_every_tokens, 0
    model.train()
    for xy in _batches(stream.tokens, cfg.ctx, cfg.batch_size, device):
        step += 1
        lr = cfg.lr * min(1.0, step / cfg.lr_warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = xy[:, :-1], xy[:, 1:]
        # flattened (N, vocab) form: the (B, vocab, T) spatial variant routes
        # to nll_loss2d, which has no deterministic CUDA kernel
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        window.append(loss.item())
        if len(window) == 50:
            losses.append(sum(window) / len(window)); window = []
        tokens_seen += cfg.batch_size * cfg.ctx
        if tokens_seen >= next_ckpt:
            p = out_dir / f"ckpt_{tokens_seen}.pt"
            torch.save(model.state_dict(), p)
            checkpoints.append((tokens_seen, p))
            next_ckpt += cfg.eval_every_tokens
    if window:
        losses.append(sum(window) / len(window))
    if not checkpoints or checkpoints[-1][0] < tokens_seen:
        p = out_dir / f"ckpt_{tokens_seen}.pt"
        torch.save(model.state_dict(), p)
        checkpoints.append((tokens_seen, p))
    return RunResult(losses, checkpoints)
