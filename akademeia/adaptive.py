"""Chunked adaptive training (v2-Q2 machinery, PyTorch substrate for v3-Q1's
adaptive2 arm).

Loop: materialize one checkpoint-sized chunk under the scheduler's current
mixture -> train -> checkpoint -> score dev probes -> scheduler.update().
The frozen bench is never touched here (selection-on-test law); probing is
asserted non-mutating by a parameter fingerprint taken around every probe.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # see trainer.py
import torch
import torch.nn.functional as F

from akademeia.model import TinyLM
from akademeia.stream import materialize
from akademeia.optim import MlxAdamW
from akademeia.curriculum import Segment, Curriculum, validate
from akademeia.evalrunner import _greedy
from akademeia.trainer import _batches

FULL_BAND = (1, 3)

@dataclass
class AdaptiveResult:
    losses: list = field(default_factory=list)
    checkpoints: list = field(default_factory=list)     # [(tokens, Path)]
    weight_history: list = field(default_factory=list)  # per chunk, pre-train
    probe_history: list = field(default_factory=list)   # per chunk, post-train

def _fingerprint(model) -> float:
    """Cheap parameter fingerprint; equality across a probe proves the probe
    did not train the model (the impure-forward hole, measured in mach)."""
    with torch.no_grad():
        total = sum(p.sum() for p in model.parameters())
    return float(total.item())

def probe_scores(model, tokenizer, probes, max_out: int = 24,
                 device: str = "cpu") -> dict:
    """Per-family probe accuracy. Asserts the model is unchanged."""
    before = _fingerprint(model)
    stop_id = tokenizer.encode("$")[0]
    hits, totals = {}, {}
    for p in probes:
        out_ids = _greedy(model, tokenizer, tokenizer.encode(f"I:{p.text_in};O:"),
                          max_out, stop_id, device)
        ok = tokenizer.decode(out_ids) == p.gold
        hits[p.family] = hits.get(p.family, 0) + int(ok)
        totals[p.family] = totals.get(p.family, 0) + 1
    after = _fingerprint(model)
    assert before == after, "PURITY VIOLATION: probing changed the weights"
    return {f: hits[f] / totals[f] for f in totals}

def probe_informativeness(probe_history) -> dict:
    """Which probe families could actually steer allocation.

    A family pinned at 0.0 or 1.0 for the whole run contributes no
    d(score)/d(token) and the scheduler can only hold it at the floor -- the
    saturated-benchmark trap that left an earlier machine with 3 informative
    tasks out of 19 and a promotion that was noise. Report it, never infer
    from a silent scheduler that allocation was 'working'.
    """
    if not probe_history:
        return {"informative": [], "floored": [], "ceilinged": [], "stuck": [],
                "n_informative": 0, "n_families": 0}
    fams = sorted(probe_history[0])
    informative, floored, ceilinged, stuck = [], [], [], []
    for f in fams:
        vals = [h[f] for h in probe_history]
        if len(set(vals)) > 1:
            informative.append(f)
        elif vals[0] == 0.0:
            floored.append(f)
        elif vals[0] == 1.0:
            ceilinged.append(f)
        else:
            stuck.append(f)          # constant mid-range: partly learned, frozen
    return {"informative": informative, "floored": floored,
            "ceilinged": ceilinged, "stuck": stuck,
            "n_informative": len(informative), "n_families": len(fams)}

def train_adaptive(cfg, pool, tokenizer, seed: int, out_dir, probes, scheduler,
                   stream_salt: int = 0, bands=None,
                   device: str = "cpu") -> AdaptiveResult:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = TinyLM(cfg, tokenizer.vocab_size).to(device)
    opt = MlxAdamW(model.parameters(), lr=cfg.lr)
    res = AdaptiveResult()
    chunk_tokens = cfg.eval_every_tokens
    n_chunks = max(1, cfg.total_tokens // chunk_tokens)
    step, tokens_seen, window = 0, 0, []
    for chunk in range(n_chunks):
        weights = dict(scheduler.current())
        band = {f: (bands or {}).get(f, FULL_BAND) for f in weights}
        curr = Curriculum(f"adaptive-c{chunk}",
                          (Segment(chunk_tokens, weights, band),))
        validate(curr, chunk_tokens)
        res.weight_history.append(weights)
        stream = materialize(curr, pool, tokenizer,
                             seed=(seed * 1_000_003 + stream_salt + chunk) % 2**31)
        model.train()
        for xy in _batches(stream.tokens, cfg.ctx, cfg.batch_size, device):
            step += 1
            lr = cfg.lr * min(1.0, step / cfg.lr_warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = xy[:, :-1], xy[:, 1:]
            logits = model(x)   # flattened form; see trainer.py
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            window.append(loss.item())
            if len(window) == 50:
                res.losses.append(sum(window) / len(window)); window = []
            tokens_seen += cfg.batch_size * cfg.ctx
        p = out_dir / f"ckpt_{tokens_seen}.pt"
        torch.save(model.state_dict(), p)
        res.checkpoints.append((tokens_seen, p))
        model.eval()
        scores = probe_scores(model, tokenizer, probes, device=device)
        res.probe_history.append(scores)
        scheduler.update(scores)
    if window:
        res.losses.append(sum(window) / len(window))
    return res
