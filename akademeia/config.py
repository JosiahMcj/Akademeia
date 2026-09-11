"""Frozen substrate. Per PROTOCOL, nothing here is searchable; seed is per-run."""
from dataclasses import dataclass, asdict
import hashlib, json

def content_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

@dataclass(frozen=True)
class SubstrateConfig:
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 8
    ctx: int = 256
    batch_size: int = 32
    total_tokens: int = 30_000_000   # B; calibrated once at P0/P2, then frozen
    eval_every_tokens: int = 1_500_000  # E; 20 checkpoints per run
    lr: float = 3e-4
    lr_warmup_steps: int = 100
    def hash(self) -> str:
        return content_hash(asdict(self))
