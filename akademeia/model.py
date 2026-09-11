"""Tiny decoder-only LM, PyTorch substrate (v3 port). FROZEN per tier; never
searched. The architecture AND initialization mirror the v1/v2 MLX substrate,
MEASURED from it (2026-08-11), so the golden parity gate compares like with
like: embeddings N(0, d^-0.5); four bias-free attention projections at the
Linear default U(+-1/sqrt(fan_in)); additive causal mask; 1/sqrt(head_dim)
query scaling. The first parity attempt failed because the port used torch's
defaults here (embeddings 16x hotter, xavier-packed biased QKV)."""
import torch
import torch.nn as nn
from akademeia.config import SubstrateConfig

class Attention(nn.Module):
    """Mirrors mlx.nn.MultiHeadAttention: separate bias-free q/k/v/out
    projections (torch Linear's default init already matches MLX's
    U(+-1/sqrt(fan_in))), explicit softmax attention -- also keeps every op
    on the deterministic CUDA path."""
    def __init__(self, d: int, h: int):
        super().__init__()
        self.h = h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, mask):
        B, T, D = x.shape
        hd = D // self.h
        q = self.q(x).view(B, T, self.h, hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (hd ** -0.5) + mask
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(out)

class Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))

class TinyLM(nn.Module):
    def __init__(self, cfg: SubstrateConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx, cfg.d_model)
        with torch.no_grad():   # MLX embedding init: N(0, d^-0.5), measured
            self.tok_emb.weight.normal_(0.0, cfg.d_model ** -0.5)
            self.pos_emb.weight.normal_(0.0, cfg.d_model ** -0.5)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.full((T, T), float("-inf"),
                          device=x.device).triu(diagonal=1)
        h = self.tok_emb(x) + self.pos_emb(
            torch.arange(T, device=x.device))
        for b in self.blocks:
            h = b(h, mask)
        return self.head(self.ln_f(h))

def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())
