import torch
from akademeia.config import SubstrateConfig
from akademeia.model import TinyLM, param_count

def small(): return SubstrateConfig(n_layers=2, d_model=64, n_heads=4, ctx=32)

def test_forward_shape():
    m = TinyLM(small(), vocab_size=40)
    out = m(torch.zeros(3, 16, dtype=torch.long))
    assert out.shape == (3, 16, 40)

def test_param_count_target_range():
    n = param_count(TinyLM(SubstrateConfig(), vocab_size=40))
    assert 2_000_000 <= n <= 4_500_000  # tier-1 matches the v1/v2 substrate

def test_init_deterministic():
    torch.manual_seed(7); a = TinyLM(small(), 40)
    torch.manual_seed(7); b = TinyLM(small(), 40)
    assert torch.equal(a.tok_emb.weight, b.tok_emb.weight)
    assert torch.equal(a.head.weight, b.head.weight)

def test_causal_mask_blocks_future():
    """A future-token change must not affect earlier positions' logits."""
    torch.manual_seed(0)
    m = TinyLM(small(), 40); m.eval()
    with torch.no_grad():
        x1 = torch.tensor([[1, 2, 3, 4]])
        x2 = torch.tensor([[1, 2, 3, 9]])     # only the LAST token differs
        o1, o2 = m(x1), m(x2)
    assert torch.allclose(o1[0, :3], o2[0, :3], atol=1e-6)
    assert not torch.allclose(o1[0, 3], o2[0, 3])
