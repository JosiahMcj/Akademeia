import hashlib
import torch
from akademeia.config import SubstrateConfig
from akademeia.model import TinyLM
from akademeia.tokenizer import CharTokenizer
from akademeia.data.bench import build_bench, save_bench
from akademeia.evalrunner import eval_checkpoint, grade

MICRO = SubstrateConfig(n_layers=2, d_model=64, n_heads=4, ctx=64)

def test_grade_exact():
    assert grade("15", "15") is True
    assert grade("-3", "-3") is True
    assert grade("16", "15") is False
    assert grade("15 ", "15") is False

def test_eval_checkpoint_pristine_and_pure(tmp_path):
    bench = build_bench(seed=7, per_section=5)
    texts = [f"I:{i.text_in};O:{i.gold}$" for i in bench]
    tok = CharTokenizer.fit(texts)
    tok.save(tmp_path / "tok.json")
    save_bench(bench, tmp_path / "bench.jsonl")
    torch.manual_seed(0)
    model = TinyLM(MICRO, tok.vocab_size)
    ck = tmp_path / "ck.pt"
    torch.save(model.state_dict(), ck)
    before = hashlib.sha256(ck.read_bytes()).hexdigest()
    res = eval_checkpoint(MICRO, ck, tmp_path / "bench.jsonl",
                          tmp_path / "tok.json", max_out=12)
    assert hashlib.sha256(ck.read_bytes()).hexdigest() == before
    assert set(res["scores"]) == {"niche_unseen", "niche_heldrange", "basics"}
    for s, v in res["scores"].items():
        assert 0.0 <= v <= 1.0 and res["n"][s] == 5
