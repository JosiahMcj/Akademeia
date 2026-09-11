"""Pristine-directory eval, PyTorch substrate (v3 port): temp dir holds only
{checkpoint, tokenizer, bench}; the checkpoint's sha256 is asserted unchanged
(the purity law); greedy decode; exact-match grading."""
import hashlib, shutil, tempfile
from pathlib import Path
import torch
from akademeia.model import TinyLM
from akademeia.tokenizer import CharTokenizer
from akademeia.data.bench import load_bench

def grade(output: str, gold: str) -> bool:
    return output == gold

def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.no_grad()
def _greedy(model, tok, prompt_ids, max_out: int, stop_id: int,
            device: str) -> list:
    ids = list(prompt_ids)
    for _ in range(max_out):
        x = torch.tensor([ids[-model.cfg.ctx:]], dtype=torch.long,
                         device=device)
        nxt = int(model(x)[0, -1].argmax().item())
        if nxt == stop_id:
            break
        ids.append(nxt)
    return ids[len(prompt_ids):]

def eval_checkpoint(cfg, ckpt_path, bench_path, tok_path, max_out: int = 24,
                    device: str = "cpu") -> dict:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for src in (ckpt_path, bench_path, tok_path):
            shutil.copy(src, td / Path(src).name)
        ck = td / Path(ckpt_path).name
        before = _sha(ck)
        tok = CharTokenizer.load(td / Path(tok_path).name)
        bench = load_bench(td / Path(bench_path).name)
        model = TinyLM(cfg, tok.vocab_size).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        stop_id = tok.encode("$")[0]
        items, scores, ns = {}, {}, {}
        for it in bench:
            out_ids = _greedy(model, tok, tok.encode(f"I:{it.text_in};O:"),
                              max_out, stop_id, device)
            items.setdefault(it.section, []).append(
                bool(grade(tok.decode(out_ids), it.gold)))
        for sec, flags in items.items():
            ns[sec] = len(flags)
            scores[sec] = sum(flags) / len(flags)
        assert _sha(ck) == before, "PURITY VIOLATION: checkpoint changed during eval"
        return {"scores": scores, "items": items, "n": ns}
