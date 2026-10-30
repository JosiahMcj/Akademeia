"""D3-trace (registered 2026-10-30): decode the final D3 checkpoints on the
D3 bench and classify every emitted trace with the frozen D2 taxonomy."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from akademeia.data.bench import load_bench
from akademeia.evalrunner import _greedy
from akademeia.journal import Journal
from akademeia.lengthdiag import summarise, classify
from akademeia.model import TinyLM
from akademeia.tiers import RUNGS
from akademeia.tokenizer import CharTokenizer
ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = RUNGS[a.rung]
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    bench = load_bench(ROOT / "freeze" / "bench-d3.jsonl")
    stop = tok.encode("$")[0]
    rows = {"niche_unseen": [], "niche_heldrange": []}
    per_ckpt, ex = [], []
    for nm in ("A", "Aprime"):
        for s in (11, 22, 33):
            d = ROOT / "runs" / f"d3-r{a.rung}-{nm}-s{s}"
            ck = sorted(d.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
            m = TinyLM(cfg, tok.vocab_size).to(a.device)
            m.load_state_dict(torch.load(ck, map_location=a.device)); m.eval()
            mine = []
            for it in bench:
                if it.section not in rows:
                    continue
                got = tok.decode(_greedy(m, tok, tok.encode(f"I:{it.text_in};O:"), 24, stop, a.device))
                rows[it.section].append((got, it.gold))
                if it.section == "niche_unseen":
                    mine.append((got, it.gold))
                    if len(ex) < 6 and classify(got, it.gold) != "CORRECT":
                        ex.append({"arm": f"{nm}-s{s}", "in": it.text_in, "gold": it.gold,
                                   "emitted": got, "class": classify(got, it.gold)})
            per_ckpt.append({"arm": f"{nm}-s{s}", **summarise(mine)})
    row = {"kind": "diag", "diag": "D3-trace", "rung": a.rung, "device": a.device,
           "heldout_abcd": summarise(rows["niche_unseen"]),
           "len5": summarise(rows["niche_heldrange"]),
           "per_checkpoint_abcd": per_ckpt, "examples": ex}
    Journal(ROOT / "runs" / "journal.jsonl").append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "per_checkpoint_abcd"}, indent=1))

if __name__ == "__main__":
    main()
