"""D3-ops (registered 2026-10-30): for every two-partial output on the held-out
(a,b,c,d) bench, what does step 2 equal?"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from akademeia.data.bench import load_bench
from akademeia.evalrunner import _greedy
from akademeia.journal import Journal
from akademeia.lengthdiag import parse_trace
from akademeia.model import TinyLM
from akademeia.tiers import RUNGS
from akademeia.tokenizer import CharTokenizer
ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rung", type=int, required=True)
    ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    cfg = RUNGS[a.rung]; tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    items = [b for b in load_bench(ROOT / "freeze" / "bench-d3.jsonl") if b.section == "niche_unseen"]
    stop = tok.encode("$")[0]
    t = {"two_step": 0, "step1_is_ab": 0, "step2_is_x_c": 0, "step2_is_x_d": 0,
         "step2_other": 0, "both_c_and_d": 0}
    for nm in ("A", "Aprime"):
        for s in (11, 22, 33):
            d = ROOT / "runs" / f"d3-r{a.rung}-{nm}-s{s}"
            ck = sorted(d.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
            m = TinyLM(cfg, tok.vocab_size).to(a.device)
            m.load_state_dict(torch.load(ck, map_location=a.device)); m.eval()
            for it in items:
                e = parse_trace(tok.decode(_greedy(m, tok, tok.encode(f"I:{it.text_in};O:"), 24, stop, a.device)))
                if not e or len(e) != 2:
                    continue
                v = {k: int(x) for k, x in (p.split("=") for p in it.text_in.split(";")[1:])}
                t["two_step"] += 1
                t["step1_is_ab"] += e[0] == v["a"] * v["b"]
                isc, isd = e[1] == e[0] * v["c"], e[1] == e[0] * v["d"]
                if isc and isd: t["both_c_and_d"] += 1
                elif isc: t["step2_is_x_c"] += 1
                elif isd: t["step2_is_x_d"] += 1
                else: t["step2_other"] += 1
    n = t["two_step"] or 1
    frac = {k: round(v / n, 4) for k, v in t.items() if k != "two_step"}
    lab = ("PREFIX_abc" if frac["step2_is_x_c"] >= 0.6 else
           "PAIR_THEN_LAST" if frac["step2_is_x_d"] >= 0.6 else "NOT_IDENTIFIED")
    row = {"kind": "diag", "diag": "D3-ops", "rung": a.rung, "device": a.device,
           "counts": t, "fractions": frac, "registered_reading": lab}
    Journal(ROOT / "runs" / "journal.jsonl").append(row); print(json.dumps(row, indent=1))

if __name__ == "__main__":
    main()
