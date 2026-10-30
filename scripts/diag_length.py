"""D2 runner: capture what a scratchpad model EMITS, and classify how it fails.

Registered 2026-09-25. Descriptive; gates nothing. Trains one Q4 model (or
reuses a checkpoint), generates greedily on the frozen scratchpad bench, and
classifies every emitted trace under the taxonomy fixed in PROTOCOL before any
trace existed.

Both bench sections are scored, because the control is the point: the SAME
model on the SAME format at the trained depth (niche_unseen, 3 variables,
2 partials) versus one step deeper (niche_heldrange, 4 variables, 3 partials).
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from akademeia.data.bench import load_bench
from akademeia.journal import Journal
from akademeia.lengthdiag import classify, summarise
from akademeia.evalrunner import _greedy
from akademeia.model import TinyLM
from akademeia.tokenizer import CharTokenizer
from akademeia.tiers import RUNGS
from akademeia.data.pool import load_pool
from akademeia.trainer import train_run
from akademeia.curriculum import validate
from study_q4 import q4_curriculum, POOL_FILE, POOL_KEY, NICHE

ROOT = Path(__file__).resolve().parents[1]

def latest_ckpt(d: Path):
    cks = sorted(d.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return cks[-1] if cks else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=1, choices=sorted(RUNGS))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-out", type=int, default=24)
    args = ap.parse_args()

    cfg = RUNGS[args.rung]
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    out_dir = ROOT / "runs" / f"d2-r{args.rung}-s{args.seed}"
    ck = latest_ckpt(out_dir)
    if ck is None:
        pool = load_pool(ROOT / "data" / POOL_FILE)
        c = q4_curriculum(cfg.total_tokens)
        validate(c, cfg.total_tokens, extra_families=(NICHE,))
        print(f"[D2] training rung {args.rung} seed {args.seed} on {args.device}")
        t0 = time.monotonic()
        res = train_run(cfg, c, pool, tok, seed=args.seed, out_dir=out_dir,
                        device=args.device)
        print(f"[D2] trained in {time.monotonic()-t0:.0f}s")
        ck = res.checkpoints[-1][1]

    model = TinyLM(cfg, tok.vocab_size).to(args.device)
    model.load_state_dict(torch.load(ck, map_location=args.device))
    model.eval()
    stop = tok.encode("$")[0]
    bench = load_bench(ROOT / "freeze" / f"bench-{NICHE}.jsonl")

    sections = {}
    examples = {}
    for sec in ("niche_unseen", "niche_heldrange"):
        rows = []
        for it in [b for b in bench if b.section == sec]:
            got = tok.decode(_greedy(model, tok, tok.encode(f"I:{it.text_in};O:"),
                                     args.max_out, stop, args.device))
            rows.append((got, it.gold))
            if len(examples.setdefault(sec, [])) < 5:
                examples[sec].append({"in": it.text_in, "gold": it.gold,
                                      "emitted": got,
                                      "class": classify(got, it.gold)})
        sections[sec] = summarise(rows)

    row = {"kind": "diag", "diag": "D2-length-mechanism", "rung": args.rung,
           "seed": args.seed, "device": args.device,
           "trained_depth_control": sections["niche_unseen"],
           "one_step_deeper": sections["niche_heldrange"],
           "examples": examples,
           "scope": ("descriptive; gates nothing and re-adjudicates no "
                     "registered verdict")}
    Journal(ROOT / "runs" / "journal.jsonl").append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "examples"},
                     indent=1, default=str))
    print("\nexamples, one step deeper than training:")
    for e in examples.get("niche_heldrange", []):
        print(f"  in={e['in']}")
        print(f"    gold={e['gold']!r}  emitted={e['emitted']!r}  -> {e['class']}")

if __name__ == "__main__":
    main()
