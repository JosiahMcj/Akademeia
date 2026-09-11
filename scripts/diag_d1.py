"""Diagnostic D1 at a rung: is the composition wall about GENERALIZATION or
DEPTH? Zero-training: evaluates the rung's stored direct50 final checkpoints
per physics template, in-distribution, grouped by (split, n_vars, n_ops).
Journaled as kind=diag; descriptive, gates nothing.

The TRAIN pool contains 3-var templates (ph-accel, ph-heat) and a 2-var/3-op
template (ph-ke), so trained-3var accuracy is measurable with no new
training."""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from akademeia.model import TinyLM
from akademeia.tokenizer import CharTokenizer
from akademeia.evalrunner import _greedy
from akademeia.data.families import (PHYSICS_TEMPLATES, _PMAG,
                                     physics_eval_independent)
from akademeia.journal import Journal
from akademeia.ops import device_lock, wait_for_load
from akademeia.tiers import RUNGS
from run_arm import SEEDS

ROOT = Path(__file__).resolve().parents[1]
N_ITEMS = 20
DIFF = 2

def n_ops(expr: str) -> int:
    return sum(expr.count(c) for c in "*/+-")

def make_items(t, r: random.Random, n: int):
    lo, hi = _PMAG[DIFF]
    items = []
    while len(items) < n:
        vals = {v: r.randint(lo, hi) for v in t.variables}
        text = f"{t.expr};" + ";".join(f"{v}={vals[v]}" for v in t.variables)
        gold = physics_eval_independent(text, require_exact=True)
        if gold is None:
            continue
        items.append((text, str(gold)))
    return items

def latest_ckpt(d: Path) -> Path:
    return sorted(d.glob("ckpt_*.pt"),
                  key=lambda p: int(p.stem.split("_")[1]))[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, required=True, choices=sorted(RUNGS))
    ap.add_argument("--tag-prefix", default=None,
                    help="checkpoint dirs to read; default ladder-r{rung}-direct50-A")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = RUNGS[args.rung]
    prefix = args.tag_prefix or f"ladder-r{args.rung}-direct50-A"
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    stop = tok.encode("$")[0]
    r = random.Random(20260805)          # the v2 item seed, for comparability
    per_template = {t.template_id: make_items(t, r, N_ITEMS)
                    for t in PHYSICS_TEMPLATES}
    dirs = [ROOT / "runs" / f"{prefix}-s{s}" for s in SEEDS]
    wait_for_load()
    acc = {t.template_id: [] for t in PHYSICS_TEMPLATES}
    with device_lock(args.device):
        for d in dirs:
            model = TinyLM(cfg, tok.vocab_size).to(args.device)
            model.load_state_dict(torch.load(latest_ckpt(d),
                                             map_location=args.device))
            model.eval()
            for t in PHYSICS_TEMPLATES:
                hit = 0
                for text, gold in per_template[t.template_id]:
                    pred = tok.decode(_greedy(model, tok,
                                              tok.encode(f"I:{text};O:"),
                                              24, stop, args.device))
                    hit += (pred == gold)
                acc[t.template_id].append(hit / N_ITEMS)
    out = {k: round(sum(v) / len(v), 3) for k, v in acc.items()}
    groups = {}
    for t in PHYSICS_TEMPLATES:
        key = (t.split, len(t.variables) >= 3, n_ops(t.expr) >= 2)
        groups.setdefault(key, []).append(t.template_id)
    print(f"\nper-template mean accuracy (rung {args.rung}, in-distribution, "
          f"d{DIFF}, n={N_ITEMS} x {len(SEEDS)} seeds):")
    for t in PHYSICS_TEMPLATES:
        print(f"  {t.template_id:14s} {t.split:5s} vars={len(t.variables)} "
              f"ops={n_ops(t.expr)}  {out[t.template_id]:.2f}")
    summary = {}
    for (split, three_var, multi_op), ids in sorted(groups.items()):
        m = sum(out[i] for i in ids) / len(ids)
        summary[f"{split}:3var={three_var}:multiop={multi_op}"] = round(m, 3)
    print("\ngroup means:", json.dumps(summary, indent=1))
    Journal(ROOT / "runs" / "journal.jsonl").append(
        {"kind": "diag", "diag": "D1-3var-wall", "rung": args.rung,
         "tag_prefix": prefix, "device": args.device, "per_template": out,
         "groups": summary, "n_items": N_ITEMS, "difficulty": DIFF,
         "note": "in-distribution per-template accuracy on stored "
                 "checkpoints; descriptive, gates nothing"})

if __name__ == "__main__":
    main()
