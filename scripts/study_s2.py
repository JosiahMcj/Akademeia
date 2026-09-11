"""Study Q2 at a rung: does product-chain template VARIETY unlock unseen
product-chain formulas, and does the answer move with scale? Descriptive; no
gates; the frozen physics bench is the measuring stick and is never rebuilt.

Stages:
  build    the S2 pool variant + laws (rung-independent, built once; the v2
           artifact seed, so the hash must land on v2's manifest value)
  runs     direct50-physics A/A' on the S2 pool at --rung (frozen-bench eval)
  verdict  registered labels vs that rung's stored direct50 baseline, plus
           the in-distribution trained-product probe

Adding physics templates shifts the shared RNG stream for every family
generated after physics, so basics items differ from the original pool and
could collide verbatim with bench items -- bench texts are therefore excluded
BY CONSTRUCTION (forbidden_texts), the only design that works against
high-coverage family cells (measured in v2)."""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from akademeia.data.pool import build_pool, save_pool
from akademeia.data.bench import load_bench
from akademeia.data.leakage import check_leakage
from akademeia.data.families import (render, PHYSICS_TEMPLATES,
                                     S2_EXTRA_TEMPLATES)
from akademeia.tokenizer import CharTokenizer
from akademeia.model import TinyLM
from akademeia.evalrunner import _greedy
from akademeia.curriculum import validate
from akademeia.journal import Journal
from akademeia.ops import device_lock, wait_for_load
from akademeia.tiers import RUNGS
from run_arm import run_curriculum
from ladder import direct50
from diag_d1 import make_items, latest_ckpt

ROOT = Path(__file__).resolve().parents[1]
POOL_FILE = "pool-s2.jsonl"
POOL_KEY = "pool_s2_hash"
SEED = 20260810                    # the v2 S2 artifact seed
V2_POOL_S2_HASH = "f099af78e665050f"
PRODUCT_BENCH_IDS = {"ph-pe", "ph-gravity"}
SEEDS = (11, 22, 33)

def cmd_build():
    bench = load_bench(ROOT / "freeze" / "bench-physics.jsonl")
    bench_texts = [f"I:{i.text_in};O:{i.gold}$" for i in bench]
    bench_vm = load_bench(ROOT / "freeze" / "bench-vm.jsonl")
    forbidden = frozenset(bench_texts) | frozenset(
        f"I:{i.text_in};O:{i.gold}$" for i in bench_vm)
    unseen_exprs = [t.expr for t in PHYSICS_TEMPLATES if t.split == "bench"]
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    mpath = ROOT / "freeze" / "manifest.json"
    m = json.load(open(mpath))
    pool = build_pool(seed=SEED, per_cell=20_000,
                      physics_extra=S2_EXTRA_TEMPLATES,
                      forbidden_texts=forbidden)
    pool_texts = [render(e) for e in pool.examples]
    check_leakage(pool_texts, bench_texts, unseen_exprs)  # verification, not hope
    tok.assert_covers(pool_texts)   # zero-new-chars law
    if pool.content_hash != V2_POOL_S2_HASH:
        print(f"FATAL: S2 pool drifted from the v2 artifact: "
              f"{pool.content_hash} != {V2_POOL_S2_HASH} (port bug)")
        sys.exit(4)
    if POOL_KEY in m and m[POOL_KEY] != pool.content_hash:
        print(f"FATAL: {POOL_KEY} frozen at {m[POOL_KEY]}"); sys.exit(4)
    (ROOT / "data").mkdir(exist_ok=True)
    save_pool(pool, ROOT / "data" / POOL_FILE)
    m[POOL_KEY] = pool.content_hash
    m["pool_s2_seed"] = SEED
    json.dump(m, open(mpath, "w"), indent=1)
    Journal(ROOT / "runs" / "journal.jsonl").append(
        {"kind": "artifact", "study": "S2", "pool_seed": SEED,
         "pool_hash": pool.content_hash,
         "note": "bench items excluded by construction; hash matches v2"})
    print(f"[S2] pool-s2 frozen: {pool.content_hash} seed={SEED} "
          f"({len(pool.examples)} examples) -- matches v2")

def cmd_runs(rung: int, device: str):
    cfg = RUNGS[rung]
    c = direct50("physics", cfg.total_tokens)
    validate(c, cfg.total_tokens)
    run_curriculum(cfg, c, "physics", tag=f"s2-r{rung}-direct50-A",
                   stream_salt=0, phase="S2",
                   pool_file=POOL_FILE, pool_key=POOL_KEY, device=device)
    run_curriculum(cfg, c, "physics", tag=f"s2-r{rung}-direct50-Aprime",
                   stream_salt=1, phase="S2",
                   pool_file=POOL_FILE, pool_key=POOL_KEY, device=device)

def _product_split(arm: dict) -> dict:
    """Per-seed accuracy on unseen PRODUCT vs NON-PRODUCT bench items."""
    bench = load_bench(ROOT / "freeze" / "bench-physics.jsonl")
    unseen = [b for b in bench if b.section == "niche_unseen"]
    out = {"product": {}, "other": {}}
    for seed, items in arm["final_items"].items():
        assert len(items) == len(unseen), "final_items / bench misalignment"
        prod = [ok for ok, b in zip(items, unseen)
                if b.template_id in PRODUCT_BENCH_IDS]
        rest = [ok for ok, b in zip(items, unseen)
                if b.template_id not in PRODUCT_BENCH_IDS]
        out["product"][seed] = sum(prod) / len(prod)
        out["other"][seed] = sum(rest) / len(rest)
    return out

def cmd_verdict(rung: int, device: str):
    journal = Journal(ROOT / "runs" / "journal.jsonl")
    a = json.loads((ROOT / "runs" / f"arm-s2-r{rung}-direct50-A.json").read_text())
    ap = json.loads(
        (ROOT / "runs" / f"arm-s2-r{rung}-direct50-Aprime.json").read_text())
    base = json.loads(
        (ROOT / "runs" / f"arm-ladder-r{rung}-direct50-A.json").read_text())
    s2, s2p, b = _product_split(a), _product_split(ap), _product_split(base)
    seeds = sorted(s2["product"])
    n = len(seeds)
    s2_mean = sum(s2["product"].values()) / n
    base_mean = sum(b["product"].values()) / n
    band = max(abs(s2["product"][s] - s2p["product"][s]) for s in seeds)
    margin = 3 * max(band, 0.05)
    diff = s2_mean - base_mean
    # in-distribution probe of trained product templates, new and old
    cfg = RUNGS[rung]
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    stop = tok.encode("$")[0]
    r = random.Random(20260810)
    probe_templates = ([t for t in PHYSICS_TEMPLATES
                        if t.template_id in ("ph-heat", "ph-ke")]
                       + list(S2_EXTRA_TEMPLATES))
    items = {t.template_id: make_items(t, r, 20) for t in probe_templates}
    trained = {}
    wait_for_load()
    with device_lock(device):
        for seed in SEEDS:
            model = TinyLM(cfg, tok.vocab_size).to(device)
            model.load_state_dict(torch.load(
                latest_ckpt(ROOT / "runs" / f"s2-r{rung}-direct50-A-s{seed}"),
                map_location=device))
            model.eval()
            for tid, its in items.items():
                hit = sum(tok.decode(_greedy(model, tok,
                                             tok.encode(f"I:{t};O:"), 24, stop,
                                             device)) == g
                          for t, g in its)
                trained.setdefault(tid, []).append(hit / len(its))
    trained_mean = {k: round(sum(v) / len(v), 3) for k, v in trained.items()}
    new_mean = (sum(trained_mean[t.template_id] for t in S2_EXTRA_TEMPLATES)
                / len(S2_EXTRA_TEMPLATES))
    if diff > margin:
        label = "ABSTRACTION-LIMITED (variety unlocked unseen products)"
    elif new_mean >= 0.2:
        label = "DEPTH-LIMITED (variety trained but did not transfer)"
    else:
        label = "DILUTED (variety failed to train at all)"
    row = {"kind": "study", "study": "S2-Q2", "rung": rung, "device": device,
           "s2_unseen_product": {k: round(v, 3) for k, v in s2["product"].items()},
           "s2_unseen_other_mean": round(sum(s2["other"].values()) / n, 3),
           "base_unseen_product_mean": round(base_mean, 3),
           "s2_unseen_product_mean": round(s2_mean, 3),
           "diff": round(diff, 4), "band": round(band, 4),
           "margin": round(margin, 4),
           "trained_product_in_distribution": trained_mean,
           "new_templates_mean": round(new_mean, 3), "label": label}
    journal.append(row)
    print(json.dumps(row, indent=1))
    print(f"[S2 r{rung}] VERDICT: {label}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "runs", "verdict"])
    ap.add_argument("--rung", type=int, choices=sorted(RUNGS))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.stage == "build":
        cmd_build()
    elif args.rung is None:
        print(f"{args.stage} needs --rung"); sys.exit(2)
    elif args.stage == "runs":
        cmd_runs(args.rung, args.device)
    else:
        cmd_verdict(args.rung, args.device)

if __name__ == "__main__":
    main()
