"""Build + freeze BOTH niche benchmarks; enforce leakage + coverage laws FIRST.
Seeds are v2's, so the bench hashes must land on v2's manifest values."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akademeia.data.bench import build_bench, save_bench, bench_hash
from akademeia.data.pool import load_pool
from akademeia.data.leakage import check_leakage
from akademeia.tokenizer import CharTokenizer
from akademeia.data.families import render, PHYSICS_TEMPLATES, NICHE_FAMILIES

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {"physics": 20260731, "vm": 20260801}

def main():
    pool = load_pool(ROOT / "data" / "pool.jsonl")
    pool_texts = [render(e) for e in pool.examples]
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    manifest_path = ROOT / "freeze" / "manifest.json"
    manifest = json.load(open(manifest_path))
    out = {}
    for niche in NICHE_FAMILIES:
        bench = build_bench(seed=SEEDS[niche], pool_texts=pool_texts, niche=niche)
        bench_texts = [f"I:{i.text_in};O:{i.gold}$" for i in bench]
        unseen = ([t.expr for t in PHYSICS_TEMPLATES if t.split == "bench"]
                  if niche == "physics" else [])   # vm: shape-hash split law
        check_leakage(pool_texts, bench_texts, unseen)
        tok.assert_covers(bench_texts)
        save_bench(bench, ROOT / "freeze" / f"bench-{niche}.jsonl")
        h = bench_hash(bench)
        key = f"bench_hash_{niche}"
        if key in manifest and manifest[key] != h:
            print(f"FATAL: {key} frozen at {manifest[key]}, refusing {h}")
            sys.exit(4)
        manifest[key] = h
        manifest[f"bench_seed_{niche}"] = SEEDS[niche]
        out[niche] = h
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
