"""Build + freeze the pool manifest. Pool data itself is regenerable (data/,
gitignored); freeze/manifest.json pins seed, per_cell, and hash.

v3 port law: the seeds are v2's, so the pool and tokenizer hashes must land
byte-identical on v2's manifest values -- proving the generators survived the
substrate move. Any drift is a port bug, not a new dataset."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akademeia.data.pool import build_pool, save_pool
from akademeia.tokenizer import CharTokenizer
from akademeia.data.families import render

ROOT = Path(__file__).resolve().parents[1]
SEED, PER_CELL = 20260730, 20_000

def main():
    pool = build_pool(seed=SEED, per_cell=PER_CELL)
    (ROOT / "data").mkdir(exist_ok=True)
    save_pool(pool, ROOT / "data" / "pool.jsonl")
    tok = CharTokenizer.fit([render(e) for e in pool.examples])
    (ROOT / "freeze").mkdir(exist_ok=True)
    tok.save(ROOT / "freeze" / "tokenizer.json")
    manifest_path = ROOT / "freeze" / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {}
    frozen = {"pool_seed": SEED, "pool_per_cell": PER_CELL,
              "pool_hash": pool.content_hash, "tokenizer_hash": tok.hash()}
    for k, v in frozen.items():
        if k in manifest and manifest[k] != v:
            print(f"FATAL: frozen {k} would change {manifest[k]} -> {v}"); sys.exit(4)
    manifest.update(frozen)
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    print(json.dumps(frozen, indent=1))

if __name__ == "__main__":
    main()
