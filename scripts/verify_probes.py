"""Verify + pin the frozen dev-probe set.

The probes are a frozen instrument carried from v2, like the benches. Their
generative seed predates later family-set changes, so REGENERATION IS NOT
EXPECTED TO REPRODUCE THEM -- the laws are what carries: at port time this
script re-verifies, against the rebuilt pool and both benches, that the set
is count-exact, uses no bench-split templates, and overlaps nothing. Verified
facts get pinned in the manifest; any drift is a hard exit."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akademeia.probes import load_probes, assert_probe_integrity, probes_hash
from akademeia.data.pool import load_pool
from akademeia.data.bench import load_bench
from akademeia.data.families import render, NICHE_FAMILIES

ROOT = Path(__file__).resolve().parents[1]
PER_FAMILY = 8

def main():
    items = load_probes(ROOT / "freeze" / "probes.jsonl", PER_FAMILY)
    pool_texts = [render(e) for e in load_pool(ROOT / "data" / "pool.jsonl").examples]
    bench_texts = []
    for niche in NICHE_FAMILIES:
        bench = load_bench(ROOT / "freeze" / f"bench-{niche}.jsonl")
        bench_texts += [f"I:{i.text_in};O:{i.gold}$" for i in bench]
    assert_probe_integrity(items, PER_FAMILY, pool_texts=pool_texts,
                           bench_texts=bench_texts)
    manifest_path = ROOT / "freeze" / "manifest.json"
    manifest = json.load(open(manifest_path))
    frozen = {"probes_per_family": PER_FAMILY, "probes_hash": probes_hash(items),
              "probes_provenance": "frozen v2 instrument; disjointness "
                                   "re-verified against the rebuilt pool+benches"}
    for k, v in frozen.items():
        if k in manifest and manifest[k] != v:
            print(f"FATAL: frozen {k} would change {manifest[k]} -> {v}")
            sys.exit(4)
    manifest.update(frozen)
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    print(json.dumps(frozen, indent=1))

if __name__ == "__main__":
    main()
