"""D3 (registered 2026-10-29): build the length-exposure variant pool + bench.

Training chains include length 4; the held-out bench is (a,b,c,d) -- the exact
template that scored 0.000 in 36/36 Q4 runs -- now a novel combination at a
TRAINED length. A length-5 chain is the never-gated diagnostic. Every law is
checked here and the build refuses rather than warns.
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akademeia.data.families import FAMILIES, BASIC_FAMILIES, Example, render
from akademeia.data.bench import BenchItem, bench_hash, save_bench
from akademeia.data.pool import build_pool, save_pool
from akademeia.tokenizer import CharTokenizer
from akademeia.scratchpad import ChainTemplate, gen_item, trace_eval_independent

ROOT = Path(__file__).resolve().parents[1]
NICHE, FAMILY = "d3", "scratchpad"
POOL_FILE, POOL_KEY = "pool_d3.jsonl", "pool_d3_hash"
POOL_SEED, BENCH_SEED, PER_CELL, PER_SECTION, MAX_OUT = 20261029, 20261030, 20_000, 60, 24

TEMPLATES = (
    ChainTemplate("sp-ab", ("a", "b"), "train"),
    ChainTemplate("sp-abc", ("a", "b", "c"), "train"),
    ChainTemplate("sp-mnc", ("m", "n", "c"), "train"),
    ChainTemplate("sp-bcd", ("b", "c", "d"), "train"),
    ChainTemplate("sp-mncd", ("m", "n", "c", "d"), "train"),
    ChainTemplate("sp-abmn", ("a", "b", "m", "n"), "train"),
    ChainTemplate("sp-abcd", ("a", "b", "c", "d"), "bench"),
    ChainTemplate("sp-abcdm", ("a", "b", "c", "d", "m"), "extrap"),
)

def split_law_ok(ts) -> bool:
    tr = [t for t in ts if t.split == "train"]
    syms = {s for t in tr for s in t.variables}
    lens = {len(t.variables) for t in tr}
    tups = {t.variables for t in tr}
    for t in ts:
        if t.split == "bench" and (set(t.variables) - syms
                                   or len(t.variables) not in lens
                                   or t.variables in tups):
            return False
        if t.split == "extrap" and (set(t.variables) - syms
                                    or len(t.variables) in lens):
            return False
    return True

def main():
    assert split_law_ok(TEMPLATES), "split law fails"
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    r = random.Random(POOL_SEED)
    trains = [t for t in TEMPLATES if t.split == "train"]
    extras = []
    for d in (1, 2, 3):
        for i in range(PER_CELL):
            t = trains[i % len(trains)]
            ti, to = gen_item(t, r, d)
            extras.append(Example(FAMILY, d, t.template_id, ti, to))
    pool = build_pool(seed=POOL_SEED, per_cell=PER_CELL, extra_examples=extras)
    texts = [render(e) for e in pool.examples]
    tok.assert_covers(texts)
    pset, pins = set(texts), {e.text_in for e in pool.examples}
    rb = random.Random(BENCH_SEED)
    items, seen = [], set()
    def add(sec, fam, tid, d, ti, g):
        if ti in seen or ti in pins or f"I:{ti};O:{g}$" in pset:
            return False
        seen.add(ti); items.append(BenchItem(sec, fam, tid, d, ti, g)); return True
    bench_t = next(t for t in TEMPLATES if t.split == "bench")
    ext_t = next(t for t in TEMPLATES if t.split == "extrap")
    n = 0
    while n < PER_SECTION:
        ti, g = gen_item(bench_t, rb, rb.choice([1, 2, 3])); n += add("niche_unseen", FAMILY, bench_t.template_id, 2, ti, g)
    n = 0
    while n < PER_SECTION:
        ti, g = gen_item(ext_t, rb, 1); n += add("niche_heldrange", FAMILY, ext_t.template_id, 1, ti, g)
    n = 0
    while n < PER_SECTION:
        ex = FAMILIES[rb.choice(BASIC_FAMILIES)].gen(rb, rb.choice([2, 3]))
        n += add("basics", ex.family, ex.template_id, ex.difficulty, ex.text_in, ex.text_out)
    tok.assert_covers([f"I:{i.text_in};O:{i.gold}$" for i in items])
    chains = [i for i in items if i.section != "basics"]
    for i in chains:
        assert trace_eval_independent(i.text_in, i.gold), i
        assert not trace_eval_independent(i.text_in, i.gold + "0"), i
    longest = max(len(i.gold) for i in chains)
    assert longest + 1 <= MAX_OUT, f"trace {longest} exceeds cap"
    ids = {e.template_id for e in pool.examples if e.family == FAMILY}
    assert "sp-abcd" not in ids and "sp-abcdm" not in ids, "held-out leaked into pool"
    (ROOT / "data").mkdir(exist_ok=True)
    save_pool(pool, ROOT / "data" / POOL_FILE)
    save_bench(items, ROOT / "freeze" / f"bench-{NICHE}.jsonl")
    mp = ROOT / "freeze" / "manifest.json"; m = json.load(open(mp))
    fz = {POOL_KEY: pool.content_hash, f"bench_hash_{NICHE}": bench_hash(items)}
    for k, v in fz.items():
        if k in m and m[k] != v:
            sys.exit(f"FATAL: frozen {k} would change")
    m.update(fz); json.dump(m, open(mp, "w"), indent=1)
    print(json.dumps({**fz, "longest_trace": longest, "pool_examples": len(pool.examples)}))

if __name__ == "__main__":
    main()
