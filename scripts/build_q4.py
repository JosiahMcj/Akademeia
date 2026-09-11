"""Q4 wiring: build the scratchpad variant pool + its frozen bench.

The scratchpad family is deliberately NOT in the frozen FAMILIES registry --
adding it there would rewrite every pinned pool hash -- so it rides in as
extra_examples on a VARIANT pool (the Study S2 precedent) with its own
manifest keys. Nothing here changes an existing frozen value; build_pool's
default path stays byte-identical and a test pins that.

Every registered law is checked HERE, at build time, and the build refuses
rather than warns:
  - zero new characters against the frozen tokenizer
  - bench chains are novel COMBINATIONS of trained symbols at trained
    lengths; extrap is the only split with a new length (split_law_ok)
  - no bench item appears in the pool, verbatim, in either direction
  - the independent grader accepts every gold trace and rejects a
    perturbed one (a grader that cannot fail is not a grader)
  - every gold trace fits the registered generation cap
"""
import json, random, sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akademeia.data.families import FAMILIES, BASIC_FAMILIES, Example, render
from akademeia.data.bench import BenchItem, bench_hash, save_bench
from akademeia.data.pool import build_pool, save_pool
from akademeia.tokenizer import CharTokenizer
from akademeia.scratchpad import (CHAIN_TEMPLATES, gen_item, gold_trace,
                                  trace_eval_independent, split_law_ok)

ROOT = Path(__file__).resolve().parents[1]
POOL_FILE = "pool_q4.jsonl"
POOL_KEY = "pool_q4_hash"
BENCH_NICHE = "scratchpad"
POOL_SEED, BENCH_SEED = 20260904, 20260905
PER_CELL = 20_000
PER_SECTION = 60
FAMILY = "scratchpad"
MAX_OUT = 24          # eval_checkpoint's generation cap, asserted against the
                      # longest gold trace at build time: a cap below a legal
                      # trace would silently turn a correct answer into a
                      # wrong grade, so the build refuses rather than warns

def chain_examples(seed: int, per_cell: int) -> list:
    """Train-split chains only. Bench and extrap templates are absent BY
    CONSTRUCTION, not by filtering."""
    r = random.Random(seed)
    out = []
    trains = [t for t in CHAIN_TEMPLATES if t.split == "train"]
    for d in (1, 2, 3):
        made = 0
        while made < per_cell:
            t = trains[made % len(trains)]
            text_in, text_out = gen_item(t, r, d)
            out.append(Example(FAMILY, d, t.template_id, text_in, text_out))
            made += 1
    return out

def build_q4_bench(seed: int, pool_texts, per_section: int) -> list:
    r = random.Random(seed)
    pool_set = set(pool_texts)
    items, seen = [], set()
    def add(section, family, template_id, difficulty, text_in, gold):
        if text_in in seen:
            return False
        if section == "basics" and f"I:{text_in};O:{gold}$" in pool_set:
            return False
        seen.add(text_in)
        items.append(BenchItem(section, family, template_id, difficulty,
                               text_in, gold))
        return True
    bench_ts = [t for t in CHAIN_TEMPLATES if t.split == "bench"]
    extrap_ts = [t for t in CHAIN_TEMPLATES if t.split == "extrap"]
    # "niche_heldrange" is the machinery's reported-never-gated diagnostic
    # slot; the registration names that precedent for the length-4 chains, so
    # they ride it rather than growing a parallel section the arm files and
    # gates would not know about.
    for section, ts in (("niche_unseen", bench_ts),
                        ("niche_heldrange", extrap_ts)):
        n = 0
        while n < per_section:
            t = ts[n % len(ts)]
            text_in, gold = gen_item(t, r, r.choice([1, 2, 3]))
            n += add(section, FAMILY, t.template_id, 2, text_in, gold)
    n = 0
    while n < per_section:
        f = FAMILIES[r.choice(BASIC_FAMILIES)]
        ex = f.gen(r, r.choice([2, 3]))
        n += add("basics", ex.family, ex.template_id, ex.difficulty,
                 ex.text_in, ex.text_out)
    return items

def main():
    if not split_law_ok():
        print("FATAL: registered split law fails"); sys.exit(4)
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    extras = chain_examples(POOL_SEED, PER_CELL)
    pool = build_pool(seed=POOL_SEED, per_cell=PER_CELL, extra_examples=extras)
    pool_texts = [render(e) for e in pool.examples]
    tok.assert_covers(pool_texts)              # zero-new-characters law
    bench = build_q4_bench(BENCH_SEED, pool_texts, PER_SECTION)
    tok.assert_covers([f"I:{i.text_in};O:{i.gold}$" for i in bench])
    # leakage, both directions, on the exact rendered strings
    pool_set = set(pool_texts)
    leaked = [i for i in bench if f"I:{i.text_in};O:{i.gold}$" in pool_set]
    if leaked:
        print(f"FATAL: {len(leaked)} bench items appear in the pool"); sys.exit(4)
    pool_ins = {e.text_in for e in pool.examples}
    leaked_in = [i for i in bench if i.section != "basics" and i.text_in in pool_ins]
    if leaked_in:
        print(f"FATAL: {len(leaked_in)} bench prompts appear in the pool"); sys.exit(4)
    # the grader must accept every gold trace AND reject a perturbation
    chains = [i for i in bench if i.section != "basics"]
    for i in chains:
        if not trace_eval_independent(i.text_in, i.gold):
            print(f"FATAL: grader rejects its own gold: {i.text_in}"); sys.exit(4)
        wrong = ("9" + i.gold) if not i.gold.startswith("9") else i.gold[1:]
        if trace_eval_independent(i.text_in, wrong):
            print(f"FATAL: grader accepts a wrong trace: {i.text_in}"); sys.exit(4)
    longest = max(len(i.gold) for i in chains)
    if longest + 1 > MAX_OUT:
        print(f"FATAL: longest gold trace {longest} exceeds cap {MAX_OUT}")
        sys.exit(4)
    (ROOT / "data").mkdir(exist_ok=True)
    save_pool(pool, ROOT / "data" / POOL_FILE)
    save_bench(bench, ROOT / "freeze" / f"bench-{BENCH_NICHE}.jsonl")
    mpath = ROOT / "freeze" / "manifest.json"
    m = json.load(open(mpath))
    frozen = {POOL_KEY: pool.content_hash,
              f"bench_hash_{BENCH_NICHE}": bench_hash(bench),
              "pool_q4_seed": POOL_SEED, "bench_q4_seed": BENCH_SEED,
              "q4_max_out": MAX_OUT}
    for k, v in frozen.items():
        if k in m and m[k] != v:
            print(f"FATAL: frozen {k} would change {m[k]} -> {v}"); sys.exit(4)
    m.update(frozen)
    json.dump(m, open(mpath, "w"), indent=1)
    counts = {}
    for i in bench:
        counts[i.section] = counts.get(i.section, 0) + 1
    print(json.dumps({"pool_examples": len(pool.examples),
                      "scratchpad_examples": len(extras),
                      "bench_sections": counts,
                      "longest_gold_trace": longest, **frozen}, indent=1))

if __name__ == "__main__":
    main()
