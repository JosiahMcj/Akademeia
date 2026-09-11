"""Dev probes for Phase B learning-progress allocation.

The frozen bench is NEVER consulted mid-run (selection-on-test law). Probes are
a separate, frozen, in-distribution generalization set:

  - TRAIN-split templates/shapes only, TRAIN operand ranges only, so probes
    never overlap the bench's held-out dimensions (unseen templates, held
    ranges). Enforced at build time and asserted at load (group-level law).
  - Item-disjoint from the pool, so a probe score is generalization, not recall.
  - Item-disjoint from every niche bench, so steering can never be
    selection-on-test.

Any violation hard-exits. Counts are asserted exactly at load: a silently
shrunken eval set has burned this lab before.
"""
from dataclasses import dataclass, asdict
import json, sys
from pathlib import Path
import random

from akademeia.config import content_hash
from akademeia.data.families import (FAMILIES, BASIC_FAMILIES, NICHE_FAMILIES,
                                    PHYSICS_TEMPLATES, render)

PER_FAMILY = 8

@dataclass(frozen=True)
class ProbeItem:
    family: str
    difficulty: int
    template_id: str
    text_in: str
    gold: str

def _bench_split_ids() -> set:
    """Template/shape ids the BENCH owns. Probes must never use them."""
    ids = {t.template_id for t in PHYSICS_TEMPLATES if t.split == "bench"}
    return ids

def build_probes(seed: int, per_family: int = PER_FAMILY,
                 forbidden_texts=()) -> list:
    """Generate probes, rejection-sampling against forbidden texts (pool +
    every bench). Niche probes use split='train' only."""
    r = random.Random(seed)
    forbidden = set(forbidden_texts)
    items, seen = [], set()
    for name in sorted(FAMILIES):
        fam = FAMILIES[name]
        made = 0
        tries = 0
        while made < per_family:
            tries += 1
            if tries > per_family * 5000:
                print(f"FATAL: cannot build {per_family} novel probes for {name}",
                      file=sys.stderr)
                raise SystemExit(9)
            d = r.choice([1, 2, 3])
            ex = fam.gen(r, d, split="train") if name in NICHE_FAMILIES \
                else fam.gen(r, d)
            if ex.text_in in seen or render(ex) in forbidden:
                continue
            seen.add(ex.text_in)
            items.append(ProbeItem(ex.family, ex.difficulty, ex.template_id,
                                   ex.text_in, ex.text_out))
            made += 1
    return items

def assert_probe_integrity(items, per_family: int, pool_texts=(),
                           bench_texts=()) -> None:
    """Hard-exit on any count, split, or overlap violation."""
    expected = per_family * len(FAMILIES)
    if len(items) != expected:
        print(f"FATAL: probe count {len(items)} != expected {expected}",
              file=sys.stderr)
        raise SystemExit(9)
    counts = {}
    for it in items:
        counts[it.family] = counts.get(it.family, 0) + 1
    bad = {f: n for f, n in counts.items() if n != per_family}
    if bad or set(counts) != set(FAMILIES):
        print(f"FATAL: per-family probe counts wrong: {bad or counts}",
              file=sys.stderr)
        raise SystemExit(9)
    owned = _bench_split_ids()
    stolen = [it.template_id for it in items if it.template_id in owned]
    if stolen:
        print(f"FATAL: probes use bench-split templates: {sorted(set(stolen))}",
              file=sys.stderr)
        raise SystemExit(9)
    forbidden = set(pool_texts) | set(bench_texts)
    if forbidden:
        overlap = [f"I:{it.text_in};O:{it.gold}$" for it in items
                   if f"I:{it.text_in};O:{it.gold}$" in forbidden]
        if overlap:
            print(f"FATAL: {len(overlap)} probes overlap pool/bench, "
                  f"e.g. {overlap[:2]}", file=sys.stderr)
            raise SystemExit(9)

def probes_hash(items) -> str:
    return content_hash([asdict(i) for i in items])

def save_probes(items, path) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(asdict(it)) + "\n")

def load_probes(path, per_family: int = PER_FAMILY) -> list:
    items = [ProbeItem(**json.loads(l)) for l in open(path)]
    assert_probe_integrity(items, per_family)   # count/split law at LOAD
    return items
