"""Frozen benchmark (spec section 5): held-out TEMPLATES (group key) +
held-out RANGES + basics retention. Committed to freeze/, content-hashed."""
from dataclasses import dataclass, asdict
import json, random
from akademeia.config import content_hash
from akademeia.data.families import (FAMILIES, BASIC_FAMILIES, PHYSICS_TEMPLATES,
                                   _physics_gen, _vm_gen, Example)

@dataclass(frozen=True)
class BenchItem:
    section: str
    family: str
    template_id: str
    difficulty: int
    text_in: str
    gold: str

# Strictly above each niche's own train cap: physics trains to 40, vm to 20.
_HELD_RANGE = {"physics": (41, 80), "vm": (21, 40)}

def _heldrange_example(r: random.Random, niche: str = "physics",
                       difficulty: int = 2) -> Example:
    """Train-split templates/shapes, operands drawn from the held-out range."""
    import akademeia.data.families as fam
    band = _HELD_RANGE[niche]
    override = {1: band, 2: band, 3: band}
    if niche == "physics":
        saved = fam._PMAG
        try:
            fam._PMAG = override
            # exclude_division: see _physics_gen -- exact division inside a
            # narrow band degenerates to the constant 1
            return fam._physics_gen(r, difficulty, split="train",
                                    exclude_division=True)
        finally:
            fam._PMAG = saved
    saved = fam._VMMAG
    try:
        fam._VMMAG = override
        return fam._vm_gen(r, difficulty, split="train")
    finally:
        fam._VMMAG = saved

def build_bench(seed: int, per_section: int = 60, pool_texts=(),
                niche: str = "physics") -> list:
    """PROTOCOL Amendment 1: basics items come from difficulty 2-3 only and are
    rejection-sampled to full-item novelty against the pool. Niche novelty is
    by construction (held-out templates/shapes / held-out ranges)."""
    from akademeia.data.families import render
    gen = _physics_gen if niche == "physics" else _vm_gen
    r = random.Random(seed)
    pool_set = set(pool_texts)
    items, seen = [], set()
    def add(section, ex):
        if ex.text_in in seen: return False
        if section == "basics" and render(ex) in pool_set: return False
        seen.add(ex.text_in)
        items.append(BenchItem(section, ex.family, ex.template_id, ex.difficulty,
                               ex.text_in, ex.text_out))
        return True
    # Amendment 4: vm's split is now stratified by complexity, so held-out
    # shapes exist at EVERY difficulty and unseen spans d1-3 for both niches.
    # heldrange difficulty is per-niche: physics keeps the d2 design that
    # measured 0.367; vm spans d1-3 because its d2+ chains measured 0.000.
    heldrange_ds = [2] if niche == "physics" else [1, 2, 3]
    # vm holds out COMBINATIONS only, which exist at d2-3; its d1 stratum is
    # all primitives and is deliberately fully trainable (Amendment 5).
    unseen_ds = [1, 2, 3] if niche == "physics" else [2, 3]
    n = 0
    while n < per_section:
        n += add("niche_unseen", gen(r, r.choice(unseen_ds), split="bench"))
    n = 0
    while n < per_section:
        # draw only when there IS a choice: r.choice on a 1-element list still
        # consumes from the shared stream and would silently shift physics's
        # entire bench, which must stay byte-identical across this amendment
        d = r.choice(heldrange_ds) if len(heldrange_ds) > 1 else heldrange_ds[0]
        n += add("niche_heldrange", _heldrange_example(r, niche, d))
    n = 0
    while n < per_section:
        f = FAMILIES[r.choice(BASIC_FAMILIES)]
        n += add("basics", f.gen(r, r.choice([2, 3])))
    return items

def bench_hash(items) -> str: return content_hash([asdict(i) for i in items])

def save_bench(items, path) -> None:
    with open(path, "w") as f:
        for it in items: f.write(json.dumps(asdict(it)) + "\n")

def load_bench(path) -> list:
    return [BenchItem(**json.loads(l)) for l in open(path)]
