"""Frozen pool: built once, hashed; curricula reorder/reweight it, never add.
Regenerable from (seed, per_cell); freeze/manifest.json pins the hash."""
from dataclasses import dataclass, asdict
import json, random
from akademeia.config import content_hash
from akademeia.data.families import FAMILIES, Example

@dataclass
class Pool:
    examples: list
    index: dict           # (family, difficulty) -> list[int]
    content_hash: str

def _index(examples) -> dict:
    idx = {}
    for i, ex in enumerate(examples):
        idx.setdefault((ex.family, ex.difficulty), []).append(i)
    return idx

def build_pool(seed: int, per_cell: int = 20_000, physics_extra=(),
               forbidden_texts=frozenset(), extra_examples=()) -> Pool:
    """physics_extra (Study S2): extra physics templates for a VARIANT pool.
    extra_examples: pre-built Examples APPENDED to a variant pool, for a niche
    whose family is deliberately absent from the frozen FAMILIES registry
    (adding one there would rewrite every pinned pool hash). They are appended
    after the registry families and indexed like any other example, so the
    stream can weight them by (family, difficulty) without the registry
    knowing. Empty by default -- the default path stays byte-identical.
    forbidden_texts: rendered items the pool may not contain -- variant pools
    exclude the frozen bench BY CONSTRUCTION, because high-coverage family
    cells (e.g. compare-d2: 8100 pairs vs 20k draws) make any fresh
    realization near-certain to contain some bench items, so seed retries
    cannot work. The default path (no extras, no forbidden) must remain
    byte-identical -- a test pins it."""
    from akademeia.data.families import PHYSICS_TEMPLATES, _physics_gen, render
    r = random.Random(seed)
    pool_templates = (PHYSICS_TEMPLATES + tuple(physics_extra)
                      if physics_extra else None)
    examples = []
    for name in sorted(FAMILIES):
        fam = FAMILIES[name]
        for d in fam.difficulties:
            made = 0
            while made < per_cell:
                if name == "physics" and pool_templates is not None:
                    ex = _physics_gen(r, d, pool_templates=pool_templates)
                else:
                    ex = fam.gen(r, d)
                if forbidden_texts and render(ex) in forbidden_texts:
                    continue                      # redraw: bench item verbatim
                examples.append(ex)
                made += 1
    examples.extend(extra_examples)
    h = content_hash([asdict(e) for e in examples])
    return Pool(examples, _index(examples), h)

def save_pool(pool: Pool, path) -> None:
    with open(path, "w") as f:
        for ex in pool.examples:
            f.write(json.dumps(asdict(ex)) + "\n")

def load_pool(path) -> Pool:
    examples = [Example(**json.loads(l)) for l in open(path)]
    return Pool(examples, _index(examples),
                content_hash([asdict(e) for e in examples]))
