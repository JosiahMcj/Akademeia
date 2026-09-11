from akademeia.data.pool import build_pool, save_pool, load_pool
from akademeia.data.families import FAMILIES

def test_build_deterministic_and_indexed(tmp_path):
    p1 = build_pool(seed=42, per_cell=30)
    p2 = build_pool(seed=42, per_cell=30)
    assert p1.content_hash == p2.content_hash
    assert build_pool(seed=43, per_cell=30).content_hash != p1.content_hash
    # every (family, difficulty) cell present with per_cell examples
    for fam in FAMILIES:
        for d in (1, 2, 3):
            assert len(p1.index[(fam, d)]) == 30
    # physics cells contain ONLY train-split templates
    from akademeia.data.families import PHYSICS_TEMPLATES
    train_ids = {t.template_id for t in PHYSICS_TEMPLATES if t.split == "train"}
    for i in p1.index[("physics", 2)]:
        assert p1.examples[i].template_id in train_ids

def test_save_load_roundtrip(tmp_path):
    p = build_pool(seed=1, per_cell=5)
    save_pool(p, tmp_path / "pool.jsonl")
    q = load_pool(tmp_path / "pool.jsonl")
    assert q.content_hash == p.content_hash and len(q.examples) == len(p.examples)
