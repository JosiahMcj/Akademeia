from akademeia.data.bench import build_bench, save_bench, load_bench
from akademeia.data.families import PHYSICS_TEMPLATES

def test_bench_sections_and_sizes():
    b = build_bench(seed=7)
    sec = {}
    for it in b:
        sec[it.section] = sec.get(it.section, 0) + 1
    assert sec == {"niche_unseen": 60, "niche_heldrange": 60, "basics": 60}

def test_unseen_uses_bench_templates_only():
    bench_ids = {t.template_id for t in PHYSICS_TEMPLATES if t.split == "bench"}
    b = build_bench(seed=7)
    for it in b:
        if it.section == "niche_unseen":
            assert it.template_id in bench_ids

def test_heldrange_operands_exceed_train_range():
    import re
    b = build_bench(seed=7)
    for it in b:
        if it.section == "niche_heldrange":
            ops = [int(x) for x in re.findall(r"=(\d+)", it.text_in.split(";", 1)[1])]
            assert max(ops) > 40  # train _PMAG hi at d=3 is 40

def test_deterministic_and_roundtrip(tmp_path):
    b1, b2 = build_bench(seed=7), build_bench(seed=7)
    assert [i.text_in for i in b1] == [i.text_in for i in b2]
    save_bench(b1, tmp_path / "bench.jsonl")
    b3 = load_bench(tmp_path / "bench.jsonl")
    assert [i.gold for i in b3] == [i.gold for i in b1]

def test_basics_difficulty_2_3_only():
    # Amendment 1: tiny d=1 spaces cannot yield item-novel bench draws
    for it in build_bench(seed=7):
        if it.section == "basics":
            assert it.difficulty in (2, 3)

def test_basics_rejection_sampling_excludes_pool_items():
    from akademeia.data.families import render, FAMILIES
    import random
    # build a pool-text set that contains EVERY d2/d3 compare item the bench
    # seed would draw first; the builder must skip them all and still finish
    r = random.Random(7)
    probe = build_bench(seed=7)
    basics_texts = {f"I:{i.text_in};O:{i.gold}$" for i in probe if i.section == "basics"}
    b = build_bench(seed=7, pool_texts=basics_texts)
    new_texts = {f"I:{i.text_in};O:{i.gold}$" for i in b if i.section == "basics"}
    assert not (new_texts & basics_texts)
