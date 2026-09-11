import pytest
from akademeia.probes import (build_probes, assert_probe_integrity, load_probes,
                             save_probes, probes_hash, ProbeItem, PER_FAMILY)
from akademeia.data.families import FAMILIES, NICHE_FAMILIES, PHYSICS_TEMPLATES, render

def test_builds_exact_per_family_counts():
    items = build_probes(seed=5, per_family=4)
    assert len(items) == 4 * len(FAMILIES)
    assert_probe_integrity(items, per_family=4)

def test_niche_probes_never_use_bench_templates():
    bench_ids = {t.template_id for t in PHYSICS_TEMPLATES if t.split == "bench"}
    items = build_probes(seed=5, per_family=4)
    for it in items:
        assert it.template_id not in bench_ids

def test_vm_probes_use_train_shapes_only():
    from akademeia.data.families import _vm_split
    for it in build_probes(seed=5, per_family=4):
        if it.family == "vm":
            assert _vm_split(tuple(it.template_id[3:].split(","))) == "train"

def test_rejection_sampling_excludes_forbidden_texts():
    first = build_probes(seed=5, per_family=3)
    forbidden = {f"I:{i.text_in};O:{i.gold}$" for i in first}
    second = build_probes(seed=5, per_family=3, forbidden_texts=forbidden)
    got = {f"I:{i.text_in};O:{i.gold}$" for i in second}
    assert not (got & forbidden)

def test_count_assertion_fires_on_shrunken_set():
    """A silently shrunken eval set has burned this lab; the guard must exit."""
    items = build_probes(seed=5, per_family=4)
    with pytest.raises(SystemExit):
        assert_probe_integrity(items[:-1], per_family=4)

def test_split_assertion_fires_on_bench_template():
    items = build_probes(seed=5, per_family=4)
    stolen = [t for t in PHYSICS_TEMPLATES if t.split == "bench"][0]
    items[0] = ProbeItem("physics", 2, stolen.template_id, "E=m*g*h;m=2;g=3;h=4", "24")
    with pytest.raises(SystemExit):
        assert_probe_integrity(items, per_family=4)

def test_overlap_assertion_fires_on_pool_contact():
    items = build_probes(seed=5, per_family=4)
    leaked = f"I:{items[0].text_in};O:{items[0].gold}$"
    with pytest.raises(SystemExit):
        assert_probe_integrity(items, per_family=4, pool_texts=[leaked])

def test_save_load_roundtrip_and_hash(tmp_path):
    items = build_probes(seed=5, per_family=PER_FAMILY)
    p = tmp_path / "probes.jsonl"
    save_probes(items, p)
    back = load_probes(p)
    assert probes_hash(back) == probes_hash(items)

def test_load_hard_fails_on_truncated_file(tmp_path):
    items = build_probes(seed=5, per_family=PER_FAMILY)
    p = tmp_path / "probes.jsonl"
    save_probes(items[:-3], p)
    with pytest.raises(SystemExit):
        load_probes(p)
