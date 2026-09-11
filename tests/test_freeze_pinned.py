"""The frozen instruments are v2's, byte-for-byte -- the cross-version claim
Q1 rests on. Pool/tokenizer/bench pins were REGENERATED equal on this
substrate at port time; the probe set is the carried v2 artifact. Any drift
in these constants is a port bug, never a new dataset."""
import json
from pathlib import Path
from akademeia.data.bench import load_bench, bench_hash
from akademeia.probes import load_probes, probes_hash
from akademeia.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]

PINS = {
    "pool_hash": "9b57f5e3d89cdaa0",
    "tokenizer_hash": "a2f260e5d7115716",
    "bench_hash_physics": "151af137e50a89e7",
    "bench_hash_vm": "88a159d01786c8ed",
    "probes_hash": "bbc1ffb9b5544fbe",
    "pool_s2_hash": "f099af78e665050f",
}

def manifest():
    return json.load(open(ROOT / "freeze" / "manifest.json"))

def test_manifest_pins_are_v2s():
    m = manifest()
    for k, v in PINS.items():
        assert m[k] == v, f"{k} drifted: {m[k]} != {v}"

def test_frozen_files_match_their_pins():
    assert CharTokenizer.load(
        ROOT / "freeze" / "tokenizer.json").hash() == PINS["tokenizer_hash"]
    for niche in ("physics", "vm"):
        bench = load_bench(ROOT / "freeze" / f"bench-{niche}.jsonl")
        assert bench_hash(bench) == PINS[f"bench_hash_{niche}"]
    probes = load_probes(ROOT / "freeze" / "probes.jsonl",
                         manifest()["probes_per_family"])
    assert probes_hash(probes) == PINS["probes_hash"]
