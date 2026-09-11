"""The MLX parity reference must stay internally consistent and pinned to the
same frozen instruments -- a hand-mangled reference would make the parity
gate meaningless."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

def ref():
    return json.load(open(ROOT / "reference" / "mlx_tier1.json"))

def test_band_is_the_registered_formula():
    r = ref()
    d = max(abs(x - y) for x, y in zip(r["final_A"], r["final_Aprime"]))
    assert r["noise_final"] == max(0.02, round(3 * d, 4))
    db = max(abs(x - y) for x, y in zip(r["basics_A"], r["basics_Aprime"]))
    assert r["noise_basics"] == max(0.02, round(3 * db, 4))

def test_mean_matches_the_stored_finals():
    r = ref()
    assert r["mean_final_A"] == pytest.approx(
        sum(r["final_A"]) / len(r["final_A"]))

def test_reference_binds_to_the_frozen_instruments():
    r = ref()
    m = json.load(open(ROOT / "freeze" / "manifest.json"))
    assert r["hashes"]["bench"] == m["bench_hash_physics"]
    assert r["hashes"]["pool"] == m["pool_hash"]
    assert r["hashes"]["tokenizer"] == m["tokenizer_hash"]

def test_reference_substrate_is_tier1():
    from akademeia.tiers import TIER1
    s = ref()["substrate"]
    assert s["n_layers"] == TIER1.n_layers
    assert s["d_model"] == TIER1.d_model
    assert s["total_tokens"] == TIER1.total_tokens
    assert s["lr"] == TIER1.lr

def test_seeds_are_the_lab_triplet():
    assert ref()["seeds"] == [11, 22, 33]
