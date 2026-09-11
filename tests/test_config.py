import dataclasses, pytest
from akademeia.config import SubstrateConfig, content_hash

def test_substrate_hash_stable_and_seed_free():
    a = SubstrateConfig()
    b = SubstrateConfig()
    assert a.hash() == b.hash()
    assert len(a.hash()) == 16
    assert "seed" not in [f.name for f in dataclasses.fields(SubstrateConfig)]

def test_substrate_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SubstrateConfig().d_model = 128  # type: ignore

def test_content_hash_deterministic():
    assert content_hash({"b": 2, "a": 1}) == content_hash({"a": 1, "b": 2})
    assert content_hash("x") != content_hash("y")
