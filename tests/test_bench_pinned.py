"""Pin the benches against silent drift.

An `r.choice([2])` on a single-element list still consumes from the shared RNG
stream, which silently rebuilt the ENTIRE physics benchmark while I was editing
only the vm path. Nothing errored; only the hash moved. These pins turn that
class of accident into a failing test.

Computed with no pool_texts, so they are a pure function of the generators and
the seeds. Changing either of these values must be a deliberate, dated
PROTOCOL amendment -- never a side effect.
"""
from akademeia.data.bench import build_bench, bench_hash

# Updated under PROTOCOL Amendment 5 (physics: heldrange excludes division;
# vm: primitives always trainable, combinations-only held out, d3 operands
# small, held band 21-40). Changing a pin requires an amendment in the same
# commit -- that is the whole point of pinning.
PINNED = {"physics": ("101bfd693bf91780", 20260731),
          "vm": ("aec4f5d0fb819dc6", 20260801)}

def test_bench_hashes_are_pinned():
    for niche, (want, seed) in PINNED.items():
        got = bench_hash(build_bench(seed=seed, niche=niche))
        assert got == want, (
            f"{niche} bench drifted: {got} != {want}. If this change is "
            f"intended, amend PROTOCOL and update the pin in the same commit.")

def test_physics_and_vm_benches_differ():
    assert PINNED["physics"][0] != PINNED["vm"][0]
