"""Is a homogeneous node class ONE instrument, or many?

The resume law says two hosts are different instruments even when the device
string matches. On a cluster that would fragment a campaign across identical
nodes -- but "identical" is a claim, and this script is how it gets tested
rather than assumed.

The test is BITWISE equality, not the parity band: parity asks whether two
instruments agree within a measured tolerance, which is far too coarse to
prove two boxes are the same one. Under deterministic algorithms the same
seed on the same hardware must reproduce the loss curve EXACTLY, so exact
equality across nodes is the discriminating check.

Writes one JSON per (host, gpu) to runs/bitwise/; compare with --compare.
"""
import argparse, hashlib, json, os, platform, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "bitwise"

def run(device: str) -> dict:
    import torch
    from akademeia.config import SubstrateConfig
    from akademeia.trainer import train_run, tf32_enabled
    from akademeia.curriculum import Segment, Curriculum
    from akademeia.data.pool import build_pool
    from akademeia.tokenizer import CharTokenizer
    from akademeia.data.families import render
    import tempfile
    micro = SubstrateConfig(n_layers=2, d_model=64, n_heads=4, ctx=64,
                            batch_size=8, total_tokens=60_000,
                            eval_every_tokens=30_000, lr_warmup_steps=5)
    pool = build_pool(seed=1, per_cell=100)
    tok = CharTokenizer.fit([render(e) for e in pool.examples])
    c = Curriculum("bitwise", (Segment(60_000, {"copy": 1.0, "physics": 1.0},
                                       {"copy": (1, 2), "physics": (1, 1)}),))
    with tempfile.TemporaryDirectory() as td:
        res = train_run(micro, c, pool, tok, seed=11, out_dir=Path(td) / "a",
                        device=device)
        # the full-precision loss curve is the fingerprint: repr() keeps every
        # bit, so two instruments that differ anywhere in the numerics differ
        # here, and a mean-based comparison could not see it
        exact = [repr(float(x)) for x in res.losses]
    gpu = None
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu = {"name": torch.cuda.get_device_name(0),
               "capability": list(torch.cuda.get_device_capability(0))}
    return {"host": platform.node().split(".")[0], "device": device,
            "gpu": gpu, "tf32": tf32_enabled(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "losses_exact": exact,
            "fingerprint": hashlib.sha256(
                "|".join(exact).encode()).hexdigest()[:32]}

def compare() -> int:
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    if len(recs) < 2:
        print(f"need at least 2 records, have {len(recs)}"); return 2
    groups = {}
    for r in recs:
        groups.setdefault(r["fingerprint"], []).append(r)
    for r in recs:
        print(f"  {r['host']:>6} gpu={str(r['cuda_visible']):>4} "
              f"tf32={str(r['tf32']):>5} job={str(r['slurm_job']):>6} "
              f"-> {r['fingerprint'][:16]}")
    print()
    if len(groups) == 1:
        print(f"IDENTICAL across all {len(recs)} records: the node class is "
              f"ONE instrument.")
        print("A node-class instrument key is justified by measurement.")
        return 0
    print(f"DIFFERENT: {len(groups)} distinct fingerprints across {len(recs)} "
          f"records. These are NOT one instrument; pin the campaign to a "
          f"single node and keep the per-host resume key.")
    return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.compare:
        sys.exit(compare())
    rec = run(args.device)
    tag = args.tag or f"{rec['host']}-gpu{rec['cuda_visible']}-tf32{int(rec['tf32'])}"
    (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=1))
    print(f"[bitwise] {tag}: {rec['fingerprint'][:16]} "
          f"({rec['gpu']['name'] if rec['gpu'] else rec['device']})")

if __name__ == "__main__":
    main()
