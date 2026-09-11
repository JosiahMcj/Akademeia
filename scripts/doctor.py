"""Preflight + P0 determinism proof for the torch substrate. --smoke runs a
micro end-to-end train+eval TWICE on the chosen device and requires matching
curves (no false greens: it exercises the real path). Exits nonzero on any
failure. Run with --device cuda on the campaign box before anything counts there."""
import argparse, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    import numpy as np
    import torch
    from akademeia.ops import device_lock, wait_for_load, gpu_state
    print(f"torch {torch.__version__}; cuda available: "
          f"{torch.cuda.is_available()}; device requested: {args.device}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("FATAL: cuda requested but unavailable"); sys.exit(2)
    gpu = gpu_state()
    if gpu is not None:
        print(f"nvidia-smi: mem free {gpu['mem_free_pct']:.0f}%, "
              f"util {gpu['util_pct']:.0f}%")
    wait_for_load()   # delay on a busy box, never kill the chain
    if not args.smoke:
        print("doctor OK"); return
    from akademeia.config import SubstrateConfig
    from akademeia.trainer import train_run
    from akademeia.curriculum import Segment, Curriculum
    from akademeia.data.pool import build_pool
    from akademeia.tokenizer import CharTokenizer
    from akademeia.data.families import render
    from akademeia.evalrunner import eval_checkpoint
    from akademeia.data.bench import build_bench, save_bench
    micro = SubstrateConfig(n_layers=2, d_model=64, n_heads=4, ctx=64,
                            batch_size=8, total_tokens=60_000,
                            eval_every_tokens=30_000, lr_warmup_steps=5)
    pool = build_pool(seed=1, per_cell=100)
    tok = CharTokenizer.fit([render(e) for e in pool.examples])
    c = Curriculum("smoke", (Segment(60_000, {"copy": 1.0, "physics": 1.0},
                                     {"copy": (1, 2), "physics": (1, 1)}),))
    with device_lock(args.device):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            r1 = train_run(micro, c, pool, tok, seed=11, out_dir=td / "a",
                           device=args.device)
            r2 = train_run(micro, c, pool, tok, seed=11, out_dir=td / "b",
                           device=args.device)
            if not np.allclose(r1.losses, r2.losses, rtol=1e-6):
                print(f"FATAL: same-seed curves diverge on {args.device}; "
                      f"pairing is fiction until P0 passes here"); sys.exit(6)
            bench = build_bench(seed=7, per_section=10)
            tok.assert_covers([f"I:{i.text_in};O:{i.gold}$" for i in bench])
            save_bench(bench, td / "bench.jsonl"); tok.save(td / "tok.json")
            res = eval_checkpoint(micro, r1.checkpoints[-1][1],
                                  td / "bench.jsonl", td / "tok.json",
                                  device=args.device)
            print(f"smoke OK on {args.device}: losses {r1.losses[0]:.3f}->"
                  f"{r1.losses[-1]:.3f}, eval {res['scores']}")

if __name__ == "__main__":
    main()
