"""Cheap pre-flight: ONE full-budget single-seed run per niche, to check a
bench is not floored before committing 12 runs to it. Full budget because v2
proved 1/3-budget score probes uninformative.

Results are journaled as kind="preflight" and are NEVER measured arms: they
are single-seed, unpaired, and must not be reported as a result or fed to a
gate."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from akademeia.curriculum import validate
from akademeia.journal import Journal
from akademeia.tiers import RUNGS
from run_arm import run_curriculum
from ladder import direct50

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niches", nargs="*", default=["physics"])
    ap.add_argument("--rung", type=int, default=1, choices=sorted(RUNGS))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = RUNGS[args.rung]
    journal = Journal(ROOT / "runs" / "journal.jsonl")
    for niche in args.niches:
        c = direct50(niche, cfg.total_tokens)
        validate(c, cfg.total_tokens)
        arm = run_curriculum(cfg, c, niche, seeds=(args.seed,),
                             tag=f"preflight-r{args.rung}-{niche}",
                             phase="preflight", device=args.device)
        final = arm["final"][0]
        floored = final < 0.05
        journal.append({"kind": "preflight", "rung": args.rung, "niche": niche,
                        "seed": args.seed, "final": final, "floored": floored,
                        "device": arm["device"]})
        print(f"[preflight r{args.rung}] {niche}: final={final:.3f} "
              f"{'FLOORED -- do not commit a campaign' if floored else 'ok'}")

if __name__ == "__main__":
    main()
