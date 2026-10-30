"""Study Q3: does the structural-generalization lottery NARROW with scale?

n=6 same-seed A/A' pairs per rung, vm niche, direct50. DESCRIPTIVE of spread:
no gates, no promotion. The registered rule (PROTOCOL amendment 2026-08-11):
per-seed outcome = pair mean (A+A')/2; spread = sample sd across seeds;
adjacent measured rungs compared by the EXACT Brown-Forsythe permutation test;
p < 0.05 labels the step NARROWS or WIDENS by direction, else FLAT. The
realization band is journaled alongside so spread is never mistaken for pure
structure. Resume-safe via run_curriculum's hash+device-matched arm reuse."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from akademeia.curriculum import validate
from akademeia.journal import Journal
from akademeia.tiers import RUNGS
from akademeia.variance import (seed_outcomes, realization_band, spread,
                                bf_exact_perm)
from run_arm import run_curriculum
from ladder import direct50

ROOT = Path(__file__).resolve().parents[1]
NICHE = "vm"
SEEDS6 = (11, 22, 33, 44, 55, 66)
EXT_SEEDS = (77, 88, 99, 110, 121, 132)   # 2026-09-21 power amendment
ALPHA = 0.05

def seeds_for(extended: bool):
    """The registered n=6 set, or the doubled set. Unlike a mean, a VARIANCE
    estimate genuinely gains from more pairs: this study reported FLAT three
    times while declaring itself underpowered at n=6 by its own rule, and
    doubling the pairs is the remedy it prescribed for itself."""
    return SEEDS6 + EXT_SEEDS if extended else SEEDS6

def cmd_run(rung: int, device: str, extended: bool = False, only_seed=None):
    cfg = RUNGS[rung]
    B = cfg.total_tokens
    c = direct50(NICHE, B); validate(c, B)
    seeds = (only_seed,) if only_seed else seeds_for(extended)
    suffix = "-ext" if extended and not only_seed else ""
    for salt, nm in ((0, "A"), (1, "Aprime")):
        run_curriculum(cfg, c, NICHE, seeds=seeds,
                       tag=f"s1-r{rung}-vm-{nm}{suffix}",
                       stream_salt=salt, phase="S1", device=device,
                       assemble=(only_seed is None))

def _arms(rung: int, extended: bool):
    """The n=6 arms, or the n=12 extension arms. Separate tags by design: the
    registered n=6 verdict must stay reproducible from its own files after
    the extension exists (2026-09-21 amendment)."""
    sfx = "-ext" if extended else ""
    return (ROOT / "runs" / f"arm-s1-r{rung}-vm-A{sfx}.json",
            ROOT / "runs" / f"arm-s1-r{rung}-vm-Aprime{sfx}.json")

def rung_stats(rung: int, extended: bool = False) -> dict | None:
    pa, pp = _arms(rung, extended)
    if not (pa.exists() and pp.exists()):
        return None
    a, ap = json.loads(pa.read_text()), json.loads(pp.read_text())
    out = seed_outcomes(a, ap)
    return {"outcomes": [round(x, 4) for x in out], "n_pairs": len(out),
            "mean": round(sum(out) / len(out), 4),
            "spread": round(spread(out), 4),
            "band": round(realization_band(a, ap), 4)}

def cmd_verdict(extended: bool = False):
    journal = Journal(ROOT / "runs" / "journal.jsonl")
    measured = {r: rung_stats(r, extended) for r in sorted(RUNGS)}
    measured = {r: s for r, s in measured.items() if s}
    if len(measured) < 2:
        print(f"[S1] only {len(measured)} rung(s) measured; nothing to compare")
        return
    steps = []
    rungs = sorted(measured)
    for lo, hi in zip(rungs, rungs[1:]):
        # recompute outcomes at full precision for the exact test
        pa, pp = _arms(lo, extended); pb, pbp = _arms(hi, extended)
        a, ap = json.loads(pa.read_text()), json.loads(pp.read_text())
        b, bp = json.loads(pb.read_text()), json.loads(pbp.read_text())
        x, y = seed_outcomes(a, ap), seed_outcomes(b, bp)
        stat, p = bf_exact_perm(x, y)
        if p < ALPHA:
            label = ("NARROWS" if spread(y) < spread(x) else "WIDENS")
        else:
            label = "FLAT"
        steps.append({"step": f"r{lo}->r{hi}", "stat": round(stat, 4),
                      "p": round(p, 5), "label": label,
                      "spread_lo": round(spread(x), 4),
                      "spread_hi": round(spread(y), 4)})
        print(f"[S1] r{lo}->r{hi}: spread {spread(x):.4f} -> {spread(y):.4f}, "
              f"BF stat {stat:.4f}, exact p={p:.5f} -> {label}")
    row = {"kind": "study", "study": "S1-Q3-ext" if extended else "S1-Q3",
           "n_pairs": (len(SEEDS6 + EXT_SEEDS) if extended else len(SEEDS6)),
           "niche": NICHE,
           "per_rung": measured, "steps": steps,
           "registered_rule": "2026-08-11 amendment: pair-mean outcomes, "
                              "sample-sd spread, exact Brown-Forsythe "
                              f"permutation, alpha={ALPHA}"}
    journal.append(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--only-seed", type=int, default=None)
    ap.add_argument("stage", choices=["run", "verdict"])
    ap.add_argument("--rung", type=int, choices=sorted(RUNGS))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.stage == "run":
        if args.rung is None:
            print("run needs --rung"); sys.exit(2)
        cmd_run(args.rung, args.device, extended=args.extended,
                only_seed=args.only_seed)
    else:
        cmd_verdict(extended=args.extended)

if __name__ == "__main__":
    main()
