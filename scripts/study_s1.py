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
ALPHA = 0.05

def cmd_run(rung: int, device: str):
    cfg = RUNGS[rung]
    B = cfg.total_tokens
    c = direct50(NICHE, B); validate(c, B)
    run_curriculum(cfg, c, NICHE, seeds=SEEDS6, tag=f"s1-r{rung}-vm-A",
                   stream_salt=0, phase="S1", device=device)
    run_curriculum(cfg, c, NICHE, seeds=SEEDS6, tag=f"s1-r{rung}-vm-Aprime",
                   stream_salt=1, phase="S1", device=device)

def rung_stats(rung: int) -> dict | None:
    pa = ROOT / "runs" / f"arm-s1-r{rung}-vm-A.json"
    pp = ROOT / "runs" / f"arm-s1-r{rung}-vm-Aprime.json"
    if not (pa.exists() and pp.exists()):
        return None
    a, ap = json.loads(pa.read_text()), json.loads(pp.read_text())
    out = seed_outcomes(a, ap)
    return {"outcomes": [round(x, 4) for x in out],
            "mean": round(sum(out) / len(out), 4),
            "spread": round(spread(out), 4),
            "band": round(realization_band(a, ap), 4)}

def cmd_verdict():
    journal = Journal(ROOT / "runs" / "journal.jsonl")
    measured = {r: rung_stats(r) for r in sorted(RUNGS)}
    measured = {r: s for r, s in measured.items() if s}
    if len(measured) < 2:
        print(f"[S1] only {len(measured)} rung(s) measured; nothing to compare")
        return
    steps = []
    rungs = sorted(measured)
    for lo, hi in zip(rungs, rungs[1:]):
        # recompute outcomes at full precision for the exact test
        a = json.loads((ROOT / "runs" / f"arm-s1-r{lo}-vm-A.json").read_text())
        ap = json.loads((ROOT / "runs" / f"arm-s1-r{lo}-vm-Aprime.json").read_text())
        b = json.loads((ROOT / "runs" / f"arm-s1-r{hi}-vm-A.json").read_text())
        bp = json.loads((ROOT / "runs" / f"arm-s1-r{hi}-vm-Aprime.json").read_text())
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
    row = {"kind": "study", "study": "S1-Q3", "niche": NICHE,
           "per_rung": measured, "steps": steps,
           "registered_rule": "2026-08-11 amendment: pair-mean outcomes, "
                              "sample-sd spread, exact Brown-Forsythe "
                              f"permutation, alpha={ALPHA}"}
    journal.append(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "verdict"])
    ap.add_argument("--rung", type=int, choices=sorted(RUNGS))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.stage == "run":
        if args.rung is None:
            print("run needs --rung"); sys.exit(2)
        cmd_run(args.rung, args.device)
    else:
        cmd_verdict()

if __name__ == "__main__":
    main()
