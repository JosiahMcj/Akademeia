"""Study Q4: what does externalizing intermediates buy on its own task?

Registered claim discipline (PROTOCOL A3): this measures the scratchpad
format ON ITS OWN TASK. It is never a comparison against the plain format --
the tasks differ, so a cross-format score difference would be meaningless.

Design, mirroring the ladder's own baseline so the numbers are read the same
way: a 50/50 basics/scratchpad curriculum on the Q4 variant pool, three
seeds, same-seed A/A' pair (stream salts 0 and 1) for the paired noise this
rung's thresholds are derived from. Gated metric = niche_unseen (novel
combinations of trained symbols at trained lengths, FULL-TRACE exact match).
The length-4 chains ride the reported-never-gated diagnostic slot.

Stages:
  preflight  one seed, one rung: the registered gate on wiring this study at
             all. If the score sits at a floor or a ceiling the study cannot
             discriminate and does not run -- that is the finding, recorded
             (a statistic must be able to MOVE).
  arms       the A/A' pair
  verdict    thresholds derived by the same calibrate law, then the report
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from akademeia.curriculum import Curriculum, Segment, validate
from akademeia.data.families import BASIC_FAMILIES
from akademeia.journal import Journal
from akademeia.tiers import RUNGS
from run_arm import run_curriculum
from calibrate import derive_v2

ROOT = Path(__file__).resolve().parents[1]
NICHE = "scratchpad"
POOL_FILE, POOL_KEY = "pool_q4.jsonl", "pool_q4_hash"
EXTRA = (NICHE,)
SEEDS = (11, 22, 33)
BAND = (0.20, 0.85)          # the registered discrimination band

def q4_curriculum(budget: int) -> Curriculum:
    """50/50 basics/scratchpad, the direct50 shape the ladder baseline uses."""
    w = {f: 1.0 / len(BASIC_FAMILIES) for f in BASIC_FAMILIES}
    w[NICHE] = 1.0
    d = {f: (1, 3) for f in BASIC_FAMILIES}
    d[NICHE] = (1, 3)
    c = Curriculum(f"q4-direct50-{NICHE}", (Segment(budget, w, d),))
    validate(c, budget, extra_families=EXTRA)
    return c

def run(rung: int, device: str, seeds, tag: str, salt: int) -> dict:
    cfg = RUNGS[rung]
    return run_curriculum(cfg, q4_curriculum(cfg.total_tokens), NICHE,
                          seeds=tuple(seeds), tag=tag, stream_salt=salt,
                          phase=f"Q4-r{rung}", pool_file=POOL_FILE,
                          pool_key=POOL_KEY, device=device)

def cmd_preflight(rung, device, journal):
    arm = run(rung, device, (SEEDS[0],), f"q4-preflight-r{rung}", 0)
    score = arm["final"][0]
    ok = BAND[0] <= score <= BAND[1]
    row = {"kind": "study", "study": "Q4-preflight", "rung": rung,
           "device": device, "niche_unseen": score,
           "heldrange_extrap": list(arm["heldrange"].values()),
           "basics": arm["basics"][0], "band": list(BAND),
           "verdict": "IN BAND" if ok else "OUT OF BAND"}
    journal.append(row)
    print(json.dumps(row, indent=1, default=str))
    if not ok:
        print(f"PREFLIGHT OUT OF BAND: {score:.3f} outside {BAND}; the study "
              f"cannot discriminate at this rung and does not run. Recorded.")
        sys.exit(7)

def cmd_arms(rung, device):
    run(rung, device, SEEDS, f"q4-r{rung}-A", 0)
    run(rung, device, SEEDS, f"q4-r{rung}-Aprime", 1)

def cmd_verdict(rung, device, journal):
    a = json.load(open(ROOT / "runs" / f"arm-q4-r{rung}-A.json"))
    ap = json.load(open(ROOT / "runs" / f"arm-q4-r{rung}-Aprime.json"))
    out = derive_v2(a, ap, min_mixed_frac=0.10 if rung == 1 else 0.0)
    report = out.pop("_report")
    th = ROOT / "freeze" / f"thresholds_v3_r{rung}_{NICHE}.json"
    if th.exists():
        print(f"FATAL: {th} already frozen"); sys.exit(4)
    th.write_text(json.dumps(out, indent=1))
    mean = lambda xs: sum(xs) / len(xs)
    row = {"kind": "study", "study": "Q4", "rung": rung, "device": device,
           "niche_unseen_A": a["final"], "niche_unseen_Aprime": ap["final"],
           "niche_unseen_mean": round(mean(a["final"] + ap["final"]), 4),
           "extrap_diagnostic": {"A": a["heldrange"], "Aprime": ap["heldrange"],
                                 "gated": False},
           "basics_mean": round(mean(a["basics"] + ap["basics"]), 4),
           "thresholds": out, "calibration_report": report,
           "claim_scope": "the scratchpad format on its own task; never a "
                          "comparison against the plain format"}
    journal.append(row)
    print(json.dumps(row, indent=1, default=str))

def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("stage", choices=["preflight", "arms", "verdict"])
    ap_.add_argument("--rung", type=int, required=True, choices=sorted(RUNGS))
    ap_.add_argument("--device", default="cpu")
    args = ap_.parse_args()
    journal = Journal(ROOT / "runs" / "journal.jsonl")
    if args.stage == "preflight":
        cmd_preflight(args.rung, args.device, journal)
    elif args.stage == "arms":
        cmd_arms(args.rung, args.device)
    else:
        cmd_verdict(args.rung, args.device, journal)

if __name__ == "__main__":
    main()
