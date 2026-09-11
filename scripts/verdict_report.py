"""Full statistical report for a ladder-rung verdict -- the ab-verdict
discipline as a command instead of a habit: paired per-seed differences,
discordant pairs, the exact one-sided sign-test p, margins vs the rung's
frozen thresholds, retention, and the heldrange extrapolation diagnostic.

Read-only. Prints; never journals, never gates -- the gate already ran.
Ported from v2; adds the v3 instrument law: arms from different hosts or
hashes are a VOID comparison unless a pair gate covered them.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akademeia.stats import sign_test, discordant
from akademeia.gates import ThresholdsV2

ROOT = Path(__file__).resolve().parents[1]

def report(arm: dict, base: dict, th: ThresholdsV2, arm_name: str,
           base_name: str, out=print) -> dict:
    seeds = sorted(arm["final_items"])
    if sorted(base["final_items"]) != seeds:
        raise SystemExit(f"seed mismatch: {seeds} vs {sorted(base['final_items'])}")
    out(f"== {arm_name} vs {base_name} ==")
    out(f"{'seed':>6} {'arm':>7} {'base':>7} {'paired diff':>12}")
    diffs = []
    for i, s in enumerate(seeds):
        d = arm["final"][i] - base["final"][i]
        diffs.append(d)
        out(f"{s:>6} {arm['final'][i]:>7.3f} {base['final'][i]:>7.3f} {d:>+12.3f}")
    mean_d = sum(diffs) / len(diffs)
    out(f"mean paired diff {mean_d:+.4f}  (H2 margin: noise_final={th.noise_final})")
    w = l = 0
    for s in seeds:
        dw, dl = discordant(arm["final_items"][s], base["final_items"][s])
        w += dw; l += dl
    n = w + l
    p_sup = sign_test(w, l)
    p_inf = sign_test(l, w)
    out(f"items: wins={w} losses={l} discordant={n} "
        f"(power floor {th.min_discordant}{' -- UNDERPOWERED' if n < th.min_discordant else ''})")
    out(f"sign test: P(arm better | null) = {p_sup:.4g}; "
        f"P(arm worse | null) = {p_inf:.4g}")
    rd = (sum(arm["basics"]) / len(arm["basics"])
          - sum(base["basics"]) / len(base["basics"]))
    out(f"retention delta {rd:+.3f} (band {th.noise_basics}; "
        f"reported, gating only for generalist claims)")
    if "heldrange" in arm:
        hr_a = [arm["heldrange"][s] for s in seeds]
        hr_b = [base.get("heldrange", {}).get(s) for s in seeds]
        out(f"heldrange diagnostic (NOT in metric): arm={hr_a} base={hr_b}")
    out(f"per-seed sign agreement: "
        f"{sum(d > 0 for d in diffs)}/{len(diffs)} seeds favor the arm")
    return {"mean_paired_diff": mean_d, "wins": w, "losses": l,
            "p_superior": p_sup, "p_worse": p_inf, "retention_delta": rd,
            "underpowered": n < th.min_discordant}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="tag, e.g. ladder-r1-branch")
    ap.add_argument("--base", required=True, help="tag, e.g. ladder-r1-direct50-A")
    ap.add_argument("--rung", type=int, required=True)
    ap.add_argument("--niche", default="physics")
    args = ap.parse_args()
    th = ThresholdsV2(**json.load(
        open(ROOT / "freeze" / f"thresholds_v3_r{args.rung}_{args.niche}.json")))
    arm = json.load(open(ROOT / "runs" / f"arm-{args.arm}.json"))
    base = json.load(open(ROOT / "runs" / f"arm-{args.base}.json"))
    if arm["hashes"] != base["hashes"]:
        raise SystemExit("hash mismatch between arms: comparison is VOID, "
                         "do not report these numbers")
    if arm.get("host") != base.get("host") or arm.get("device") != base.get("device"):
        raise SystemExit("host/device mismatch between arms: a cross-instrument "
                         "comparison is VOID without a pair gate covering it")
    report(arm, base, th, args.arm, args.base)

if __name__ == "__main__":
    main()
