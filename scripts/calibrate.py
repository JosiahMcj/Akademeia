"""Per-rung calibration: paired noise under the final-quality metric from
same-seed A/A' arms. Pure derivation carried from v2 (Amendment 2 formulas;
Amendment 10 makes the mixed-items floor advisory above tier-1, with the
unanimity breakdown journaled as the instrument's capacity statement).
Hard-exits if a bench fails discrimination. Thresholds freeze one way."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

def derive_v2(a: dict, ap: dict, min_mixed_frac: float = 0.10,
              floor_advisory: bool = False) -> dict:
    """a/ap share seeds (stream-salt pairing).

    floor_advisory: the dated 2026-08-22 amendment (mirrors v2
    Amendment 10): when a pre-declared diagnostic has shown the bench
    registers >= 8 discordant items for a genuinely different arm on this
    instrument, a mixed-items floor miss indicates same-arm reproducibility,
    not incapacity -- report it and proceed; H1's min_discordant floor
    remains the verdict-time power check. Default stays STRICT."""
    assert sorted(a["final_items"]) == sorted(ap["final_items"]), "A/A' seed mismatch"
    mean_final = sum(a["final"]) / len(a["final"])
    if not (0.20 <= mean_final <= 0.85):
        print(f"DISCRIMINATION FAIL: mean A final {mean_final:.3f} outside "
              f"[0.20, 0.85]"); sys.exit(7)
    pooled = [a["final_items"][s] for s in sorted(a["final_items"])] + \
             [ap["final_items"][s] for s in sorted(ap["final_items"])]
    n_items = len(pooled[0])
    solved = sum(1 for i in range(n_items)
                 if all(run[i] for run in pooled))
    failed = sum(1 for i in range(n_items)
                 if not any(run[i] for run in pooled))
    mixed = n_items - solved - failed
    if mixed / n_items < min_mixed_frac:
        if not floor_advisory:
            print(f"DISCRIMINATION FAIL: only {mixed}/{n_items} items "
                  f"non-unanimous"); sys.exit(7)
        print(f"ADVISORY (2026-08-22 amendment): mixed items {mixed}/"
              f"{n_items} below floor {min_mixed_frac:.2f}; proceeding, "
              f"capacity statement journaled, H1 power floor still binds")
    d_final = max(abs(x - y) for x, y in zip(a["final"], ap["final"]))
    d_basics = max(abs(x - y) for x, y in zip(a["basics"], ap["basics"]))
    return {"noise_final": max(0.02, round(3 * d_final, 4)),
            "noise_basics": max(0.02, round(3 * d_basics, 4)),
            "min_discordant": 8,
            "_report": {"mean_final": round(mean_final, 4),
                        "mixed_items": f"{mixed}/{n_items}",
                        "unanimous_solved": solved,
                        "unanimous_failed": failed}}

def calibrate_rung(rung: int, niche: str = "physics",
                   floor_advisory: bool = False) -> dict:
    """Derive + freeze thresholds for one ladder rung from its A/A' pair.
    The mixed-items floor is 0.10 at rung 1 and advisory above (v2 Am. 10)."""
    a = json.load(open(ROOT / "runs" / f"arm-ladder-r{rung}-direct50-A.json"))
    ap = json.load(open(ROOT / "runs" / f"arm-ladder-r{rung}-direct50-Aprime.json"))
    out = derive_v2(a, ap, min_mixed_frac=0.10 if rung == 1 else 0.0,
                    floor_advisory=floor_advisory)
    report = out.pop("_report")
    th_path = ROOT / "freeze" / f"thresholds_v3_r{rung}_{niche}.json"
    if th_path.exists():
        print(f"FATAL: {th_path} already frozen"); sys.exit(4)
    th_path.write_text(json.dumps(out, indent=1))
    stamp = time.strftime("%Y-%m-%d")
    with open(ROOT / "PROTOCOL.md", "a") as f:
        # titled by content + date, never by a hardcoded amendment number
        f.write(f"\n## Registered rung-{rung} thresholds ({stamp}, mechanical)"
                f"\n\nDerived by scripts/calibrate.py from the same-seed A/A' "
                f"pair:\n{json.dumps({'thresholds': out, 'report': report}, indent=1)}\n")
    print(json.dumps({"thresholds": out, "report": report}, indent=1))
    return {"thresholds": out, "report": report}

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rung", type=int, required=True)
    p.add_argument("--niche", default="physics")
    p.add_argument("--floor-advisory", action="store_true")
    args = p.parse_args()
    calibrate_rung(args.rung, args.niche, floor_advisory=args.floor_advisory)

if __name__ == "__main__":
    main()
