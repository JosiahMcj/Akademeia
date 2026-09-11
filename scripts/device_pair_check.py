"""Device-pair parity: are two devices the same instrument within the band?

PROTOCOL: cross-device comparisons are VOID unless the parity gate covered
that pair. Each device first passes its own gate vs the MLX reference
(scripts/parity_check.py); this script then compares the two stored parity
arms DIRECTLY -- pair covered iff |mean_a - mean_b| <= the registered band.
Journaled either way; FAIL exits 3 and any chain mixing the devices must
stop."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akademeia.journal import Journal

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs=2, default=["cpu", "cuda"])
    args = ap.parse_args()
    ref = json.load(open(ROOT / "reference" / "mlx_tier1.json"))
    band = ref["noise_final"]
    arms = {}
    for d in args.devices:
        p = ROOT / "runs" / f"arm-parity-direct50-{d}.json"
        if not p.exists():
            print(f"FATAL: {p.name} missing -- run parity_check --device {d} "
                  f"first"); sys.exit(4)
        arms[d] = json.loads(p.read_text())
        if arms[d].get("device") != d:
            print(f"FATAL: {p.name} journal device mismatch"); sys.exit(4)
    a, b = args.devices
    mean = {d: sum(arms[d]["final"]) / len(arms[d]["final"]) for d in args.devices}
    diff = mean[a] - mean[b]
    verdict = "PASS" if abs(diff) <= band else "FAIL"
    row = {"kind": "parity_pair", "devices": args.devices,
           "means": {d: round(m, 4) for d, m in mean.items()},
           "finals": {d: arms[d]["final"] for d in args.devices},
           "diff": round(diff, 4), "band": band, "verdict": verdict,
           "consequence": (f"{a}<->{b} comparisons legal with the band as "
                           f"caveat" if verdict == "PASS" else
                           f"{a}<->{b} comparisons VOID; one device must own "
                           f"any chained claim end to end")}
    Journal(ROOT / "runs" / "journal.jsonl").append(row)
    print(json.dumps(row, indent=1))
    print(f"[parity-pair {a}<->{b}] {verdict}: {mean[a]:.4f} vs {mean[b]:.4f} "
          f"(band {band})")
    if verdict == "FAIL":
        sys.exit(3)

if __name__ == "__main__":
    main()
