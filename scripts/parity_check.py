"""Port gate 2: golden parity between substrates (PROTOCOL).

Runs ONE tier-1 direct50-physics arm (3 seeds, salt 0) on the torch substrate
and compares its mean final against the stored MLX reference. PASS iff the
difference sits within v2's registered A/A' band (noise_final). Either
outcome is journaled: PASS makes cross-version comparisons legal with the
band as the caveat; FAIL makes v3 its own baseline era, never comparing
backward. Run per device that will host campaigns (cpu now, cuda on the
campaign box) -- the verdict binds only the device it ran on."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from akademeia.curriculum import validate
from akademeia.journal import Journal
from akademeia.tiers import TIER1
from run_arm import precision, HOST, NODE, run_curriculum
from ladder import direct50

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    ref = json.load(open(ROOT / "reference" / "mlx_tier1.json"))
    manifest = json.load(open(ROOT / "freeze" / "manifest.json"))
    for k, v in ref["hashes"].items():
        key = "bench_hash_physics" if k == "bench" else f"{k}_hash"
        if manifest[key] != v:
            print(f"FATAL: frozen {key} differs from the reference's -- the "
                  f"instruments are not v2's; parity is undefined")
            sys.exit(4)
    B = TIER1.total_tokens
    c = direct50("physics", B); validate(c, B)
    arm = run_curriculum(TIER1, c, "physics", stream_salt=0,
                         tag=f"parity-direct50-{args.device}",
                         phase="parity", device=args.device)
    mean_t = sum(arm["final"]) / len(arm["final"])
    mean_m = ref["mean_final_A"]
    diff = mean_t - mean_m
    band = ref["noise_final"]
    verdict = "PASS" if abs(diff) <= band else "FAIL"
    mean_tb = sum(arm["basics"]) / len(arm["basics"])
    mean_mb = sum(ref["basics_A"]) / len(ref["basics_A"])
    row = {"kind": "parity", "device": args.device,
           "precision": precision(), "host": HOST, "node": NODE,
           "torch_final": arm["final"], "mlx_final": ref["final_A"],
           "torch_mean": round(mean_t, 4), "mlx_mean": round(mean_m, 4),
           "diff": round(diff, 4), "band": band, "verdict": verdict,
           "basics_diff_advisory": round(mean_tb - mean_mb, 4),
           "consequence": ("cross-version comparisons legal with the band "
                           "as caveat" if verdict == "PASS" else
                           "v3 is its own baseline era; never compares "
                           "backward")}
    Journal(ROOT / "runs" / "journal.jsonl").append(row)
    print(json.dumps(row, indent=1))
    print(f"[parity {args.device}] {verdict}: torch {mean_t:.4f} vs "
          f"mlx {mean_m:.4f} (band {band})")
    if verdict == "FAIL":
        sys.exit(3)

if __name__ == "__main__":
    main()
