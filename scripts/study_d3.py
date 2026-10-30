"""D3 runner (registered 2026-10-29): Q4 curriculum on the length-exposure
pool, A/A' over three seeds; then a descriptive pooled report on the held-out
(a,b,c,d) combination and the length-5 diagnostic."""
import argparse, json, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from akademeia.journal import Journal
from akademeia.tiers import RUNGS
from run_arm import run_curriculum
from study_q4 import q4_curriculum

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["arm", "report"])
    ap.add_argument("--rung", type=int, required=True)
    ap.add_argument("--salt", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    cfg = RUNGS[a.rung]
    if a.stage == "arm":
        nm = "A" if a.salt == 0 else "Aprime"
        run_curriculum(cfg, q4_curriculum(cfg.total_tokens), "d3",
                       tag=f"d3-r{a.rung}-{nm}", stream_salt=a.salt,
                       phase=f"D3-r{a.rung}", pool_file="pool_d3.jsonl",
                       pool_key="pool_d3_hash", device=a.device)
        return
    fin, ext = [], []
    for nm in ("A", "Aprime"):
        arm = json.loads((ROOT / "runs" / f"arm-d3-r{a.rung}-{nm}.json").read_text())
        fin += arm["final"]; ext += list(arm["heldrange"].values())
    m = st.mean(fin)
    label = ("EXPOSURE" if m >= 0.20 else "COMPOSITION" if m < 0.05
             else "NOT IDENTIFIED")
    row = {"kind": "study", "study": "D3", "rung": a.rung, "device": a.device,
           "heldout_abcd_runs": [round(x, 4) for x in fin],
           "heldout_abcd_mean": round(m, 4),
           "heldout_abcd_sem": round(st.stdev(fin) / len(fin) ** 0.5, 4),
           "len5_runs": ext, "len5_all_zero": all(float(x) == 0 for x in ext),
           "registered_reading": label,
           "baseline_q4_same_template": "0.000 in 36/36 (never trained on length 4)"}
    Journal(ROOT / "runs" / "journal.jsonl").append(row)
    print(json.dumps(row, indent=1))

if __name__ == "__main__":
    main()
