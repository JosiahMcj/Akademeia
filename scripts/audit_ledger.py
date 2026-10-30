import json, sys
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from akademeia.journal import Journal

# NOTE: this script deliberately reads the raw journal file as well as going
# through Journal.read(). The raw read is how duplicate LINES are detected --
# a merge-and-deduplicate reader would conceal the very corruption this audit
# exists to find. tests/test_journal_concurrency.py exempts this file by name.
rows = Journal("runs/journal.jsonl").read()
print("=== ledger ===")
print(f"  rows: {len(rows)} | unique lines: "
      f"{len(set(open('runs/journal.jsonl').read().splitlines()))}")
kinds = {}
for r in rows:
    kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
print(f"  {kinds}")

print("\n=== every verdict traces to arms that exist, with matching provenance ===")
bad = []
for r in rows:
    if r.get("kind") != "verdict":
        continue
    for tag in (f"ladder-r{r['rung']}-{r['arm']}", r["baseline"]):
        p = Path(f"runs/arm-{tag}.json")
        if not p.exists():
            bad.append(f"MISSING {tag}"); continue
        a = json.loads(p.read_text())
        if not a.get("hashes") or not a.get("device"):
            bad.append(f"NO PROVENANCE {tag}")
print(f"  {'OK' if not bad else bad}")

print("\n=== arms compared within a verdict share instrument + hashes ===")
mism = []
for r in rows:
    if r.get("kind") != "verdict":
        continue
    a = json.loads(Path(f"runs/arm-ladder-r{r['rung']}-{r['arm']}.json").read_text())
    b = json.loads(Path(f"runs/arm-{r['baseline']}.json").read_text())
    for k in ("hashes", "device", "host", "precision"):
        if a.get(k) != b.get(k):
            mism.append(f"r{r['rung']} {r['arm']}: {k} {a.get(k)} != {b.get(k)}")
print(f"  {'OK -- every comparison is same-instrument' if not mism else mism}")

print("\n=== instruments used, and their gates ===")
inst = {}
for r in rows:
    if r.get("kind") == "run":
        key = (r.get("device"), r.get("host") or r.get("node"), r.get("precision"))
        inst[key] = inst.get(key, 0) + 1
for k, v in sorted(inst.items(), key=lambda x: -x[1]):
    print(f"  {str(k):<52} {v:>4} runs")
print("  parity gates:")
for r in rows:
    if r.get("kind") in ("parity", "parity_pair"):
        print(f"    {r['kind']:<12} {str(r.get('device') or r.get('devices')):<14}"
              f" {str(r.get('precision','-')):<5} -> {r['verdict']}")

print("\n=== frozen thresholds present per rung ===")
for p in sorted(Path("freeze").glob("thresholds_*.json")):
    print(f"  {p.name}: {json.loads(p.read_text())}")
