"""One screen: what the machine is doing, what it has measured, what is next.

Reads only durable state (PID files, logs, journal, freeze manifest). Never
infers "running" from a process-name match -- liveness is a PID check,
because a name match self-matches the checker and once hid a dead chain for
46 hours."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akademeia.ops import box_ok

ROOT = Path(__file__).resolve().parents[1]

def pid_alive(pid: str) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False

def main():
    runs = ROOT / "runs"
    print("== chains ==")
    pids = sorted(runs.glob("*.pid")) if runs.exists() else []
    if not pids:
        print("  none registered")
    for pf in pids:
        pid = pf.read_text().strip()
        state = "ALIVE" if pid_alive(pid) else "DEAD (stale pidfile)"
        print(f"  {pf.stem}: pid {pid} {state}")
        log = runs / f"{pf.stem}.log"
        if log.exists():
            tail = log.read_text().splitlines()[-3:]
            for line in tail:
                print(f"    | {line}")
    print("== box ==")
    ok, why = box_ok(probes=1, gap=0)
    print(f"  {'ok' if ok else 'BUSY'}: {why}")
    print("== freeze ==")
    mpath = ROOT / "freeze" / "manifest.json"
    if mpath.exists():
        m = json.load(open(mpath))
        for k in sorted(m):
            if "_hash" in k:
                print(f"  {k}: {m[k]}")
    print("== journal (last 5) ==")
    jpath = runs / "journal.jsonl"
    if jpath.exists():
        for line in jpath.read_text().splitlines()[-5:]:
            row = json.loads(line)
            keys = [row.get("kind", "?")] + [
                str(row[k]) for k in ("rung", "tag", "arm", "study", "label")
                if k in row]
            print("  " + " ".join(keys))
    else:
        print("  no journal yet")

if __name__ == "__main__":
    main()
