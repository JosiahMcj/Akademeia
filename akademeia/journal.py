"""Append-only JSONL journal (PROTOCOL R1: every run, failure, and void).

Concurrency law (2026-09-21): O_APPEND is NOT atomic on NFS -- the client
does a read-modify-write of the file offset, so simultaneous appends from
different nodes can interleave or silently drop rows. The journal is the
evidence, so under a scheduler each allocation writes its OWN shard and the
shards are merged by a single process afterwards. No locking is involved,
because a lock on the same shared filesystem has the same defect.
"""
import json, os, time
from pathlib import Path

class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        job = os.environ.get("SLURM_JOB_ID")
        if job:
            shard_dir = self.path.parent / "shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            task = os.environ.get("SLURM_ARRAY_TASK_ID")
            name = f"{self.path.stem}-{job}" + (f"-{task}" if task else "")
            self.write_path = shard_dir / f"{name}.jsonl"
        else:
            self.write_path = self.path

    def append(self, row: dict) -> None:
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
        with open(self.write_path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())      # the row is evidence; get it on disk

    def read(self) -> list:
        """The whole record: the merged journal plus any unmerged shards."""
        rows = []
        for p in [self.path] + sorted((self.path.parent / "shards").glob("*.jsonl")
                                      if (self.path.parent / "shards").exists()
                                      else []):
            if p.exists():
                for line in open(p):
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        return rows

def merge_shards(path, skip=()) -> int:
    """Fold shards into the main journal. Single process, serial, by design:
    this is the one writer that may touch the merged file.

    Idempotent, which it has to be: a shard pulled from a remote site is
    deleted HERE after merging, but the remote keeps its copy, so the next
    sync downloads it again. Merging blindly duplicated 141 rows into the
    evidence ledger before this was caught. Two defences, because one was
    evidently not enough:
      - `skip`: shard filenames still being written by a live job, which
        must not be folded in until their job is done.
      - a content check: any row already present in the journal is not
        appended again.
    """
    path = Path(path)
    shard_dir = path.parent / "shards"
    if not shard_dir.exists():
        return 0
    existing = set()
    if path.exists():
        existing = {l.strip() for l in open(path) if l.strip()}
    shards = [p for p in sorted(shard_dir.glob("*.jsonl"))
              if p.name not in set(skip)]
    rows, merged_files = [], []
    for sp in shards:
        got = False
        for line in open(sp):
            line = line.strip()
            if line and line not in existing:
                rows.append((json.loads(line).get("ts", ""), line))
                existing.add(line)
                got = True
        merged_files.append(sp)
    if rows:
        rows.sort(key=lambda r: r[0])
        with open(path, "a") as f:
            for _, line in rows:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
    for sp in merged_files:
        sp.unlink()
    return len(rows)
