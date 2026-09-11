"""Append-only JSONL journal (PROTOCOL R1: every run, failure, and void)."""
import json, time
from pathlib import Path

class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def append(self, row: dict) -> None:
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
    def read(self) -> list:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in open(self.path)]
