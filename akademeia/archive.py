"""Mach-style archive. Admission (>= parent, paired) is NOT success and is
never reported as improvement; only full-gate promotion counts (PROTOCOL)."""
import json
from pathlib import Path

class Archive:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = ([json.loads(l) for l in open(self.path)]
                     if self.path.exists() else [])
    def admit(self, name: str, parent: str, gene: str, metrics: dict) -> None:
        row = {"name": name, "parent": parent, "gene": gene, "metrics": metrics}
        self.rows.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
    def parents(self, k: int = 3) -> list:
        """Top-k by mean final niche score."""
        return sorted(self.rows,
                      key=lambda r: -r["metrics"].get("final_mean", 0.0))[:k]
