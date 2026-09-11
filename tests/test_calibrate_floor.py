"""The discrimination floor must fail strictly by default and pass only
under the explicit advisory flag of the dated 2026-08-22 amendment.
A gate edit ships with proof it can still fail."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate import derive_v2

def make_pair(mixed_items: int, n: int = 60):
    """A/A' over 2 seeds: `mixed_items` items disagree across runs, the
    rest are unanimous (half solved, half failed). Finals in [0.20, 0.85]."""
    base = [i < n // 2 for i in range(n)]           # unanimous half/half
    flip = dict(base_items=None)
    runs = []
    for k in range(4):                              # 2 seeds x A/A'
        items = list(base)
        for j in range(mixed_items):
            items[j] = (k % 2 == 0)                 # flickers across runs
        runs.append(items)
    a = {"final_items": {"1": runs[0], "2": runs[1]},
         "final": [sum(runs[0]) / n, sum(runs[1]) / n],
         "basics": [0.5, 0.5]}
    ap = {"final_items": {"1": runs[2], "2": runs[3]},
          "final": [sum(runs[2]) / n, sum(runs[3]) / n],
          "basics": [0.5, 0.5]}
    return a, ap

def test_floor_fails_strictly_by_default():
    a, ap = make_pair(mixed_items=3)                # 3/60 < 10%
    with pytest.raises(SystemExit) as e:
        derive_v2(a, ap, min_mixed_frac=0.10)
    assert e.value.code == 7                        # the registered exit

def test_floor_advisory_proceeds_and_reports():
    a, ap = make_pair(mixed_items=3)
    out = derive_v2(a, ap, min_mixed_frac=0.10, floor_advisory=True)
    assert out["min_discordant"] == 8               # H1 power floor intact
    assert out["_report"]["mixed_items"] == "3/60"  # capacity statement

def test_floor_passes_above_threshold_without_flag():
    a, ap = make_pair(mixed_items=9)                # 9/60 >= 10%
    out = derive_v2(a, ap, min_mixed_frac=0.10)
    assert out["_report"]["mixed_items"] == "9/60"
