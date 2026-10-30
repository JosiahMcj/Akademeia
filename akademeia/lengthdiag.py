"""D2: classify HOW a scratchpad trace fails (PROTOCOL amendment 2026-09-25).

The taxonomy is registered before any trace exists. This module is pure --
no model, no I/O -- so it can be proven against synthetic examples of every
class before it is pointed at real output.
"""

CLASSES = ("CORRECT", "SHORT_PREFIX_CORRECT", "SHORT_PREFIX_WRONG",
           "RIGHT_LENGTH_WRONG_VALUE", "LONG", "MALFORMED")

def parse_trace(text: str):
    """';'-separated run of integers, or None if it is not one."""
    if text is None:
        return None
    t = text.strip()
    if t == "":
        return None
    parts = t.split(";")
    out = []
    for p in parts:
        p = p.strip()
        if not p or not (p.lstrip("-").isdigit()):
            return None
        out.append(int(p))
    return out

def classify(emitted: str, gold: str) -> str:
    """Exactly one class per (emitted, gold) pair."""
    g = parse_trace(gold)
    e = parse_trace(emitted)
    if g is None:
        raise ValueError(f"gold is not a trace: {gold!r}")
    if e is None:
        return "MALFORMED"
    if e == g:
        return "CORRECT"
    if len(e) > len(g):
        return "LONG"
    if len(e) == len(g):
        return "RIGHT_LENGTH_WRONG_VALUE"
    # shorter than required: is what it did emit a correct prefix?
    return ("SHORT_PREFIX_CORRECT" if e == g[:len(e)]
            else "SHORT_PREFIX_WRONG")

def summarise(rows) -> dict:
    """rows: iterable of (emitted, gold). Returns counts + the registered
    60% majority rule applied, without interpreting it."""
    counts = {c: 0 for c in CLASSES}
    for emitted, gold in rows:
        counts[classify(emitted, gold)] += 1
    n = sum(counts.values())
    top, top_n = max(counts.items(), key=lambda kv: kv[1]) if n else (None, 0)
    return {"n": n, "counts": counts,
            "majority": top if n and top_n / n >= 0.60 else None,
            "majority_frac": round(top_n / n, 4) if n else None}
