"""Mechanical gates (PROTOCOL). No human judgment inside.
v1 gate (G1-G5, time-to-T primary) kept for reference; v2 gate_v2 (H1-H4,
final-quality primary) is what Akademeia phases use."""
from dataclasses import dataclass
from akademeia.stats import sign_test, powered

@dataclass(frozen=True)
class Thresholds:
    margin_tokens: int        # G1: registered at P2 from paired noise
    noise_final: float        # G2 band
    noise_basics: float       # G3 band
    T: float                  # competence threshold (registered at P2)
    min_discordant: int       # G5 power floor

BUDGET_EXHAUSTED = float("inf")

def _tt(vals):
    return [BUDGET_EXHAUSTED if v is None else v for v in vals]

def gate(arm: dict, base: dict, th: Thresholds) -> dict:
    """arm/base: per-seed lists 'tokens_to_T', 'final', 'basics'; arm carries
    pooled item-level 'wins'/'losses' vs base and both carry 'hashes'."""
    if arm["hashes"] != base["hashes"]:
        return {"promote": False, "void": True, "underpowered": False,
                "gates": {}, "reason": "hash mismatch: comparison voided"}
    if any(v is None for v in base["tokens_to_T"]):
        # A baseline that "never reaches T" is defective input (P2 requires it
        # to), and mapping it to inf makes G1 trivially true for every arm --
        # the bug that voided the first P3 verdicts.
        return {"promote": False, "void": True, "underpowered": False,
                "gates": {}, "reason": "baseline tokens_to_T contains None: "
                "defective baseline input, comparison voided"}
    at, bt = _tt(arm["tokens_to_T"]), _tt(base["tokens_to_T"])
    diffs = [b - a for a, b in zip(at, bt)]           # positive = arm faster
    g1 = all(d >= th.margin_tokens for d in diffs)
    g2 = (sum(arm["final"]) / len(arm["final"])
          >= sum(base["final"]) / len(base["final"]) - th.noise_final)
    g3 = (sum(arm["basics"]) / len(arm["basics"])
          >= sum(base["basics"]) / len(base["basics"]) - th.noise_basics)
    g4 = True   # hashes matched above; purity/leakage guards hard-exit upstream
    under = not powered(arm["wins"], arm["losses"], th.min_discordant)
    g5 = (not under) and sign_test(arm["wins"], arm["losses"]) < 0.05
    gates = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5}
    return {"promote": all(gates.values()), "void": False,
            "underpowered": under, "gates": gates,
            "reason": ",".join(k for k, v in gates.items() if not v) or "all pass"}

@dataclass(frozen=True)
class ThresholdsV2:
    noise_final: float        # H2 margin, from A/A' under the v2 metric
    noise_basics: float       # H4 band (when gating)
    min_discordant: int       # H1 power floor

def _mean(xs): return sum(xs) / len(xs)

def gate_v2(arm: dict, base: dict, th: ThresholdsV2, mode: str = "superior",
            retention_gates: bool = False) -> dict:
    """H1-H4 (Akademeia PROTOCOL). mode='superior': arm must beat base.
    mode='noninferior': arm must not be worse (Phase A branch arms).
    retention_gates: H4 gates only for generalist claims; always reported."""
    for side, d in (("arm", arm), ("base", base)):
        if any(v is None for v in d["final"]) or not d["final"]:
            return {"promote": False, "void": True, "underpowered": False,
                    "gates": {}, "reason": f"defective {side} input: final"}
    if arm["hashes"] != base["hashes"]:
        return {"promote": False, "void": True, "underpowered": False,
                "gates": {}, "reason": "hash mismatch: comparison voided"}
    w, l = arm["wins"], arm["losses"]
    under = not powered(w, l, th.min_discordant)
    if mode == "superior":
        h1 = (not under) and sign_test(w, l) < 0.05
        h2 = _mean(arm["final"]) >= _mean(base["final"]) + th.noise_final
    elif mode == "noninferior":
        # not significantly WORSE, and mean within the noise band below base
        h1 = (not under) and sign_test(l, w) >= 0.05
        h2 = _mean(arm["final"]) >= _mean(base["final"]) - th.noise_final
    else:
        raise ValueError(f"unknown mode {mode}")
    h3 = True  # hashes + input shape checked above; upstream guards hard-exit
    retention_delta = _mean(arm["basics"]) - _mean(base["basics"])
    h4 = (retention_delta >= -th.noise_basics) if retention_gates else True
    gates = {"H1": h1, "H2": h2, "H3": h3, "H4": h4}
    return {"promote": all(gates.values()), "void": False, "mode": mode,
            "underpowered": under, "gates": gates,
            "retention_delta": round(retention_delta, 4),
            "reason": ",".join(k for k, v in gates.items() if not v) or "all pass"}
