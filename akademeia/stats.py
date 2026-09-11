"""Paired measurement (PROTOCOL): margins from paired-difference noise; power
from discordant pairs; sign test is exact binomial, one-sided."""
from math import comb

def tokens_to_threshold(curve, T):
    """curve: [(tokens_seen, score)]; earliest checkpoint with score >= T."""
    for tokens, score in curve:
        if score >= T:
            return tokens
    return None

def aulc(curve) -> float:
    """Trapezoidal area under the (tokens, score) curve, normalized to [0,1]."""
    if len(curve) < 2:
        return curve[0][1] if curve else 0.0
    area, span = 0.0, curve[-1][0] - curve[0][0]
    for (x1, y1), (x2, y2) in zip(curve, curve[1:]):
        area += (x2 - x1) * (y1 + y2) / 2
    return area / span if span else curve[-1][1]

def discordant(arm_items, base_items):
    """Item-level paired outcomes -> (wins, losses). Ties carry no information."""
    wins = sum(1 for a, b in zip(arm_items, base_items, strict=True) if a and not b)
    losses = sum(1 for a, b in zip(arm_items, base_items, strict=True) if b and not a)
    return wins, losses

def sign_test(wins: int, losses: int) -> float:
    """One-sided exact binomial: P(X >= wins | n = wins+losses, p = .5)."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n

def tts_from_curves(curves: dict, T: float) -> list:
    """Per-seed tokens-to-T recomputed from stored curves {seed: [[tokens,
    niche, basics], ...]}. ALWAYS use this at gate time -- stored tokens_to_T
    reflect whatever T the arm was assembled with (the P3 rejudge lesson)."""
    return [tokens_to_threshold([(r[0], r[1]) for r in curves[s]], T)
            for s in sorted(curves)]

def paired_noise(paired_diffs) -> float:
    """Conservative scale of paired-difference noise from A/A' runs: max |diff|.
    Margins are set at 2-3x this (PROTOCOL)."""
    return max(abs(d) for d in paired_diffs)

def powered(wins: int, losses: int, min_discordant: int = 8) -> bool:
    return (wins + losses) >= min_discordant
