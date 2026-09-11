"""Learning-progress allocation (Phase B, V2-Q2).

Weight families by how fast they are still improving on the dev probes:
w_f is proportional to floor + max(0, delta score since last checkpoint),
then capped so no family can monopolize the mixture.

Guards are registered design, not tuning knobs:
  - FLOOR: every steered family keeps a slice forever. v1 measured that
    starving the basics destroys them (0.73 -> 0.017).
  - CAP: no steered family may exceed `cap` of total mass, so one fast mover
    cannot collapse the mixture into the single-family shape v1 proved
    catastrophic.
  - FIXED shares (Amendment 8, arm 2): families whose share is pinned and
    never steered -- e.g. the niche at 50%, identical to the direct50
    baseline, so allocation acts only on the complement. Arm 1 measured why
    this exists: greedy progress-chasing starved the slow-starting niche to
    0.12 vs 0.59 (cold-start loop: no tokens -> no progress -> no tokens).

Pure function of probe scores: same scores in, same weights out.
"""

class LearningProgressScheduler:
    def __init__(self, families, floor: float = 0.02, cap: float = 0.35,
                 fixed: dict | None = None):
        if not families:
            raise ValueError("no families")
        self.fixed = dict(fixed or {})
        for f, share in self.fixed.items():
            if f not in families:
                raise ValueError(f"fixed family {f!r} not in families")
            if not (0 < share < 1):
                raise ValueError(f"fixed share for {f!r} out of (0,1)")
        self.mass = 1.0 - sum(self.fixed.values())
        if self.mass <= 0:
            raise ValueError("fixed shares consume all mass")
        self.steered = tuple(sorted(f for f in families if f not in self.fixed))
        if not self.steered:
            raise ValueError("no steered families left")
        if not (0 < floor < 1) or not (0 < cap <= 1):
            raise ValueError("floor/cap out of range")
        if cap * len(self.steered) < self.mass:
            raise ValueError("cap too small to distribute the steered mass")
        self.families = tuple(sorted(families))
        self.floor, self.cap = floor, cap
        self.prev = None
        self.history = []

    def _uniform(self) -> dict:
        w = self.mass / len(self.steered)
        return {**self.fixed, **{f: w for f in self.steered}}

    def current(self) -> dict:
        return self.history[-1] if self.history else self._uniform()

    def update(self, scores: dict) -> dict:
        """scores: family -> probe score in [0,1] (must cover every steered
        family; fixed families' scores are recorded by the caller but do not
        steer). Returns the new weights, fixed shares held exactly."""
        missing = set(self.steered) - set(scores)
        if missing:
            raise ValueError(f"probe scores missing families: {sorted(missing)}")
        if self.prev is None:
            self.prev = {f: scores[f] for f in self.steered}
            w = self._uniform()
            self.history.append(w)
            return w
        raw = {f: self.floor + max(0.0, scores[f] - self.prev[f])
               for f in self.steered}
        self.prev = {f: scores[f] for f in self.steered}
        steered_w = self._normalize_with_cap(raw)
        w = {**self.fixed, **steered_w}
        self.history.append(w)
        return w

    def _normalize_with_cap(self, raw: dict) -> dict:
        """Normalize steered weights to self.mass, then water-fill: clip at cap
        and redistribute the excess among uncapped families."""
        total = sum(raw.values())
        w = {f: v / total * self.mass for f, v in raw.items()}
        for _ in range(len(self.steered)):
            over = {f: v for f, v in w.items() if v > self.cap + 1e-12}
            if not over:
                break
            excess = sum(v - self.cap for v in over.values())
            free = [f for f in w if f not in over]
            if not free:
                break
            free_mass = sum(w[f] for f in free)
            for f in over:
                w[f] = self.cap
            for f in free:
                share = (w[f] / free_mass) if free_mass > 0 else 1.0 / len(free)
                w[f] += excess * share
        return w
