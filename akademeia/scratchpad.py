"""Q4 scratchpad DSL (registered 2026-08-11, PROTOCOL amendment of the same
date). The depth wall's remaining escape is task-side: let the model emit the
intermediate products before the answer, and train/eval that format as its
own niche under the same held-out laws.

This module is deliberately STANDALONE: nothing in the package imports it
(test-enforced), so it cannot perturb the ladder machinery. Wiring it into
FAMILIES/pool/bench is a separate, dated amendment gated on a wave-1 call
AND a preflight inside the discrimination band.

Registered design, each rule test-enforced:
  - Format: input `E=a*b*c;a=3;b=5;c=2`, output `15;30$` -- the ';'-separated
    left-fold partials, last partial = the answer. ZERO new characters
    against the frozen tokenizer.
  - Grading: FULL-TRACE exact match. A wrong intermediate with a lucky final
    answer grades False; the steps are the point of the format.
  - Held-out law: bench chains are NOVEL COMBINATIONS of trained symbols at
    trained lengths (never new primitives); length-4 chains are an
    extrapolation diagnostic, reported and never gated (the heldrange
    precedent).
  - The claim is what externalizing intermediates buys. It is NEVER a
    comparison smuggled against the plain format: the tasks differ.
"""
from dataclasses import dataclass
import random

from akademeia.data.families import _PMAG

RESULT = "E"
# registered symbol alphabet; every emitted char must exist in the frozen
# tokenizer (test-enforced against freeze/tokenizer.json)
VARS = ("a", "b", "c", "d", "m", "n")

@dataclass(frozen=True)
class ChainTemplate:
    template_id: str
    variables: tuple      # ordered symbol tuple; expr is E=v1*v2*...
    split: str            # train | bench | extrap

    @property
    def expr(self) -> str:
        return f"{RESULT}=" + "*".join(self.variables)

CHAIN_TEMPLATES = (
    ChainTemplate("sp-ab",   ("a", "b"), "train"),
    ChainTemplate("sp-abc",  ("a", "b", "c"), "train"),
    ChainTemplate("sp-mnc",  ("m", "n", "c"), "train"),
    ChainTemplate("sp-bcd",  ("b", "c", "d"), "train"),
    # bench: novel combinations of trained symbols at trained lengths
    ChainTemplate("sp-acd",  ("a", "c", "d"), "bench"),
    ChainTemplate("sp-mbn",  ("m", "b", "n"), "bench"),
    # extrapolation diagnostic: a length no training chain has
    ChainTemplate("sp-abcd", ("a", "b", "c", "d"), "extrap"),
)

def gold_trace(vals: dict, variables) -> str:
    """Left-fold partial products, ';'-separated; the last IS the answer."""
    acc = vals[variables[0]]
    steps = []
    for v in variables[1:]:
        acc *= vals[v]
        steps.append(str(acc))
    return ";".join(steps)

def render_chain(t: ChainTemplate, vals: dict) -> tuple:
    text_in = f"{t.expr};" + ";".join(f"{v}={vals[v]}" for v in t.variables)
    return text_in, gold_trace(vals, t.variables)

def gen_item(t: ChainTemplate, r: random.Random, difficulty: int) -> tuple:
    lo, hi = _PMAG[difficulty]
    vals = {v: r.randint(lo, hi) for v in t.variables}
    return render_chain(t, vals)

def trace_eval_independent(text_in: str, text_out: str) -> bool:
    """Independent grader: reparse the input, recompute the fold, require
    EVERY intermediate to match."""
    try:
        head, *assigns = text_in.split(";")
        lhs, rhs = head.split("=", 1)
        variables = rhs.split("*")
        vals = {}
        for a in assigns:
            k, v = a.split("=", 1)
            vals[k] = int(v)
        if lhs != RESULT:
            return False
        if len(variables) != len(vals) or set(variables) != set(vals):
            return False
        return text_out == gold_trace(vals, tuple(variables))
    except (ValueError, KeyError):
        return False

def split_law_ok(templates=CHAIN_TEMPLATES) -> bool:
    """Bench = novel combinations of trained symbols at trained lengths;
    extrap is the only split allowed a new length, still no new symbols."""
    train = [t for t in templates if t.split == "train"]
    train_syms = {s for t in train for s in t.variables}
    train_lens = {len(t.variables) for t in train}
    train_tuples = {t.variables for t in train}
    for t in templates:
        if t.split == "bench":
            if (set(t.variables) - train_syms
                    or len(t.variables) not in train_lens
                    or t.variables in train_tuples):
                return False
        elif t.split == "extrap":
            if set(t.variables) - train_syms or len(t.variables) in train_lens:
                return False
    return True
