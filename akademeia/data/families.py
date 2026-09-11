"""Task families. Golds are correct BY CONSTRUCTION; tests cross-check them
independently. Integer-only payloads; '$' is the terminator, banned in payloads."""
from dataclasses import dataclass
from typing import Callable
import random

@dataclass(frozen=True)
class Example:
    family: str
    difficulty: int
    template_id: str
    text_in: str
    text_out: str

def render(ex: Example) -> str: return f"I:{ex.text_in};O:{ex.text_out}$"
def prompt(ex: Example) -> str: return f"I:{ex.text_in};O:"

@dataclass(frozen=True)
class Family:
    name: str
    kind: str                       # "basic" | "niche"
    gen: Callable                   # (rng, difficulty, split="train") -> Example
    difficulties: tuple = (1, 2, 3)

# ---------- basics ----------
_LC = "abcdefghijklmnopqrstuvwxyz"

def _word(r, n): return "".join(r.choice(_LC) for _ in range(n))
def _digits(r, n): return "".join(r.choice("0123456789") for _ in range(n))
_LEN = {1: (3, 5), 2: (6, 10), 3: (11, 16)}     # payload length by difficulty
_MAG = {1: (0, 9), 2: (10, 99), 3: (100, 999)}  # operand magnitude by difficulty

def _mk(name):
    def deco(fn):
        def gen(r, d, split="train"):
            ti, to = fn(r, d)
            return Example(name, d, f"{name}-t0", ti, to)
        return Family(name, "basic", gen)
    return deco

@_mk("copy")
def _copy(r, d):
    s = _word(r, r.randint(*_LEN[d])); return f"copy {s}", s

@_mk("reverse")
def _reverse(r, d):
    s = _word(r, r.randint(*_LEN[d])); return f"rev {s}", s[::-1]

@_mk("sortdigits")
def _sort(r, d):
    s = _digits(r, r.randint(*_LEN[d])); return f"sort {s}", "".join(sorted(s))

@_mk("countchar")
def _count(r, d):
    s = _word(r, r.randint(*_LEN[d])); c = r.choice(s)
    return f"count {c} in {s}", str(s.count(c))

@_mk("compare")
def _cmp(r, d):
    a, b = r.randint(*_MAG[d]), r.randint(*_MAG[d])
    return f"max {a} {b}", str(max(a, b))

@_mk("addchain")
def _add(r, d):
    n = d + 1
    terms = [str(r.randint(*_MAG[d]))] + [
        r.choice("+-") + str(r.randint(*_MAG[d])) for _ in range(n)]
    expr = "".join(terms); return expr, str(eval(expr))

@_mk("modarith")
def _mod(r, d):
    a, m = r.randint(*_MAG[d]), r.randint(2, 9 + d * 3)
    return f"{a} mod {m}", str(a % m)

@_mk("substitute")
def _sub(r, d):
    s = _word(r, r.randint(*_LEN[d])); a = r.choice(s); b = r.choice(_LC)
    return f"swap {a}>{b} in {s}", s.replace(a, b)

@_mk("brackets")
def _br(r, d):
    n = r.randint(*_LEN[d]); s = "".join(r.choice("()") for _ in range(n))
    depth = bal = 0; ok = True
    for ch in s:
        bal += 1 if ch == "(" else -1
        if bal < 0: ok = False
        depth = max(depth, bal)
    out = str(depth) if (ok and bal == 0) else "no"
    return f"depth {s}", out

@_mk("statetrack")
def _st(r, d):
    n = r.randint(*_LEN[d]); x = r.randint(0, 9)
    moves = "".join(r.choice("ud") for _ in range(n))
    v = x + moves.count("u") - moves.count("d")
    return f"start {x} {moves}", str(v)

# ---------- niche: physics DSL ----------
@dataclass(frozen=True)
class PhysicsTemplate:
    template_id: str
    expr: str                      # e.g. "F=m*a"
    variables: tuple               # e.g. ("m","a")
    fn: Callable                   # ints -> int
    split: str                     # "train" | "bench"

PHYSICS_TEMPLATES = (
    PhysicsTemplate("ph-force",    "F=m*a",       ("m","a"),     lambda m,a: m*a, "train"),
    PhysicsTemplate("ph-ohm",      "V=i*r",       ("i","r"),     lambda i,r: i*r, "train"),
    PhysicsTemplate("ph-momentum", "p=m*v",       ("m","v"),     lambda m,v: m*v, "train"),
    PhysicsTemplate("ph-weight",   "W=m*g",       ("m","g"),     lambda m,g: m*g, "train"),
    PhysicsTemplate("ph-distance", "d=v*t",       ("v","t"),     lambda v,t: v*t, "train"),
    PhysicsTemplate("ph-work",     "E=f*d",       ("f","d"),     lambda f,d: f*d, "train"),
    PhysicsTemplate("ph-power",    "P=e/t",       ("e","t"),     lambda e,t: e//t, "train"),
    PhysicsTemplate("ph-density",  "r=m/v",       ("m","v"),     lambda m,v: m//v, "train"),
    PhysicsTemplate("ph-pressure", "P=f/a",       ("f","a"),     lambda f,a: f//a, "train"),
    PhysicsTemplate("ph-speed",    "v=d/t",       ("d","t"),     lambda d,t: d//t, "train"),
    PhysicsTemplate("ph-accel",    "a=(v-u)/t",   ("v","u","t"), lambda v,u,t: (v-u)//t, "train"),
    PhysicsTemplate("ph-ke",       "K=m*v*v/2",   ("m","v"),     lambda m,v: m*v*v//2, "train"),
    PhysicsTemplate("ph-spring",   "F=k*x",       ("k","x"),     lambda k,x: k*x, "train"),
    PhysicsTemplate("ph-charge",   "Q=i*t",       ("i","t"),     lambda i,t: i*t, "train"),
    PhysicsTemplate("ph-heat",     "Q=m*c*dT",    ("m","c","dT"),lambda m,c,dT: m*c*dT, "train"),
    PhysicsTemplate("ph-freq",     "f=1000/T",    ("T",),        lambda T: 1000//T, "train"),
    PhysicsTemplate("ph-wave",     "v=f*L",       ("f","L"),     lambda f,L: f*L, "train"),
    PhysicsTemplate("ph-torque",   "T=f*r",       ("f","r"),     lambda f,r: f*r, "train"),
    # bench-only templates: the model NEVER trains on these expression strings
    # result symbols must exist in pool vocab (pool-only tokenizer law);
    # expressions stay unseen as strings
    PhysicsTemplate("ph-pe",       "E=m*g*h",     ("m","g","h"), lambda m,g,h: m*g*h, "bench"),
    PhysicsTemplate("ph-impulse",  "p=f*t",       ("f","t"),     lambda f,t: f*t, "bench"),
    PhysicsTemplate("ph-flow",     "Q=a*v",       ("a","v"),     lambda a,v: a*v, "bench"),
    PhysicsTemplate("ph-current",  "i=q/t",       ("q","t"),     lambda q,t: q//t, "bench"),
    PhysicsTemplate("ph-gravity",  "F=g*m*n/100", ("g","m","n"), lambda g,m,n: g*m*n//100, "bench"),
    PhysicsTemplate("ph-wavelen",  "L=v/f",       ("v","f"),     lambda v,f: v//f, "bench"),
)

# Parameter ranges: train draws lo..hi; bench "heldrange" draws above the hi.
_PMAG = {1: (2, 9), 2: (2, 20), 3: (2, 40)}

def _physics_gen(r: random.Random, d: int, split: str = "train",
                 exclude_division: bool = False,
                 pool_templates=None) -> Example:
    """pool_templates: alternate template set (Study S2 variety pools). The
    default path must stay byte-identical -- bench pins enforce it."""
    pool = [t for t in (pool_templates if pool_templates is not None
                        else PHYSICS_TEMPLATES) if t.split == split]
    if exclude_division:
        # Within a narrow operand band the only EXACT divisions are x/x, so
        # division templates collapse to the constant 1: 18 of 60 heldrange
        # golds were "1" (30%), handing a constant-predictor most of the score.
        pool = [t for t in pool if "/" not in t.expr]
    t = r.choice(pool)
    lo, hi = _PMAG[d]
    for _ in range(1000):
        vals = {v: r.randint(lo, hi) for v in t.variables}
        try:
            out = t.fn(*[vals[v] for v in t.variables])
        except ZeroDivisionError:
            continue
        # divisions must be exact so grading is exact-integer
        if _physics_check_exact(t, vals, out):
            binds = ";".join(f"{v}={vals[v]}" for v in t.variables)
            return Example("physics", d, t.template_id, f"{t.expr};{binds}", str(out))
    raise RuntimeError(f"could not instantiate {t.template_id}")

def _physics_check_exact(t: PhysicsTemplate, vals: dict, out: int) -> bool:
    return physics_eval_independent(
        f"{t.expr};" + ";".join(f"{v}={vals[v]}" for v in t.variables),
        require_exact=True) == out

def physics_eval_independent(text_in: str, require_exact: bool = False):
    """Independent gold check: parse and evaluate with fractions, no template fn."""
    from fractions import Fraction
    expr, *binds = text_in.split(";")
    env = {k: Fraction(int(v)) for k, v in (b.split("=") for b in binds)}
    rhs = expr.split("=", 1)[1]
    val = eval(rhs, {"__builtins__": {}}, dict(env))
    if isinstance(val, Fraction) and val.denominator != 1:
        if require_exact:
            return None  # inexact division -> reject instantiation
        val = val.numerator // val.denominator
    return int(val)

# ---------- niche 2: vm (stack-machine program evaluation) ----------
# Ladder matched to physics's ARITHMETIC load so the two niches differ in SKILL
# (stack simulation vs formula substitution), not in how hard the sums are.
# The first design chained 2-5 ops over operands up to 40 (e.g. 22*39*29*3) and
# measured 0.033 -- a floored bench that can measure nothing. See PROTOCOL
# Amendment 4. d -> (pushes, min_ops, max_ops); swap is depth-neutral, so
# d1/d2 carry exactly ONE real arithmetic operation.
_VM_LADDER = {1: (2, 1, 1), 2: (2, 1, 2), 3: (3, 2, 2)}
# d3 operands stay SMALL: two composed operations over 2-digit operands measured
# 0/40 even in-distribution, i.e. items no arm can solve are dead bench weight
# (the informative-task rule). Train max is therefore 20, at d2.
_VMMAG = {1: (2, 9), 2: (2, 20), 3: (2, 9)}
_VM_OPS = ("add", "sub", "mul", "dup", "swap")
_ARITH = ("add", "sub", "mul")
_VM_BENCH_FRACTION = 0.25

def _vm_legal(pushes: int, ops) -> bool:
    d = pushes
    for o in ops:
        if o == "dup":
            if d < 1: return False
            d += 1
        elif o == "swap":
            if d < 2: return False
        else:
            if d < 2: return False
            d -= 1
    return d == 1

def _vm_shape_space(pushes: int, n_ops: int) -> tuple:
    import itertools
    return tuple(c for c in itertools.product(_VM_OPS, repeat=n_ops)
                 if _vm_legal(pushes, c))

def _vm_bench_shapes() -> frozenset:
    """Held-out shapes: COMBINATIONS only, never a primitive.

    A held-out group must be a novel combination of primitives the model can
    learn elsewhere -- that is what physics does (its held-out formulas reuse
    trained operators). Stratifying by (pushes, n_ops) without that rule held
    out ('mul',) at the 1-op strata AND ('swap','mul') at the 2-op stratum,
    leaving multiplication learnable only inside 3-push programs the model
    solves 0/40. The model then scored 1.00 in-distribution at d1 and 0.03 on
    a bench full of multiplication: a split-design bug, not a difficulty wall.

    So: 1-op strata are never held out, and a combination is eligible only if
    every arithmetic primitive it uses is trainable at a 1-op stratum.
    """
    import hashlib, math
    trainable_primitives = set()
    for pushes, lo, hi in _VM_LADDER.values():
        for shape in _vm_shape_space(pushes, 1):
            trainable_primitives.update(o for o in shape if o in _ARITH)
    chosen = set()
    for pushes, lo, hi in _VM_LADDER.values():
        for n_ops in range(lo, hi + 1):
            if n_ops < 2:
                continue                     # never hold out a primitive
            space = [s for s in _vm_shape_space(pushes, n_ops)
                     if all(o in trainable_primitives for o in s if o in _ARITH)]
            if len(space) < 2:               # a stratum must keep train shapes
                continue
            ranked = sorted(space, key=lambda c: hashlib.sha256(
                ",".join(c).encode()).hexdigest())
            k = max(1, math.ceil(_VM_BENCH_FRACTION * len(space)))
            chosen.update(ranked[:k])
    return frozenset(chosen)

def _vm_shape(r: random.Random, d: int):
    pushes, lo, hi = _VM_LADDER[d]
    space = _vm_shape_space(pushes, r.randint(lo, hi))
    if not space:
        raise RuntimeError(f"no legal vm shape at d={d}")
    return space[r.randrange(len(space))]

def _vm_split(ops) -> str:
    return "bench" if tuple(ops) in _vm_bench_shapes() else "train"

def _vm_run(ops, vals):
    stack = list(vals)
    for op in ops:
        if op == "dup": stack.append(stack[-1])
        elif op == "swap": stack[-1], stack[-2] = stack[-2], stack[-1]
        else:
            b, a = stack.pop(), stack.pop()
            stack.append(a + b if op == "add" else a - b if op == "sub" else a * b)
    assert len(stack) == 1
    return stack[0]

def vm_eval_independent(text_in: str) -> int:
    """Independent cross-check: functional reduction, shares no code with the
    generator's interpreter."""
    from functools import reduce
    toks = text_in.split(";")
    assert toks[-1] == "out"
    def step(stack, tok):
        if tok.startswith("push "): return stack + (int(tok[5:]),)
        if tok == "dup": return stack + (stack[-1],)
        if tok == "swap": return stack[:-2] + (stack[-1], stack[-2])
        a, b = stack[-2], stack[-1]
        v = {"add": a + b, "sub": a - b, "mul": a * b}[tok]
        return stack[:-2] + (v,)
    final = reduce(step, toks[:-1], ())
    assert len(final) == 1
    return final[0]

def _vm_gen(r: random.Random, d: int, split: str = "train") -> Example:
    pushes, _, _ = _VM_LADDER[d]
    lo, hi = _VMMAG[d]
    for _ in range(500):
        ops = _vm_shape(r, d)
        if _vm_split(ops) != split:
            continue
        vals = [r.randint(lo, hi) for _ in range(pushes)]
        out = _vm_run(ops, vals)
        prog = ";".join([f"push {v}" for v in vals] + list(ops) + ["out"])
        return Example("vm", d, f"vm-{','.join(ops)}", prog, str(out))
    raise RuntimeError(f"no vm shape for split {split} at d={d}")

# Study S2 (Amendment 12): product-chain VARIETY templates. NOT part of
# PHYSICS_TEMPLATES -- they exist only in the separate S2 pool variant, so the
# frozen pool/bench path is untouched. Zero new characters; integer products.
S2_EXTRA_TEMPLATES = (
    PhysicsTemplate("ph3-volume",  "V=a*b*c",  ("a", "b", "c"),
                    lambda a, b, c: a * b * c, "train"),
    PhysicsTemplate("ph3-emc",     "E=m*c*c",  ("m", "c"),
                    lambda m, c: m * c * c, "train"),
    PhysicsTemplate("ph3-power3",  "P=i*i*r",  ("i", "r"),
                    lambda i, r: i * i * r, "train"),
    PhysicsTemplate("ph3-torque3", "T=f*r*n",  ("f", "r", "n"),
                    lambda f, r, n: f * r * n, "train"),
    PhysicsTemplate("ph3-work3",   "W=f*d*n",  ("f", "d", "n"),
                    lambda f, d, n: f * d * n, "train"),
    PhysicsTemplate("ph3-heat3",   "Q=m*L*n",  ("m", "L", "n"),
                    lambda m, L, n: m * L * n, "train"),
)

FAMILIES: dict[str, Family] = {}
for fam in [_copy, _reverse, _sort, _count, _cmp, _add, _mod, _sub, _br, _st]:
    FAMILIES[fam.name] = fam
FAMILIES["physics"] = Family("physics", "niche", _physics_gen)
FAMILIES["vm"] = Family("vm", "niche", _vm_gen)
BASIC_FAMILIES = tuple(n for n, f in FAMILIES.items() if f.kind == "basic")
NICHE_FAMILIES = ("physics", "vm")
