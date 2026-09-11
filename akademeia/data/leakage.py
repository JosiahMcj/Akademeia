"""Leakage law (spec section 5, PROTOCOL Amendment 1): HARD EXIT on contact.
A printed warning does not count.

- No bench item's full rendered text may appear in the pool (memorized-Q+A
  channel, all sections).
- niche_unseen template expressions may not appear anywhere in pool text
  (template-level held-out).
"""
import sys

def check_leakage(pool_texts, bench_texts, unseen_exprs=()) -> None:
    pool_set = set(pool_texts)
    item_hits = [t for t in bench_texts if t in pool_set]
    if item_hits:
        print(f"FATAL leakage: {len(item_hits)} bench items verbatim in pool, "
              f"e.g. {item_hits[:3]}", file=sys.stderr)
        raise SystemExit(3)
    corpus = "\x00".join(pool_texts)
    expr_hits = [e for e in unseen_exprs if e in corpus]
    if expr_hits:
        print(f"FATAL leakage: unseen-template expressions present in pool: "
              f"{expr_hits}", file=sys.stderr)
        raise SystemExit(3)
