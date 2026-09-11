"""Char tokenizer fit on POOL TEXT ONLY (mach memorizing-tokenizer guard).
Bench coverage is asserted separately and hard-exits on miss."""
import json, sys
from akademeia.config import content_hash

class CharTokenizer:
    def __init__(self, vocab: str):
        self.vocab = vocab
        self._to_id = {c: i for i, c in enumerate(vocab)}
    @classmethod
    def fit(cls, texts) -> "CharTokenizer":
        return cls("".join(sorted(set("".join(texts)))))
    @property
    def vocab_size(self) -> int: return len(self.vocab)
    def hash(self) -> str: return content_hash(self.vocab)
    def encode(self, s: str) -> list[int]:
        try:
            return [self._to_id[c] for c in s]
        except KeyError as e:
            raise ValueError(f"OOV char {e.args[0]!r}") from None
    def decode(self, ids) -> str: return "".join(self.vocab[i] for i in ids)
    def assert_covers(self, texts) -> None:
        missing = set("".join(texts)) - set(self.vocab)
        if missing:
            print(f"FATAL: bench chars not in pool vocab: {sorted(missing)}", file=sys.stderr)
            raise SystemExit(2)
    def save(self, path) -> None:
        json.dump({"vocab": self.vocab}, open(path, "w"))
    @classmethod
    def load(cls, path) -> "CharTokenizer":
        return cls(json.load(open(path))["vocab"])
