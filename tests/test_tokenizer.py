import pytest
from akademeia.tokenizer import CharTokenizer

def test_roundtrip_and_determinism():
    t1 = CharTokenizer.fit(["I:rev ab;O:ba$", "I:max 3 5;O:5$"])
    t2 = CharTokenizer.fit(["I:max 3 5;O:5$", "I:rev ab;O:ba$"])  # order-free
    assert t1.vocab == t2.vocab and t1.hash() == t2.hash()
    s = "I:rev ab;O:ba$"
    assert t1.decode(t1.encode(s)) == s

def test_oov_hard_fails():
    t = CharTokenizer.fit(["abc"])
    with pytest.raises(ValueError, match="OOV"):
        t.encode("abcZ")

def test_assert_covers():
    t = CharTokenizer.fit(["abc$"])
    t.assert_covers(["cab$"])
    with pytest.raises(SystemExit):
        t.assert_covers(["xyz!"])

def test_save_load(tmp_path):
    t = CharTokenizer.fit(["I:copy q;O:q$"])
    p = tmp_path / "tok.json"; t.save(p)
    t2 = CharTokenizer.load(p)
    assert t2.vocab == t.vocab and t2.hash() == t.hash()
