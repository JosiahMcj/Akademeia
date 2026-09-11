from akademeia.journal import Journal

def test_append_only_and_readback(tmp_path):
    j = Journal(tmp_path / "runs.jsonl")
    j.append({"kind": "run", "run_id": "r1"})
    j.append({"kind": "verdict", "run_id": "r1", "promote": False})
    rows = j.read()
    assert [r["kind"] for r in rows] == ["run", "verdict"]
    assert all("ts" in r for r in rows)
