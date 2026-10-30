"""The journal is the evidence, so its writes must survive a parallel
campaign on a shared filesystem (2026-09-21 amendment)."""
import json, os, importlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import akademeia.journal as J

def test_shards_per_allocation_under_a_scheduler(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "555")
    importlib.reload(J)
    j = J.Journal(tmp_path / "journal.jsonl")
    j.append({"kind": "run"})
    assert not (tmp_path / "journal.jsonl").exists()      # main file untouched
    assert (tmp_path / "shards" / "journal-555.jsonl").exists()

def test_array_tasks_get_separate_shards(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    importlib.reload(J)
    a = J.Journal(tmp_path / "journal.jsonl").write_path
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "4")
    importlib.reload(J)
    b = J.Journal(tmp_path / "journal.jsonl").write_path
    assert a != b

def test_read_sees_main_plus_unmerged_shards(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    importlib.reload(J)
    main = J.Journal(tmp_path / "journal.jsonl")
    main.append({"kind": "old"})
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    importlib.reload(J)
    J.Journal(tmp_path / "journal.jsonl").append({"kind": "new"})
    kinds = [r["kind"] for r in J.Journal(tmp_path / "journal.jsonl").read()]
    assert "old" in kinds and "new" in kinds       # nothing invisible

def test_merge_folds_every_row_then_clears(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    importlib.reload(J)
    for jid in ("10", "11", "12"):
        monkeypatch.setenv("SLURM_JOB_ID", jid)
        importlib.reload(J)
        for i in range(5):
            J.Journal(tmp_path / "journal.jsonl").append({"kind": f"j{jid}-{i}"})
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    importlib.reload(J)
    assert J.merge_shards(tmp_path / "journal.jsonl") == 15
    rows = [json.loads(l) for l in open(tmp_path / "journal.jsonl")]
    assert len(rows) == 15                          # every row survived
    assert not list((tmp_path / "shards").glob("*.jsonl"))
    assert J.merge_shards(tmp_path / "journal.jsonl") == 0   # idempotent

def test_concurrent_writers_lose_no_rows(tmp_path, monkeypatch):
    """The failure this design exists to prevent: rows vanishing when many
    writers append at once."""
    monkeypatch.setenv("SLURM_JOB_ID", "900")
    importlib.reload(J)
    def w(i):
        # each 'job' is its own shard, so writers never share a file offset
        os.environ["SLURM_JOB_ID"] = f"900{i}"
        importlib.reload(J)
        J.Journal(tmp_path / "journal.jsonl").append({"kind": "r", "i": i})
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(w, range(24)))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    importlib.reload(J)
    assert len(J.Journal(tmp_path / "journal.jsonl").read()) == 24

def test_a_zero_second_wait_is_not_journalled_as_contention(tmp_path, monkeypatch):
    """A 'waited' row asserts the guard DELAYED work. Emitting one for a
    guard that passed instantly put 19 rows reading 'waited 0 seconds' into
    the evidence ledger, which reads as contention that never occurred."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_arm, inspect
    src = inspect.getsource(run_arm.run_curriculum)
    assert "waited >= 1.0" in src
    assert "if waited > 0:" not in src

def test_no_reader_parses_the_raw_journal_file(tmp_path):
    """Sharding the WRITES silently broke a reader that parsed the raw file:
    under a scheduler it saw an empty journal and crashed. Every reader must
    go through Journal.read(), which sees main plus unmerged shards."""
    import re
    root = Path(__file__).resolve().parents[1]
    # audit_ledger.py is the one justified exception: it reads the raw file on
    # PURPOSE, to count duplicate LINES. That is a file-level integrity check,
    # and routing it through Journal.read() -- which merges and deduplicates --
    # would hide exactly the corruption it exists to detect. Every other reader
    # wants the rows, and must see shards.
    allowed = {"journal.py", "audit_ledger.py"}
    offenders = []
    for f in list((root / "scripts").glob("*.py")) + list((root / "akademeia").glob("*.py")):
        if f.name in allowed:
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r'open\([^)]*journal\.jsonl', line):
                offenders.append(f"{f.name}:{i}")
    assert not offenders, f"readers bypassing Journal.read(): {offenders}"

def test_merging_the_same_shard_twice_adds_nothing(tmp_path, monkeypatch):
    """A shard pulled from a remote is deleted locally after merging, but the
    remote keeps its copy and the next sync downloads it again. Merging must
    therefore be idempotent -- it was not, and 141 duplicate rows went into
    the evidence ledger before that was noticed."""
    monkeypatch.setenv("SLURM_JOB_ID", "31")
    importlib.reload(J)
    for i in range(4):
        J.Journal(tmp_path / "journal.jsonl").append({"kind": "r", "i": i})
    shard = tmp_path / "shards" / "journal-31.jsonl"
    body = shard.read_text()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    importlib.reload(J)
    assert J.merge_shards(tmp_path / "journal.jsonl") == 4
    # the remote still has it: simulate the next sync re-delivering the shard
    shard.parent.mkdir(exist_ok=True)
    shard.write_text(body)
    assert J.merge_shards(tmp_path / "journal.jsonl") == 0      # nothing new
    assert len(open(tmp_path / "journal.jsonl").readlines()) == 4

def test_a_live_shard_is_not_merged(tmp_path, monkeypatch):
    """A shard whose job is still writing must be left alone, or its later
    rows arrive after it has been declared merged."""
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    importlib.reload(J)
    J.Journal(tmp_path / "journal.jsonl").append({"kind": "partial"})
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    importlib.reload(J)
    assert J.merge_shards(tmp_path / "journal.jsonl",
                          skip=("journal-77.jsonl",)) == 0
    assert (tmp_path / "shards" / "journal-77.jsonl").exists()   # still there
    # and Journal.read() can still see it, so nothing is invisible meanwhile
    assert any(r["kind"] == "partial"
               for r in J.Journal(tmp_path / "journal.jsonl").read())
