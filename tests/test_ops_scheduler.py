"""Cluster-safety laws for the box guards (2026-09-21 amendment).

Every rule here exists because a shared-filesystem lock or a whole-node load
guard would misbehave under a scheduler in a way that LOOKS like slow
hardware rather than like a bug.
"""
import importlib, os
import pytest

def _reload(monkeypatch, slurm=None, tmpdir=None):
    for k, v in (("SLURM_JOB_ID", slurm), ("TMPDIR", tmpdir)):
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import akademeia.ops as ops
    return importlib.reload(ops)

def test_lock_is_never_on_a_shared_home_under_a_scheduler(monkeypatch):
    """The landmine: ~ is NFS on every compute node, so a lock there is a
    cluster-wide global lock or a no-op. Neither is same-box serialization."""
    ops = _reload(monkeypatch, slurm="12345", tmpdir="/scratch/j12345")
    assert "/scratch/j12345" in ops.DEFAULT_LOCK
    assert os.path.expanduser("~") not in ops.DEFAULT_LOCK
    assert "12345" in ops.DEFAULT_LOCK          # scoped to THIS allocation
    assert ops.DEFAULT_LOCK != ops.CPU_LOCK     # still per device class

def test_two_allocations_do_not_share_a_lock(monkeypatch):
    """Independent jobs must not serialize against each other: that is what
    turns a 28-way fan-out into a 1-way queue."""
    a = _reload(monkeypatch, slurm="1", tmpdir="/scratch").DEFAULT_LOCK
    b = _reload(monkeypatch, slurm="2", tmpdir="/scratch").DEFAULT_LOCK
    assert a != b

def test_lock_stays_in_home_off_a_scheduler(monkeypatch):
    """Unchanged where every existing run was measured: the dev boxes."""
    ops = _reload(monkeypatch, slurm=None)
    assert ops.DEFAULT_LOCK == os.path.join(os.path.expanduser("~"), ".gpulock")

def test_load_guard_stands_down_under_a_scheduler(monkeypatch):
    """Load average counts other users' jobs that cannot touch my cgroup."""
    ops = _reload(monkeypatch, slurm="777", tmpdir="/tmp")
    monkeypatch.setattr(ops.os, "getloadavg", lambda: (999.0, 999.0, 999.0))
    assert ops.load_ok(threshold=16.0) is True
    ok, why = ops.box_ok()
    assert ok and "scheduler" in why

def test_load_guard_still_bites_off_a_scheduler(monkeypatch):
    """...and the guard is NOT weakened anywhere it still applies."""
    ops = _reload(monkeypatch, slurm=None)
    monkeypatch.setattr(ops.os, "getloadavg", lambda: (999.0, 999.0, 999.0))
    assert ops.load_ok(threshold=16.0, probes=1) is False
    ok, why = ops.box_ok(probes=1)
    assert not ok

def test_gpu_lock_creates_its_directory(monkeypatch, tmp_path):
    """A node-local lock dir will not exist on a fresh node."""
    ops = _reload(monkeypatch, slurm="42", tmpdir=str(tmp_path))
    with ops.gpu_lock(ops.DEFAULT_LOCK, timeout=5):
        assert os.path.exists(ops.DEFAULT_LOCK)
