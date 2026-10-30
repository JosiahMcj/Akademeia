import pytest
import akademeia.ops as ops
from akademeia.ops import (gpu_lock, parse_meminfo, parse_memory_pressure,
                           parse_nvidia_smi, box_ok, wait_for_load)


MEMINFO = """MemTotal:       131072000 kB
MemFree:         9830400 kB
MemAvailable:   65536000 kB
Buffers:          204800 kB
"""

def test_parse_meminfo_absolute_and_pct():
    got = parse_meminfo(MEMINFO)
    assert got["available_pct"] == pytest.approx(50.0)
    assert got["available_gb"] == pytest.approx(62.5)

def test_parse_meminfo_unreadable_is_permissive():
    assert parse_meminfo("")["available_gb"] > 1e6
    assert parse_meminfo("MemTotal: 0 kB\nMemAvailable: 0 kB")["available_pct"] == 100.0

def test_parse_memory_pressure():
    assert parse_memory_pressure(
        "The system has 1 pages...\nSystem-wide memory free percentage: 37%\n"
    ) == 37.0
    assert parse_memory_pressure("garbage") == 100.0

def test_parse_nvidia_smi_row():
    got = parse_nvidia_smi("102400, 128000, 87")
    assert got["mem_free_pct"] == pytest.approx(20.0)
    assert got["util_pct"] == 87.0

def test_parse_nvidia_smi_garbage_is_permissive():
    assert parse_nvidia_smi("")["mem_free_pct"] == 100.0
    assert parse_nvidia_smi("N/A, N/A, N/A")["util_pct"] == 0.0

def test_box_ok_blocks_on_low_gpu_memory(monkeypatch):
    monkeypatch.setattr(ops, "mem_available_gb", lambda: 100.0)
    monkeypatch.setattr(ops, "load_ok", lambda *a, **k: True)
    monkeypatch.setattr(ops, "gpu_state",
                        lambda: {"mem_free_pct": 4.0, "util_pct": 99.0})
    ok, why = box_ok()
    assert not ok and "GPU" in why

def test_box_ok_no_gpu_guard_when_no_gpu(monkeypatch):
    """The dev box has no nvidia-smi; the GPU guard must simply not apply."""
    monkeypatch.setattr(ops, "mem_available_gb", lambda: 100.0)
    monkeypatch.setattr(ops, "load_ok", lambda *a, **k: True)
    monkeypatch.setattr(ops, "gpu_state", lambda: None)
    ok, why = box_ok()
    assert ok

def test_box_ok_blocks_on_low_memory(monkeypatch):
    """ABSOLUTE headroom, not a percentage: a box's resident model
    server keeps ~7GB of 121GB free (6%) and must still pass; 2GB must not."""
    monkeypatch.setattr(ops, "mem_available_gb", lambda: 2.0)
    ok, why = box_ok()
    assert not ok and "memory available" in why

def test_box_ok_passes_at_accelerator_box_headroom(monkeypatch):
    monkeypatch.setattr(ops, "mem_available_gb", lambda: 7.0)
    monkeypatch.setattr(ops, "load_ok", lambda *a, **k: True)
    monkeypatch.setattr(ops, "gpu_state", lambda: None)
    ok, _ = box_ok()
    assert ok

def test_wait_for_load_returns_when_quiet(monkeypatch):
    monkeypatch.setattr(ops, "box_ok", lambda *a, **k: (True, "ok"))
    assert wait_for_load(poll=0.01, max_wait=1.0) < 1.0

def test_wait_for_load_delays_then_times_out_never_exits(monkeypatch):
    """The guard DELAYS and eventually raises; it must never sys.exit (the
    46-hour dead-chain lesson)."""
    monkeypatch.setattr(ops, "box_ok", lambda *a, **k: (False, "busy"))
    with pytest.raises(TimeoutError):
        wait_for_load(poll=0.01, max_wait=0.05, log=lambda *a, **k: None)

def test_gpu_lock_is_exclusive(tmp_path):
    p = tmp_path / "lock"
    with gpu_lock(p):
        with pytest.raises(TimeoutError):
            with gpu_lock(p, timeout=0):
                pass

def test_cuda_box_raises_the_memory_floor_to_the_context_floor(monkeypatch):
    """Measured on the accelerator: context creation succeeds ~8.2GB, fails ~6.9GB.
    On a CUDA box the guard must hold BELOW the floor instead of letting the
    run crash into it; a no-GPU box keeps the lower floor."""
    monkeypatch.setattr(ops, "load_ok", lambda *a, **k: True)
    monkeypatch.setattr(ops, "gpu_state",
                        lambda: {"mem_free_pct": 100.0, "util_pct": 0.0})
    monkeypatch.setattr(ops, "mem_available_gb", lambda: 6.2)
    ok, why = box_ok()
    assert not ok and "OOM risk" in why
    monkeypatch.setattr(ops, "mem_available_gb", lambda: 6.6)
    ok, _ = box_ok()
    assert ok
    monkeypatch.setattr(ops, "gpu_state", lambda: None)
    monkeypatch.setattr(ops, "mem_available_gb", lambda: 6.0)
    ok, _ = box_ok()
    assert ok

def test_hidden_cuda_disables_the_context_floor(monkeypatch):
    """The CPU lane hides CUDA entirely; a run that cannot create a context
    needs no CUDA headroom, so the floor must not apply."""
    import shutil as _sh
    monkeypatch.setattr(_sh, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert ops.gpu_state() is None
    monkeypatch.setattr(ops, "load_ok", lambda *a, **k: True)
    monkeypatch.setattr(ops, "mem_available_gb", lambda: 5.0)
    ok, _ = box_ok()
    assert ok

def test_device_lock_separates_the_lanes(tmp_path, monkeypatch):
    """cpu and cuda chains must not queue behind each other; same-class
    chains still serialize."""
    monkeypatch.setattr(ops, "DEFAULT_LOCK", str(tmp_path / "gpu"))
    monkeypatch.setattr(ops, "CPU_LOCK", str(tmp_path / "cpu"))
    with ops.device_lock("cuda"):
        with ops.device_lock("cpu"):        # different silicon: no contention
            pass
        with pytest.raises(TimeoutError):   # same class: still exclusive
            with ops.device_lock("cuda:0", timeout=0):
                pass
