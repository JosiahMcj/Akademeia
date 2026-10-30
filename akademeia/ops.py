"""Box discipline for both execution homes: a shared development machine (Darwin)
and a single-tenant accelerator box (Linux/CUDA).

Laws carried from v1/v2, all measured the hard way:
  - Guards DELAY work, never kill it (a mid-chain sys.exit hid a dead chain
    for 46 hours). wait_for_load is the only sanctioned way to wait.
  - A guard that cannot read its signal must not block on it (return the
    permissive value, journal nothing false).
  - The gpulock stays even on a single-tenant box: it is the same-box
    serialization primitive for concurrent chains, not just Metal contention.
"""
import contextlib, fcntl, os, platform, re, shutil, subprocess, time

def under_scheduler() -> bool:
    """True inside a SLURM allocation. The scheduler owns arbitration there:
    it decided which GPUs and cores are mine, so this module's same-box
    guards must step aside rather than second-guess it."""
    return bool(os.environ.get("SLURM_JOB_ID"))

def _lock_dir() -> str:
    """Lock files must NEVER live on a shared filesystem. On a cluster, home
    is NFS-mounted on every node, so a lock under ~ is either a cluster-wide
    global lock (silently collapsing a parallel fan-out into a queue that
    merely looks like a slow cluster) or silently ineffective, depending on
    whether the mount honours flock. Under a scheduler the lock is therefore
    node-local and scoped to the allocation; elsewhere it stays in ~ , which
    is local disk on the dev boxes and is the behaviour every existing run
    was measured under."""
    if under_scheduler():
        base = os.environ.get("TMPDIR", "/tmp")
        return os.path.join(base, f"akademeia-{os.environ['SLURM_JOB_ID']}")
    return os.path.expanduser("~")

DEFAULT_LOCK = os.path.join(_lock_dir(), ".gpulock")
CUDA_CONTEXT_FLOOR_GB = 6.5     # with CUDA_MODULE_LOADING=LAZY. Measured: lazy
                                # context+matmul OK at 7.4GB (eager fails there);
                                # a full run's torch footprint is ~1.5GB, so a
                                # 6.5 release still leaves ~5GB -- far above the
                                # zone where the kernel OOM-killer would eye the
                                # resident server. Crash-retry absorbs marginals.
                                # FLOOR STOPS HERE: below this, coexistence risk
                                # shifts to the resident server and the decision
                                # belongs to the operator, not this constant.

CPU_LOCK = os.path.join(_lock_dir(), ".cpulock")

def device_lock(device: str = "cuda", timeout: float = 6 * 3600):
    """Same-box serialization PER DEVICE CLASS: the cpu and cuda lanes use
    different silicon and must not queue behind each other -- memory safety
    is the guard's job, not the lock's. Chains on the SAME device class
    still serialize (the original gpulock law)."""
    path = DEFAULT_LOCK if str(device).startswith("cuda") else CPU_LOCK
    return gpu_lock(path, timeout)

@contextlib.contextmanager
def gpu_lock(path=DEFAULT_LOCK, timeout: float = 6 * 3600):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "w")
    deadline = time.time() + timeout
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() > deadline:
                f.close()
                raise TimeoutError(f"gpulock busy > {timeout}s")
            time.sleep(5)
    try:
        yield f
    finally:
        fcntl.flock(f, fcntl.LOCK_UN); f.close()

# ---- pure parsers (testable without the hardware) ----

def parse_meminfo(text: str) -> dict:
    """/proc/meminfo -> {"available_gb", "available_pct"}. Permissive when
    unreadable (do not block on a guard we cannot read)."""
    m_avail = re.search(r"MemAvailable:\s*(\d+)\s*kB", text)
    m_total = re.search(r"MemTotal:\s*(\d+)\s*kB", text)
    if not (m_avail and m_total) or int(m_total.group(1)) == 0:
        return {"available_gb": 1e9, "available_pct": 100.0}
    avail, total = int(m_avail.group(1)), int(m_total.group(1))
    return {"available_gb": avail / 1024 / 1024,
            "available_pct": 100.0 * avail / total}

def parse_memory_pressure(text: str) -> float:
    """macOS `memory_pressure -Q` -> system-wide free percentage."""
    m = re.search(r"free percentage:\s*(\d+)%", text)
    return float(m.group(1)) if m else 100.0

def parse_nvidia_smi(text: str) -> dict:
    """One CSV row of `nvidia-smi --query-gpu=memory.used,memory.total,
    utilization.gpu --format=csv,noheader,nounits` -> dict. Unreadable rows
    return permissive values (mem_free_pct=100, util=0)."""
    parts = [p.strip() for p in text.strip().split(",")]
    try:
        used, total, util = float(parts[0]), float(parts[1]), float(parts[2])
        free = 100.0 * (1.0 - used / total) if total > 0 else 100.0
        return {"mem_free_pct": free, "util_pct": util}
    except (IndexError, ValueError):
        return {"mem_free_pct": 100.0, "util_pct": 0.0}

# ---- live signals ----

def mem_available_gb() -> float:
    """ABSOLUTE available memory, per platform. This -- not a percentage --
    is the signal that precedes an OOM/jetsam kill of a lab run: a box whose
    resident model server deliberately holds all but a few GB would fail a
    percentage guard forever while having room to spare for a tiny run
    (measured on the accelerator box: a resident model server keeps ~7GB
    of 121GB free)."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["memory_pressure", "-Q"], capture_output=True,
                                 text=True, timeout=20).stdout
            pct = parse_memory_pressure(out)
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                       capture_output=True, text=True,
                                       timeout=10).stdout.strip())
            return total * pct / 100.0 / 1024 ** 3
        with open("/proc/meminfo") as f:
            return parse_meminfo(f.read())["available_gb"]
    except Exception:
        return 1e9          # unknown -> do not block on a guard we cannot read

def gpu_state() -> dict | None:
    """nvidia-smi memory/utilization, or None when there is no NVIDIA GPU
    (the dev box) OR CUDA is hidden from this process
    (CUDA_VISIBLE_DEVICES="", the CPU lane) -- a run that cannot create a
    CUDA context needs no CUDA headroom, so the context floor must not
    apply to it. None means the GPU guard simply does not apply."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return None
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        return parse_nvidia_smi(out.splitlines()[0]) if out.strip() else None
    except Exception:
        return None

def load_ok(threshold: float = 16.0, probes: int = 2, gap: float = 10.0) -> bool:
    """Double-probe load guard (both probes must pass). 16 is the measured
    floor-plus-margin on the shared 10-core dev box; single-tenant chains may
    pass a tighter value.

    Under a scheduler this guard does not apply: load average reflects the
    WHOLE node, including other users' jobs that cannot touch my cgroup's
    cores or GPU, so a fixed threshold on a 64-core node defers forever on a
    machine that is doing exactly what it should. The guards-go-blind lesson
    in reverse -- a guard measuring something that is no longer the risk."""
    if under_scheduler():
        return True
    for i in range(probes):
        if os.getloadavg()[0] > threshold:
            return False
        if i + 1 < probes:
            time.sleep(gap)
    return True

def box_ok(load_threshold: float = 16.0, min_mem_free_gb: float = 4.0,
           min_gpu_mem_free_pct: float = 10.0, probes: int = 2,
           gap: float = 10.0):
    """(ok, reason). Memory headroom (absolute GB), GPU memory (when an
    NVIDIA GPU exists AND reports totals -- unified-memory boxes report N/A
    and the guard defers to the system one), and load must all pass.

    On a CUDA box the floor rises to CUDA_CONTEXT_FLOOR_GB: measured on the
    accelerator beside the resident model server, torch CUDA context creation
    succeeds at ~8.2GB available and fails outright at ~6.9GB -- waiting
    below the floor beats crashing into it."""
    gpu = gpu_state()
    if under_scheduler():
        # the scheduler already reserved this memory and these GPUs for me
        return True, "ok (scheduler-managed allocation)"
    floor = max(min_mem_free_gb, CUDA_CONTEXT_FLOOR_GB) if gpu is not None \
        else min_mem_free_gb
    free = mem_available_gb()
    if free < floor:
        return False, (f"memory available {free:.1f}GB < "
                       f"{floor:.1f}GB (OOM risk)")
    if gpu is not None and gpu["mem_free_pct"] < min_gpu_mem_free_pct:
        return False, (f"GPU memory free {gpu['mem_free_pct']:.0f}% < "
                       f"{min_gpu_mem_free_pct:.0f}%")
    if not load_ok(load_threshold, probes, gap):
        return False, f"load 1m={os.getloadavg()[0]:.1f} > {load_threshold}"
    return True, "ok"

def wait_for_load(threshold: float = 16.0, probes: int = 2, gap: float = 10.0,
                  poll: float = 120.0, max_wait: float = 12 * 3600,
                  log=print) -> float:
    """Block until the box is quiet, then return seconds waited.

    A busy box must DELAY work, never kill it. Never call while holding the
    gpulock."""
    t0 = time.monotonic()
    deadline = t0 + max_wait
    while True:
        ok, why = box_ok(threshold, probes=probes, gap=gap)
        if ok:
            return time.monotonic() - t0
        if time.monotonic() > deadline:
            raise TimeoutError(f"box busy > {max_wait}s ({why})")
        log(f"[box-guard] waiting: {why}; waited {int(time.monotonic() - t0)}s, "
            f"retry in {int(poll)}s")
        time.sleep(poll)
