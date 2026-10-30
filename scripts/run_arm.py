"""Train+eval one curriculum arm over N seeds against a NICHE bench; emit a
gate-ready dict + journal rows. All accelerator work inside gpu_lock.
Niche score = the unseen section alone (v2 Amendment 6: a section no arm can
score on adds no discrimination, only dilution; heldrange stays reported as
an extrapolation diagnostic).

The DEVICE is journaled per run and excluded from the substrate hash; a
resume with matching hashes but a different device is journaled loudly and
re-measured, because cross-device comparisons are VOID unless the parity
gate covered that pair (PROTOCOL port gates)."""
import argparse, json, os, platform, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akademeia.trainer import train_run, tf32_enabled
from akademeia.evalrunner import eval_checkpoint
from akademeia.tokenizer import CharTokenizer
from akademeia.data.pool import load_pool
from akademeia.curriculum import seeded_curricula, validate
from akademeia.stats import tokens_to_threshold, discordant
from akademeia.ops import device_lock, wait_for_load
from akademeia.journal import Journal
from akademeia.tiers import RUNGS

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33)
# Two hosts can share a device STRING ("cpu" on one machine and on
# box B are different instruments); a copied runs/ tree must never be
# adopted across hosts (2026-08-20 amendment). Short hostname only: macOS
# flips the domain suffix with the network, and a suffix flip is not an
# instrument change.
def _instrument() -> str:
    """The instrument key used for resume identity.

    Off a scheduler this is the short hostname, unchanged: two dev boxes that
    both say "cpu" are different instruments. Under a scheduler, a homogeneous
    node class may be declared ONE instrument via AK_INSTRUMENT -- but only on
    evidence. The 2026-09-21 amendment registers that key for the a30 class
    after scripts/bitwise_check.py showed four (node, gpu) targets producing
    a BITWISE identical loss curve from the same seed. The actual node is
    still recorded on every run and arm, so provenance is never lost; only
    the resume key is widened."""
    override = os.environ.get("AK_INSTRUMENT")
    if override and os.environ.get("SLURM_JOB_ID"):
        return override
    return platform.node().split(".")[0]

HOST = _instrument()
NODE = platform.node().split(".")[0]

def precision() -> str:
    """The instrument's precision policy, recorded on every arm and stamp.
    A TF32 run and a strict-fp32 run are different instruments in exactly the
    way two hosts are: the numbers differ, so neither may adopt the other's
    cached curves."""
    return "tf32" if tf32_enabled() else "fp32"

def _stamp_matches(prior: dict, want: dict) -> bool:
    """Compare stamps, treating a MISSING precision key as fp32 so that
    caches written before the policy was made explicit still match a strict
    run -- they were measured with matmul TF32 off, which is what fp32 means
    here. A tf32 stamp can never match an absent one."""
    if prior is None:
        return False
    prior = dict(prior)
    prior.setdefault("precision", "fp32")
    return prior == want

def niche_score(ev):
    return ev["scores"]["niche_unseen"]

def load_seed_curve(path: Path, stamp: dict):
    """Stamp-gated per-seed curve cache read: the curve is adopted ONLY on an
    exact stamp match (hashes+device+host+precision+...), else None -- a cache
    keyed too loosely adopts the wrong record silently."""
    if not path.exists():
        return None
    prior = json.loads(path.read_text())
    if _stamp_matches(prior.get("stamp"), json.loads(json.dumps(stamp))):
        return [(t, ev) for t, ev in prior["curve"]]
    return None

def save_seed_curve(path: Path, stamp: dict, curve) -> None:
    path.write_text(json.dumps({"stamp": stamp, "curve": curve}, default=str))

def pooled_wins(arm, base):
    w = l = 0
    for seed_key, items in arm["final_items"].items():
        dw, dl = discordant(items, base["final_items"][seed_key])
        w += dw; l += dl
    return w, l

def assemble_arm(curves: dict, T: float, hashes: dict, device: str) -> dict:
    """curves: seed -> [(tokens, EvalResult)]. Keys str(seed) for JSON safety.
    tokens_to_T is DESCRIPTIVE (final-quality primary, carried from v2)."""
    arm = {"tokens_to_T": [], "final": [], "basics": [], "final_items": {},
           "curves": {}, "heldrange": {}, "hashes": hashes, "device": device}
    for seed, curve in sorted(curves.items()):
        n_curve = [(t, niche_score(ev)) for t, ev in curve]
        arm["tokens_to_T"].append(tokens_to_threshold(n_curve, T))
        last = curve[-1][1]
        arm["final"].append(niche_score(last))
        arm["basics"].append(last["scores"]["basics"])
        # sign test runs over exactly the items the metric is computed from
        arm["final_items"][str(seed)] = list(last["items"]["niche_unseen"])
        arm["heldrange"][str(seed)] = last["scores"]["niche_heldrange"]
        arm["curves"][str(seed)] = [
            [t, niche_score(ev), ev["scores"]["basics"],
             ev["scores"]["niche_heldrange"]] for t, ev in curve]
    return arm

def run_curriculum(cfg, curriculum, niche: str, T: float = 2.0, seeds=SEEDS,
                   tag="", stream_salt: int = 0, init_from=None,
                   phase: str = "", pool_file: str = "pool.jsonl",
                   pool_key: str = "pool_hash", device: str = "cpu",
                   assemble: bool = True) -> dict:
    """init_from: optional {seed: checkpoint_path} warm-start map (branch arms).
    pool_file/pool_key: variant pools (Study S2) -- eval stays on the frozen
    per-niche bench regardless of which pool trained the arm."""
    pool = load_pool(ROOT / "data" / pool_file)
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    manifest = json.load(open(ROOT / "freeze" / "manifest.json"))
    bench_file = ROOT / "freeze" / f"bench-{niche}.jsonl"
    hashes = {"bench": manifest[f"bench_hash_{niche}"],
              "pool": manifest[pool_key],
              "tokenizer": manifest["tokenizer_hash"], "substrate": cfg.hash()}
    assert pool.content_hash == manifest[pool_key], "pool hash drift"
    assert tok.hash() == manifest["tokenizer_hash"], "tokenizer hash drift"
    journal = Journal(ROOT / "runs" / "journal.jsonl")
    arm_path = ROOT / "runs" / f"arm-{tag or curriculum.name}.json"
    if arm_path.exists():
        # Resume: reuse only if every frozen hash matches AND the device does
        # (training is seed-deterministic per device; a device change is a
        # different instrument until the parity gate says otherwise).
        prior = json.loads(arm_path.read_text())
        if prior.get("hashes") == hashes and prior.get("device") == device \
                and prior.get("host") == HOST \
                and prior.get("precision", "fp32") == precision():
            journal.append({"kind": "resume", "tag": tag or curriculum.name,
                            "why": "arm exists with matching hashes+device+host"})
            print(f"[resume] reusing {arm_path.name} (hashes+device+host match)")
            return prior
        journal.append({"kind": "stale_arm", "tag": tag or curriculum.name,
                        "why": "hash, device, or host mismatch, re-measuring",
                        "old": {"hashes": prior.get("hashes"),
                                "device": prior.get("device"),
                                "host": prior.get("host"),
                                "precision": prior.get("precision", "fp32")},
                        "new": {"hashes": hashes, "device": device,
                                "host": HOST, "precision": precision()}})
    waited = wait_for_load()
    if waited >= 1.0:
        # only a MATERIAL wait is evidence. wait_for_load returns its own
        # elapsed time, which is a fraction of a second when the guard passes
        # immediately -- journalling that produced 19 rows reading
        # "waited 0 seconds", which reads as contention that never happened.
        journal.append({"kind": "waited", "tag": tag, "seconds": int(waited)})
    curves = {}
    with device_lock(device):
        for seed in seeds:
            # Per-seed curve cache: an external kill mid-arm must cost at
            # most ONE run, not the whole arm (SIGTERMs from the shared box
            # measured 2026-08-14). Reuse is stamp-gated like every resume.
            seed_stamp = {"hashes": hashes, "device": device, "host": HOST,
                          "precision": precision(),
                          "salt": stream_salt, "seed": seed,
                          "warm": bool((init_from or {}).get(seed))}
            curve_path = ROOT / "runs" / f"curve-{tag or curriculum.name}-s{seed}.json"
            cached = load_seed_curve(curve_path, seed_stamp)
            if cached is not None:
                curves[seed] = cached
                journal.append({"kind": "resume_seed", "tag": tag,
                                "seed": seed})
                continue
            t0 = time.monotonic()
            out = ROOT / "runs" / f"{tag or curriculum.name}-s{seed}"
            res = train_run(cfg, curriculum, pool, tok, seed=seed, out_dir=out,
                            stream_salt=stream_salt,
                            init_from=(init_from or {}).get(seed),
                            device=device)
            curve = []
            for tokens, ck in res.checkpoints:
                ev = eval_checkpoint(cfg, ck, bench_file,
                                     ROOT / "freeze" / "tokenizer.json",
                                     device=device)
                curve.append((tokens, ev))
            curves[seed] = curve
            save_seed_curve(curve_path, seed_stamp, curve)
            journal.append({"kind": "run", "phase": phase, "niche": niche,
                            "curriculum": curriculum.name, "seed": seed,
                            "tag": tag, "salt": stream_salt, "device": device,
                            "warm_start": bool((init_from or {}).get(seed)),
                            "precision": precision(), "node": NODE,
                            "wall_s": round(time.monotonic() - t0, 1),
                            "final": niche_score(curve[-1][1]),
                            "basics": curve[-1][1]["scores"]["basics"],
                            "losses_first_last": [res.losses[0], res.losses[-1]],
                            "hashes": hashes})
    if not assemble:
        # fan-out task: the seed's curve is cached and that is the whole job.
        # Assembly belongs to the call that holds every seed, so a partial
        # arm file can never be written and later mistaken for a complete one.
        journal.append({"kind": "partial", "tag": tag, "seeds": list(seeds),
                        "why": "fan-out task; curve cached, arm not assembled"})
        return {}
    arm = assemble_arm(curves, T, hashes, device)
    arm["host"] = HOST
    arm["node"] = NODE          # provenance kept even when the key is a class
    arm["precision"] = precision()
    arm_path.write_text(json.dumps(arm, default=str, indent=1))
    return arm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", required=True)
    ap.add_argument("--niche", default="physics")
    ap.add_argument("--rung", type=int, default=1, choices=sorted(RUNGS))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = RUNGS[args.rung]
    c = seeded_curricula(cfg.total_tokens)[args.curriculum]
    validate(c, cfg.total_tokens)
    arm = run_curriculum(cfg, c, args.niche, seeds=tuple(args.seeds),
                         tag=f"adhoc-r{args.rung}-{args.curriculum}-{args.niche}",
                         device=args.device)
    print(json.dumps({k: arm[k] for k in ("final", "basics")}, default=str))

if __name__ == "__main__":
    main()
