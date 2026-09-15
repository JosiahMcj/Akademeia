"""Q1 ladder driver -- rank stability across rungs 1-4 (PROTOCOL Q1).

Per rung, in order:
  probe      1/3-budget TIMING run (timing only; 1/3-budget SCORES are known
             uninformative from v2). Extrapolated full wall over the rung's
             limit -> exit 8: the budget decision returns to the operator,
             never auto-adjusts.
  baseline   direct50 A (salt 0) + A' (salt 1), 3 seeds
  calibrate  fresh rung thresholds from the A/A' pair (never reused across
             rungs; scripts/calibrate.py freezes them one way)
  arms       branch (core B/2 + warm-started niche branch B/2) and adaptive2
             (niche pinned 0.5, basics steered), 3 seeds each
  verdict    per-arm gates vs the rung's direct50 + rank_agreement vs the
             previous rung

Rungs 1-2 re-measure the v2 arms ON THIS SUBSTRATE so every cross-rung
comparison is same-instrument; the MLX reference is only ever touched by
scripts/parity_check.py."""
import argparse, dataclasses, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from akademeia.curriculum import Segment, Curriculum, validate
from akademeia.data.families import BASIC_FAMILIES
from akademeia.data.pool import load_pool
from akademeia.tokenizer import CharTokenizer
from akademeia.trainer import train_run
from akademeia.evalrunner import eval_checkpoint
from akademeia.gates import gate_v2, ThresholdsV2
from akademeia.journal import Journal
from akademeia.ranks import rank_agreement
from akademeia.scheduler import LearningProgressScheduler
from akademeia.adaptive import train_adaptive, probe_informativeness
from akademeia.probes import load_probes
from akademeia.ops import device_lock, wait_for_load
from akademeia.tiers import RUNGS
from run_arm import (run_curriculum, assemble_arm, pooled_wins, SEEDS, HOST,
                     load_seed_curve, save_seed_curve)
from calibrate import calibrate_rung

ROOT = Path(__file__).resolve().parents[1]
NICHE = "physics"
ARMS = ("direct50", "branch", "adaptive2")     # the registered Q1 arm set

# est. full-run wall above which the rung aborts to the operator (exit 8).
# Rung-1 limit raised 3600 -> 6000 by the 2026-08-14 amendment: the original
# was calibrated for the accelerator (~750s/run); the gated CPU instrument
# measured est 5079s/run, and the amendment accepts the wall-clock trade
# while the GPU waits for memory windows.
# Rung-3 limit 5400 -> 9000 by the dated 2026-08-26 amendment
# (wall-clock budget, not a measurement gate): mps measured 8307s/run and
# the wall was raised for it by dated amendment; the cpu path (16553s)
# stays correctly over this limit.
PROBE_LIMIT_FULL_S = {1: 6000, 2: 5400, 3: 9000, 4: 8 * 3600}

def direct50(niche: str, budget: int) -> Curriculum:
    """Single segment, basics:niche = 50:50 by weight mass -- the v2 recipe."""
    w = {**{f: 1.0 for f in BASIC_FAMILIES}, niche: 10.0}
    d = {**{f: (1, 3) for f in BASIC_FAMILIES}, niche: (1, 3)}
    return Curriculum(f"direct50-{niche}", (Segment(budget, w, d),))

def core_curr(budget: int) -> Curriculum:
    w = {f: 1.0 for f in BASIC_FAMILIES}
    return Curriculum("core", (Segment(budget, w, {f: (1, 3) for f in w}),))

def branch_curr(niche: str, budget: int) -> Curriculum:
    return Curriculum(f"branch-{niche}",
                      (Segment(budget, {niche: 1.0}, {niche: (1, 3)}),))

def families_for(niche: str) -> list:
    return sorted(tuple(BASIC_FAMILIES) + (niche,))

def probes_for(niche: str, per_family: int):
    """Frozen probes filtered to this run's family set; count law re-asserted."""
    allp = load_probes(ROOT / "freeze" / "probes.jsonl", per_family)
    fams = set(families_for(niche))
    sub = [p for p in allp if p.family in fams]
    expected = per_family * len(fams)
    if len(sub) != expected:
        print(f"FATAL: filtered probe count {len(sub)} != {expected}")
        sys.exit(9)
    return sub

def cmd_probe(rung, journal, device):
    cfg = RUNGS[rung]
    probe_cfg = dataclasses.replace(cfg, total_tokens=cfg.total_tokens // 3,
                                    eval_every_tokens=cfg.total_tokens // 3)
    c = direct50(NICHE, probe_cfg.total_tokens)
    validate(c, probe_cfg.total_tokens)
    run_curriculum(probe_cfg, c, NICHE, seeds=(11,),
                   tag=f"ladder-r{rung}-timing", phase=f"L{rung}",
                   device=device)
    rows = [json.loads(l) for l in open(ROOT / "runs" / "journal.jsonl")]
    wall = [r for r in rows if r.get("kind") == "run"
            and r.get("tag") == f"ladder-r{rung}-timing"][-1]["wall_s"]
    est, limit = wall * 3, PROBE_LIMIT_FULL_S[rung]
    journal.append({"kind": "probe", "rung": rung, "wall_s": wall,
                    "est_full_s": est, "limit_s": limit, "device": device})
    print(f"[ladder r{rung}] probe wall={int(wall)}s -> est full {int(est)}s "
          f"(limit {limit}s)")
    if est > limit:
        print(f"ABORT: estimated {int(est)}s/run > {limit}s; the budget "
              f"decision returns to the operator (no auto-adjust)")
        sys.exit(8)

def cmd_baseline(rung, device):
    cfg = RUNGS[rung]
    B = cfg.total_tokens
    c = direct50(NICHE, B); validate(c, B)
    run_curriculum(cfg, c, NICHE, tag=f"ladder-r{rung}-direct50-A",
                   stream_salt=0, phase=f"L{rung}", device=device)
    run_curriculum(cfg, c, NICHE, tag=f"ladder-r{rung}-direct50-Aprime",
                   stream_salt=1, phase=f"L{rung}", device=device)

EXT_SEEDS = (44, 55, 66)      # the 2026-09-14 rank seed-extension amendment

def cmd_extend(rung, journal, device):
    """Seed extension for the two arms whose ORDER flipped at rung 3
    (2026-09-14 amendment). Runs under their own -ext tags: no frozen
    threshold is re-derived, no verdict is recomputed, and the report this
    feeds is descriptive. branch is excluded deliberately -- its position is
    separated with power at every rung, and its warm-start cores no longer
    exist to reproduce."""
    cfg = RUNGS[rung]
    c = direct50(NICHE, cfg.total_tokens); validate(c, cfg.total_tokens)
    run_curriculum(cfg, c, NICHE, tag=f"ladder-r{rung}-direct50-A-ext",
                   seeds=EXT_SEEDS, stream_salt=0, phase=f"L{rung}-ext",
                   device=device)
    _adaptive2_arm(rung, journal, device, seeds=EXT_SEEDS,
                   tag=f"ladder-r{rung}-adaptive2-ext")

def cmd_rankpooled(rung, journal, device):
    """Descriptive pooled comparison of the two arms at this rung."""
    import statistics as st
    out = {}
    for name in ("direct50-A", "adaptive2"):
        vals = []
        for suffix in ("", "-ext"):
            f = ROOT / "runs" / f"arm-ladder-r{rung}-{name}{suffix}.json"
            if not f.exists():
                print(f"missing {f.name}; run the extend stage first"); sys.exit(2)
            vals += json.loads(f.read_text())["final"]
        out[name] = vals
    a, b = out["direct50-A"], out["adaptive2"]
    sem = lambda x: st.stdev(x) / len(x) ** 0.5
    d = st.mean(b) - st.mean(a)
    se = (sem(a) ** 2 + sem(b) ** 2) ** 0.5
    row = {"kind": "study", "study": "Q1-rank-pooled", "rung": rung,
           "device": device, "n_per_arm": len(a),
           "direct50_runs": [round(x, 4) for x in a],
           "adaptive2_runs": [round(x, 4) for x in b],
           "direct50_mean": round(st.mean(a), 4), "direct50_sem": round(sem(a), 4),
           "adaptive2_mean": round(st.mean(b), 4), "adaptive2_sem": round(sem(b), 4),
           "difference": round(d, 4), "difference_uncertainty": round(se, 4),
           "ratio": round(abs(d) / se, 2) if se else None,
           "nominal_order": ("adaptive2 > direct50" if d > 0
                             else "direct50 > adaptive2"),
           "note": "descriptive pooled estimate; the registered rung verdicts "
                   "and frozen thresholds are unchanged and not recomputed"}
    journal.append(row)
    print(json.dumps(row, indent=1, default=str))

def cmd_arms(rung, journal, device):
    cfg = RUNGS[rung]
    half = cfg.total_tokens // 2
    pool = load_pool(ROOT / "data" / "pool.jsonl")
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    manifest = json.load(open(ROOT / "freeze" / "manifest.json"))
    core_stamp = {"pool": manifest["pool_hash"],
                  "tokenizer": manifest["tokenizer_hash"],
                  "substrate": cfg.hash(), "tokens": half, "device": device}
    core = core_curr(half); validate(core, half)
    core_ckpts, to_train = {}, []
    for seed in SEEDS:
        d = ROOT / "runs" / f"ladder-r{rung}-core-s{seed}"
        stamp_f = d / "core_stamp.json"
        done = sorted(d.glob("ckpt_*.pt"),
                      key=lambda p: int(p.stem.split("_")[1]))
        # resume ONLY on a stamp match: a core trained against a different
        # pool/substrate/device is a different core
        if (done and stamp_f.exists()
                and json.loads(stamp_f.read_text()) == core_stamp
                and int(done[-1].stem.split("_")[1]) >= half):
            core_ckpts[seed] = done[-1]
        else:
            if done:
                journal.append({"kind": "stale_core", "rung": rung, "seed": seed,
                                "why": "missing or mismatched core stamp"})
            to_train.append(seed)
    if to_train:
        waited = wait_for_load()
        if waited > 0:
            journal.append({"kind": "waited", "tag": f"ladder-r{rung}-core",
                            "seconds": int(waited)})
        with device_lock(device):
            for seed in to_train:
                out = ROOT / "runs" / f"ladder-r{rung}-core-s{seed}"
                res = train_run(cfg, core, pool, tok, seed=seed, out_dir=out,
                                device=device)
                core_ckpts[seed] = res.checkpoints[-1][1]
                (out / "core_stamp.json").write_text(json.dumps(core_stamp))
                journal.append({"kind": "run", "phase": f"L{rung}",
                                "curriculum": "core", "seed": seed,
                                "tag": f"ladder-r{rung}-core", "device": device,
                                "losses_first_last": [res.losses[0],
                                                      res.losses[-1]]})
    c = branch_curr(NICHE, half); validate(c, half)
    run_curriculum(cfg, c, NICHE, tag=f"ladder-r{rung}-branch",
                   init_from=core_ckpts, phase=f"L{rung}", device=device)
    _adaptive2_arm(rung, journal, device)

def _adaptive2_arm(rung, journal, device, seeds=SEEDS, tag=None):
    """adaptive2: niche pinned 0.5, basics steered (v2 Amendment 8 design).
    seeds/tag are parameters only so the registered seed extension can reuse
    this exact code path rather than a second copy of it."""
    cfg = RUNGS[rung]
    pool = load_pool(ROOT / "data" / "pool.jsonl")
    tok = CharTokenizer.load(ROOT / "freeze" / "tokenizer.json")
    manifest = json.load(open(ROOT / "freeze" / "manifest.json"))
    hashes = {"bench": manifest[f"bench_hash_{NICHE}"],
              "pool": manifest["pool_hash"],
              "tokenizer": manifest["tokenizer_hash"], "substrate": cfg.hash()}
    tag = tag or f"ladder-r{rung}-adaptive2"
    arm_path = ROOT / "runs" / f"arm-{tag}.json"
    if arm_path.exists():
        # Same resume law as run_curriculum: hashes AND device AND host --
        # two boxes both saying "cpu" are different instruments (this block
        # originally checked only hashes+device; closed 2026-08-26).
        prior = json.loads(arm_path.read_text())
        if prior.get("hashes") == hashes and prior.get("device") == device \
                and prior.get("host") == HOST:
            journal.append({"kind": "resume", "tag": tag})
            return
    probes = probes_for(NICHE, manifest["probes_per_family"])
    bench_file = ROOT / "freeze" / f"bench-{NICHE}.jsonl"
    curves, info = {}, {}
    wait_for_load()
    with device_lock(device):
        for seed in seeds:
            t0 = time.monotonic()
            # Per-seed curve cache, same stamp law as run_curriculum: a
            # mid-arm kill must cost at most ONE run. "informativeness" is
            # derived from probe_history, so the cache carries it alongside.
            seed_stamp = {"hashes": hashes, "device": device, "host": HOST,
                          "seed": seed, "kind": "adaptive2"}
            curve_path = ROOT / "runs" / f"curve-{tag}-s{seed}.json"
            cached = load_seed_curve(curve_path, seed_stamp)
            if cached is not None:
                prior = json.loads(curve_path.read_text())
                curves[seed] = cached
                info[seed] = prior["informativeness"]
                journal.append({"kind": "resume_seed", "tag": tag,
                                "seed": seed})
                continue
            sched = LearningProgressScheduler(families_for(NICHE),
                                              fixed={NICHE: 0.5})
            res = train_adaptive(cfg, pool, tok, seed=seed,
                                 out_dir=ROOT / "runs" / f"{tag}-s{seed}",
                                 probes=probes, scheduler=sched, device=device)
            curve = []
            for tokens, ck in res.checkpoints:
                ev = eval_checkpoint(cfg, ck, bench_file,
                                     ROOT / "freeze" / "tokenizer.json",
                                     device=device)
                curve.append((tokens, ev))
            curves[seed] = curve
            info[seed] = probe_informativeness(res.probe_history)
            save_seed_curve(curve_path, seed_stamp, curve)
            prior = json.loads(curve_path.read_text())
            prior["informativeness"] = info[seed]
            curve_path.write_text(json.dumps(prior, default=str))
            journal.append({"kind": "run", "phase": f"L{rung}", "niche": NICHE,
                            "curriculum": "adaptive2", "seed": seed, "tag": tag,
                            "device": device,
                            "wall_s": round(time.monotonic() - t0, 1),
                            "informativeness": info[seed], "hashes": hashes})
    arm = assemble_arm(curves, T=2.0, hashes=hashes, device=device)
    arm["host"] = HOST
    arm["informativeness"] = {str(k): v for k, v in info.items()}
    arm_path.write_text(json.dumps(arm, default=str, indent=1))

def rung_finals(rung) -> dict | None:
    """Mean finals per arm from the STORED arm files (never retyped by hand)."""
    out = {}
    for name in ARMS:
        tag = (f"ladder-r{rung}-direct50-A" if name == "direct50"
               else f"ladder-r{rung}-{name}")
        p = ROOT / "runs" / f"arm-{tag}.json"
        if not p.exists():
            return None
        arm = json.loads(p.read_text())
        out[name] = sum(arm["final"]) / len(arm["final"])
    return out

def cmd_verdict(rung, journal):
    th = ThresholdsV2(**json.load(
        open(ROOT / "freeze" / f"thresholds_v3_r{rung}_{NICHE}.json")))
    base = json.loads(
        (ROOT / "runs" / f"arm-ladder-r{rung}-direct50-A.json").read_text())
    for name in ("branch", "adaptive2"):
        arm = json.loads(
            (ROOT / "runs" / f"arm-ladder-r{rung}-{name}.json").read_text())
        arm["wins"], arm["losses"] = pooled_wins(arm, base)
        v_sup = gate_v2(arm, base, th, mode="superior")
        v_non = gate_v2(arm, base, th, mode="noninferior")
        journal.append({"kind": "verdict", "rung": rung, "arm": name,
                        "baseline": f"ladder-r{rung}-direct50-A",
                        "superior": v_sup, "noninferior": v_non,
                        "wins": arm["wins"], "losses": arm["losses"]})
        print(name, json.dumps({"superior": v_sup["promote"],
                                "noninferior": v_non["promote"],
                                "wins": arm["wins"], "losses": arm["losses"]}))
    cur = rung_finals(rung)
    row = {"kind": "rung", "rung": rung, "finals": cur}
    prev = rung_finals(rung - 1) if rung > 1 else None
    if prev is not None:
        row["prev_finals"] = prev
        row["rank_agreement_vs_prev"] = rank_agreement(prev, cur)
    ladder = {r: rung_finals(r) for r in sorted(RUNGS) if rung_finals(r)}
    row["ladder_so_far"] = ladder
    journal.append(row)
    print(json.dumps(row, indent=1, default=str))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["probe", "baseline", "calibrate",
                                      "arms", "verdict", "extend",
                                      "rankpooled"])
    ap.add_argument("--rung", type=int, required=True, choices=sorted(RUNGS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--floor-advisory", action="store_true")
    args = ap.parse_args()
    journal = Journal(ROOT / "runs" / "journal.jsonl")
    if args.stage == "probe":
        cmd_probe(args.rung, journal, args.device)
    elif args.stage == "baseline":
        cmd_baseline(args.rung, args.device)
    elif args.stage == "calibrate":
        calibrate_rung(args.rung, NICHE, floor_advisory=args.floor_advisory)
    elif args.stage == "arms":
        cmd_arms(args.rung, journal, args.device)
    elif args.stage == "extend":
        cmd_extend(args.rung, journal, args.device)
    elif args.stage == "rankpooled":
        cmd_rankpooled(args.rung, journal, args.device)
    else:
        cmd_verdict(args.rung, journal)

if __name__ == "__main__":
    main()
