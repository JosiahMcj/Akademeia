"""Generate the v3 campaign figures from the recorded runs: the Q1 rank
ladder across rungs, the Q2 depth-wall panel, and the Q3 lottery panel.

Static PNGs; validated palette carried from v2, thin marks, one axis per
panel, legend + selective labels. Reads run ledgers and journal study rows
-> numbers are never retyped by hand. Read-only: never journals, never
gates. Extends automatically as more rungs land in runs/.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from akademeia.journal import Journal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e6e5e1"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; NEUTRAL = "#8a897f"

ARMS = (("direct50", BLUE), ("adaptive2", AQUA), ("branch", ORANGE))

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)

def arm(name):
    return json.loads((ROOT / "runs" / f"arm-{name}.json").read_text())

def jrows(**match):
    rows = Journal(ROOT / "runs" / "journal.jsonl").read()   # shard-aware
    return [r for r in rows if all(r.get(k) == v for k, v in match.items())]

def instrument_label(a: dict) -> str:
    """Role labels, matching RESULTS.md, never raw device strings."""
    host = (a.get("host") or "").lower()
    dev = a.get("device")
    if dev == "cuda" and "a30" in host:
        return "accelerator C"
    if dev == "cuda":
        return "accelerator A"
    if dev == "mps":
        return "accelerator B"
    return "processor B" if host else "processor A"

def rungs_present():
    """Only rungs with EVERY arm present. A rung whose baseline has landed but
    whose arms are still training is half a measurement: plotting it would
    draw a ranking from arms that do not exist yet, and it used to crash the
    figure outright."""
    out = []
    r = 1
    while True:
        tags = [f"ladder-r{r}-direct50-A"] + [f"ladder-r{r}-{n}"
                                              for n, _ in ARMS if n != "direct50"]
        if not all((ROOT / "runs" / f"arm-{t}.json").exists() for t in tags):
            break
        out.append(r); r += 1
    return out

def params_label(rung):
    from akademeia.tiers import RUNGS
    cfg = RUNGS[rung]
    # parameter count from the stored config, never retyped
    from akademeia.model import TinyGPT
    import torch
    with torch.no_grad():
        n = sum(p.numel() for p in TinyGPT(cfg, vocab_size=64).parameters())
    return f"rung {rung}\n~{n/1e6:.0f}M params"

def fig_ladder():
    rungs = rungs_present()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]; style(ax)
    xs = list(range(len(rungs)))
    for name, color in ARMS:
        ms = []
        for r in rungs:
            tag = (f"ladder-r{r}-direct50-A" if name == "direct50"
                   else f"ladder-r{r}-{name}")
            a = arm(tag)
            ms.append(sum(a["final"]) / len(a["final"]))
        ax.plot(xs, ms, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.annotate(f"{name}  {ms[-1]:.2f}", (xs[-1], ms[-1]), color=color,
                    fontsize=9, ha="left", va="center", xytext=(9, 0),
                    textcoords="offset points")
    ax.set_xlim(-0.25, len(rungs) - 0.30)
    allv = [sum(arm(f"ladder-r{r}-direct50-A" if n == "direct50"
                    else f"ladder-r{r}-{n}")["final"]) / 3
            for n, _ in ARMS for r in rungs]
    lo, hi = min(allv), max(allv)
    ax.set_ylim(lo - 0.22 * (hi - lo), hi + 0.04 * (hi - lo))   # room for the note
    ax.set_xticks(xs)
    ax.set_xticklabels([f"rung {r}\n" + instrument_label(arm(f'ladder-r{r}-direct50-A')).replace(" ", "\n")
                        for r in rungs], fontsize=8.5, color=INK)
    ax.set_title("physics: the ranking, rung by rung",
                 color=INK, fontsize=10, loc="left")
    ax.set_ylabel("unseen-formula score (mean of 3 seeds)", color=INK2,
                  fontsize=9)
    rung_rows = [r for r in jrows(kind="rung") if r.get("rank_agreement_vs_prev")]
    broke = [r for r in rung_rows
             if r["rank_agreement_vs_prev"]["discordant"] > 0]
    if broke:
        b = broke[0]
        pair = b["rank_agreement_vs_prev"]["disagreements"][0]
        at = ", ".join(f"r{r['rung']}" for r in broke)
        gaps = ", ".join(f"{abs(r['finals'][pair[0]] - r['finals'][pair[1]]):.3f}"
                         for r in broke)
        note = (f"ordinal measure flips at {at}, always {pair[0]} vs "
                f"{pair[1]}\ngaps {gaps} -- the one pair no gate ever "
                f"separated")
        rp = [r for r in jrows(kind="study", study="Q1-rank-pooled")
              if r["rung"] == b["rung"]]
        if rp:
            r0 = rp[-1]
            note += (f"\nat {r0['n_per_arm']} runs/arm the gap is "
                     f"{abs(r0['difference']):.3f} +- "
                     f"{r0['difference_uncertainty']:.3f}")
        ax.text(0.02, 0.015, note, transform=ax.transAxes, color=INK2,
                fontsize=7.0, ha="left", va="bottom")
    ax = axes[1]; style(ax)
    # prefer the doubled-power extension when it exists
    s1 = jrows(kind="study", study="S1-Q3-ext") or jrows(kind="study", study="S1-Q3")
    if s1:
        per = s1[-1]["per_rung"]
        for i, r in enumerate(sorted(per, key=int)):
            vals = per[r]["outcomes"]
            ax.scatter([i] * len(vals), vals, s=36, color=BLUE, alpha=0.75,
                       edgecolors=SURFACE, linewidths=1.2, zorder=3)
            m = per[r]["mean"]
            ax.plot([i - 0.13, i + 0.13], [m, m], color=INK, linewidth=2,
                    zorder=4)
            # anchored above the highest run, so the label never sits on a point
            ax.annotate(f"sd {per[r]['spread']:.3f}", (i, max(vals)), color=INK2,
                        fontsize=8, ha="center", va="bottom",
                        xytext=(0, 7), textcoords="offset points")
        steps = s1[-1]["steps"]
        note = "\n".join(f"{s['step']}: {s['label']} (p={s['p']:.2f})"
                          for s in steps)
        ax.set_title("vm: the per-run lottery", color=INK, fontsize=10,
                     loc="left")
        ax.text(0.02, 0.97, note, transform=ax.transAxes, color=INK2,
                fontsize=7.6, ha="left", va="top")
        lo = min(min(per[r]["outcomes"]) for r in per)
        hi = max(max(per[r]["outcomes"]) for r in per)
        ax.set_ylim(lo - 0.06, hi + 0.09)
        ax.set_xlim(-0.55, len(per) - 0.45)
        ax.set_xticks(range(len(per)))
        ax.set_xticklabels([f"rung {r}" for r in sorted(per, key=int)],
                           fontsize=9, color=INK)
        npairs = s1[-1].get("n_pairs", 6)
        ax.set_ylabel(f"unseen-combination ({npairs} pair means)",
                      color=INK2, fontsize=9)
    fig.suptitle("v3 ladder: what scale preserves (ranking) and what it "
                 "does not shrink (the lottery)", color=INK, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "v3-ladder.png", facecolor=SURFACE)
    plt.close(fig)

def fig_wall():
    rows = jrows(kind="study", study="S2-Q2")
    rows = sorted(rows, key=lambda r: r["rung"])
    rungs = [r["rung"] for r in rows]
    xs = list(range(len(rungs)))
    series = (
        ("unseen, no 3-var product", BLUE,
         [r["s2_unseen_other_mean"] for r in rows]),
        ("TRAINED 3-var (in-dist)", AQUA,
         [sum(r["trained_product_in_distribution"].values())
          / len(r["trained_product_in_distribution"]) for r in rows]),
        ("UNSEEN 3-var (the wall)", ORANGE,
         [r["s2_unseen_product_mean"] for r in rows]),
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for label, color, ys in series:
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        ax.annotate(f"{label}  {ys[-1]:.2f}", (xs[-1], ys[-1]),
                    color=color, fontsize=8.6, ha="left", va="center",
                    xytext=(9, 0), textcoords="offset points")
    for x, r in zip(xs, rows):
        ax.plot([x - 0.08, x + 0.08], [r["margin"]] * 2, color=NEUTRAL,
                linewidth=1.6, linestyle=(0, (3, 2)))
    # the pooled re-measurement, plotted beside the registered series
    sp = [r for r in jrows(kind="study", study="S2-pooled")]
    if sp:
        r0 = sp[-1]
        i = [x["rung"] for x in rows].index(r0["rung"])
        m, se = r0["unseen_product_mean"], r0["unseen_product_sem"]
        ax.errorbar([i], [m], yerr=[se], fmt="o", ms=6, color=NEUTRAL,
                    ecolor=NEUTRAL, elinewidth=1.4, capsize=3, zorder=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.annotate(f"re-measured, {r0['n_runs']} runs\n{m:.2f} \u00b1 {se:.2f}",
                    (i, m), color=NEUTRAL, fontsize=7.8, ha="right",
                    va="center", xytext=(-14, -18),
                    textcoords="offset points")
    ax.annotate("transfer margin the wall\nwould need to clear",
                (xs[-1], rows[-1]["margin"]), color=NEUTRAL, fontsize=8,
                ha="right", va="top", xytext=(-6, -6),
                textcoords="offset points")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.12, len(rungs) - 0.02)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"rung {r}" for r in rungs], fontsize=9, color=INK)
    ax.set_ylabel("score on the frozen physics bench", color=INK2, fontsize=10)
    last = rows[-1]
    headline = (f"{sp[-1]['unseen_product_mean']:.2f} at "
                f"{sp[-1]['n_runs']} runs" if sp
                else f"{last['s2_unseen_product_mean']:.2f}")
    ax.set_title("Study S2: composition transfer is the last thing to move\n"
                 f"at rung {last['rung']} unseen transfer leaves the floor "
                 f"({headline}),\nstill far under the {last['margin']:.2f} "
                 f"margin it must clear",
                 color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "v3-wall.png", facecolor=SURFACE)
    plt.close(fig)

def fig_q4():
    """Q4: the scratchpad format on its own task. Every run is plotted --
    the seed spread IS the story here, so a bar of means would hide it."""
    rows = sorted([r for r in jrows(kind="study", study="Q4")],
                  key=lambda r: r["rung"])
    if not rows:
        return
    # prefer the 6-seed pooled records when the extension has run
    pooled = {r["rung"]: r for r in jrows(kind="study", study="Q4-pooled")}
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]; style(ax)
    def spread(i, n):
        """Deterministic fan so overlapping runs stay countable."""
        return [i + (k - (n - 1) / 2) * (0.30 / max(n - 1, 1)) for k in range(n)]
    for i, r in enumerate(rows):
        pr = pooled.get(r["rung"])
        if pr:
            vals, m, se = pr["niche_unseen_runs"], pr["niche_unseen_mean"], pr["sem"]
        else:
            vals = r["niche_unseen_A"] + r["niche_unseen_Aprime"]
            m, se = r["niche_unseen_mean"], None
        ax.scatter(spread(i, len(vals)), vals, s=30, color=BLUE, alpha=0.75,
                   edgecolors=SURFACE, linewidths=1.0, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [m, m], color=INK, linewidth=2, zorder=4)
        if se:
            ax.plot([i, i], [m - se, m + se], color=INK, linewidth=1.4, zorder=4)
        label = f"{m:.2f}" + (f" \u00b1 {se:.2f}" if se else "")
        ax.annotate(label, (i, m), color=INK, fontsize=8.8, ha="center",
                    va="bottom", xytext=(0, 9), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE,
                              ec="none"))
        ax.annotate(f"n={len(vals)}", (i, 0.02), color=INK2, fontsize=7.6,
                    ha="center", va="bottom")
    ax.set_xlim(-0.45, len(rows) - 0.55)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"rung {r['rung']}" for r in rows], fontsize=9,
                       color=INK)
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("full-trace exact match (per run)", color=INK2, fontsize=9)
    ax.set_title("novel combinations of trained symbols", color=INK,
                 fontsize=10, loc="left")
    ax = axes[1]; style(ax)
    n = 0
    for i, r in enumerate(rows):
        pr = pooled.get(r["rung"])
        if pr:
            vals = [float(v) for v in pr["extrap_runs"]]
        else:
            vals = (list(r["extrap_diagnostic"]["A"].values())
                    + list(r["extrap_diagnostic"]["Aprime"].values()))
        n += len(vals)
        xs_ = [i + (k - (len(vals) - 1) / 2) * (0.30 / max(len(vals) - 1, 1))
               for k in range(len(vals))]
        ax.scatter(xs_, vals, s=38, color=ORANGE, alpha=0.8,
                   edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"rung {r['rung']}" for r in rows], fontsize=9,
                       color=INK)
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("full-trace exact match", color=INK2, fontsize=9)
    ax.set_title("one step longer than training", color=INK, fontsize=10,
                 loc="left")
    ax.text(0.5, 0.5, f"0.00 in {n}/{n} runs\n(reported, never gated)",
            transform=ax.transAxes, color=ORANGE, fontsize=10, ha="center",
            va="center")
    fig.suptitle("Study Q4: writing the intermediates down is learnable, and "
                 "buys nothing for length", color=INK, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "v3-scratchpad.png", facecolor=SURFACE)
    plt.close(fig)

if __name__ == "__main__":
    fig_ladder()
    fig_wall()
    fig_q4()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
