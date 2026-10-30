"""Figures for the length-wall diagnostics (D2, D3, D3-trace, D3-ops).

Reads the diag and study rows from a journal (default evidence/journal.jsonl,
so the figures rebuild from a fresh clone) -> numbers are never retyped by
hand. Read-only: never journals, never gates. Same palette and mark specs as
make_figures.py.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
JOURNAL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evidence" / "journal.jsonl"

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e6e5e1"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; NEUTRAL = "#8a897f"
PINK = "#d6409f"   # validated with AQUA, BLUE, ORANGE (adjacent CVD + normal-vision pass)

# D2 taxonomy in display order: the two "attempted three partials" classes,
# then the two "stopped at two" classes, then the rest.
CLASSES = (("CORRECT", "correct", AQUA),
           ("RIGHT_LENGTH_WRONG_VALUE", "3 partials, wrong value", BLUE),
           ("SHORT_PREFIX_CORRECT", "stopped at 2, prefix right", PINK),
           ("SHORT_PREFIX_WRONG", "stopped at 2, prefix wrong", ORANGE),
           ("LONG", "too long", NEUTRAL),
           ("MALFORMED", "malformed", NEUTRAL))
RUNG_LABEL = {1: "rung 1 (3.2M)", 3: "rung 3 (39.5M)"}


def rows():
    return [json.loads(l) for l in JOURNAL.read_text().splitlines() if l.strip()]


def pick(kind, key, value):
    return {r["rung"]: r for r in rows() if r.get("kind") == kind and r.get(key) == value}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def stacked(ax, bars):
    """bars: list of (label, counts dict). Horizontal 100% stacks, 2px gaps."""
    used = [c for c in CLASSES if any(b[1].get(c[0], 0) for b in bars)]
    for i, (label, counts) in enumerate(bars):
        n = sum(counts.values()); left = 0.0
        for key, _, color in used:
            w = counts.get(key, 0) / n
            if w:
                ax.barh(i, w, left=left, height=0.62, color=color,
                        edgecolor=SURFACE, linewidth=2)
                if w >= 0.07:
                    ax.text(left + w / 2, i, str(counts[key]), ha="center",
                            va="center", fontsize=8.5,
                            color="white" if color in (BLUE, ORANGE, AQUA, PINK) else INK)
            left += w
        ax.text(1.01, i, f"n={n}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    return used


def legend(fig, used, y):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in used]
    fig.legend(handles, [l for _, l, _ in used], loc="lower center", ncol=len(used),
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, y),
               handlelength=1.2, columnspacing=1.4)


def fig_d2():
    d2 = pick("diag", "diag", "D2-length-mechanism")
    bars = []
    for r in (1, 3):
        bars.append((f"{RUNG_LABEL[r]}\ntrained length", d2[r]["trained_depth_control"]["counts"]))
        bars.append((f"{RUNG_LABEL[r]}\none step longer", d2[r]["one_step_deeper"]["counts"]))
    attempted = sum(d2[r]["one_step_deeper"]["counts"][k] for r in (1, 3)
                    for k in ("CORRECT", "RIGHT_LENGTH_WRONG_VALUE"))
    total = sum(d2[r]["one_step_deeper"]["n"] for r in (1, 3))
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    style(ax)
    used = stacked(ax, bars)
    fig.suptitle(f"D2: one step longer than training, the model wrote three partials "
                 f"in {attempted} of {total} items", x=0.02, ha="left", fontsize=11.5, color=INK)
    ax.set_xlabel("share of decoded outputs, by class (registered taxonomy)", color=INK2, fontsize=9)
    legend(fig, used, -0.01)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / "v3-d2-length.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_d3():
    st = pick("study", "study", "D3")
    tr = pick("diag", "diag", "D3-trace")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1, 1.6]})
    style(a); a.grid(True, axis="y", color=GRID, linewidth=0.6)
    for x, r in enumerate((1, 3)):
        runs = st[r]["heldout_abcd_runs"]
        xs = [x + (i - (len(runs) - 1) / 2) * 0.05 for i in range(len(runs))]
        a.scatter(xs, runs, s=42, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        m, se = st[r]["heldout_abcd_mean"], st[r]["heldout_abcd_sem"]
        a.errorbar([x], [m], yerr=[se], color=INK, capsize=0, linewidth=1.6, zorder=4)
        a.hlines(m, x - 0.22, x + 0.22, color=INK, linewidth=2.2, zorder=4)
        a.text(x + 0.26, m, f"{m:.3f} ± {se:.3f}", va="center", fontsize=9, color=INK)
    a.text(0.5, -0.06, "length 5: 0.000 in 12/12 runs", ha="center", va="center",
           fontsize=8.5, color=ORANGE, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    a.set_xticks([0, 1], [RUNG_LABEL[1], RUNG_LABEL[3]])
    a.set_xlim(-0.5, 1.75); a.set_ylim(-0.1, 0.55)
    a.set_ylabel("held-out (a,b,c,d) exact match, per run", color=INK2, fontsize=9)
    a.set_title("trained on length 4, tested on a held-out combination",
                loc="left", fontsize=10, color=INK)
    style(b)
    used = stacked(b, [(f"{RUNG_LABEL[r]}\nheld-out (a,b,c,d)", tr[r]["heldout_abcd"]["counts"])
                       for r in (1, 3)] +
                      [(f"{RUNG_LABEL[r]}\nlength 5", tr[r]["len5"]["counts"]) for r in (1, 3)])
    b.set_title("what the outputs contain (all seeds pooled)", loc="left", fontsize=10, color=INK)
    fig.suptitle("D3: length-4 exposure carries to a new combination at rung 1, not at rung 3",
                 x=0.02, ha="left", fontsize=11.5, color=INK)
    legend(fig, used, -0.01)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / "v3-d3-exposure.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_d3_ops():
    ops = pick("diag", "diag", "D3-ops")
    cats = (("step2_is_x_c", "x c (the rule)", AQUA),
            ("step2_is_x_d", "x d (skips c)", ORANGE),
            ("both_c_and_d", "both", PINK),
            ("step2_other", "other", NEUTRAL))
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    style(ax); ax.grid(True, axis="y", color=GRID, linewidth=0.6); ax.grid(False, axis="x")
    w = 0.2
    for j, (key, label, color) in enumerate(cats):
        for x, r in enumerate((1, 3)):
            f = ops[r]["fractions"][key]
            ax.bar(x + (j - 1.5) * (w + 0.02), f, width=w, color=color, edgecolor=SURFACE,
                   linewidth=2, label=label if x == 0 else None)
            inside = f > 0.3   # tall bars carry the label inside, clear of the 60% line
            ax.text(x + (j - 1.5) * (w + 0.02), f - 0.015 if inside else f + 0.015,
                    f"{f:.0%}\n({ops[r]['counts'][key]})", ha="center",
                    va="top" if inside else "bottom", fontsize=8.5,
                    color="white" if inside else INK)
    ax.axhline(0.60, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(1.55, 0.61, "registered bar for\na mechanism (60%)", ha="right", va="bottom",
            fontsize=8.5, color=INK2)
    ax.set_xticks([0, 1], [f"{RUNG_LABEL[r]}\n{ops[r]['counts']['two_step']} two-step outputs"
                           for r in (1, 3)])
    ax.set_xlim(-0.55, 1.6); ax.set_ylim(0, 0.8)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8], ["0", "20%", "40%", "60%", "80%"])
    ax.set_ylabel("share of two-step outputs, by step-2 operation", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=4)
    fig.suptitle("D3-ops: in the two-step outputs, step 1 is a*b and step 2 tends to skip c",
                 x=0.02, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "v3-d3-ops.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    fig_d2(); fig_d3(); fig_d3_ops()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("v3-d*.png"))))
