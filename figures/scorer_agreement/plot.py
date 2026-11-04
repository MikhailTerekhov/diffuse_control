"""Scorer agreement and variance decomposition.

Left:  ground truth proxy vs weak score, 2D histogram with a 1-sigma ellipse per
       generator. The scores correlate across generators but barely within one.
Right: variance decomposition of each score, plus the covariance decomposition of the
       pair, into generator / sample / generator-noise / judge-noise. Each column sums
       to 100%, with 95% bootstrap CIs available via --print-ci.

Needs figures/scorer_agreement/data.json from extract.py. Writes figures/scorer_agreement/scorer_agreement.{png,pdf}.

Set USETEX=1 for the paper's Helvetica + LaTeX rendering (needs a working latex and
dvipng). The default is mathtext, which looks close enough and needs no LaTeX.

Usage:
  uv run python figures/scorer_agreement/plot.py
  USETEX=1 uv run python figures/scorer_agreement/plot.py --print-ci
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Ellipse, Patch

USETEX = os.environ.get("USETEX") == "1"
if USETEX:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{helvet}"
                               r"\renewcommand{\familydefault}{\sfdefault}"
                               r"\usepackage{amsfonts}",
    })


def pct(v):
    return rf"{v:.0f}\%" if USETEX else f"{v:.0f}%"


MODEL_LABELS = {
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "claude-opus-4-1": "Opus 4.1",
    "claude-opus-4-6": "Opus 4.6",
    "gpt-oss-20b": "GPT-oss-20b",
    "gpt-oss-120b": "GPT-oss-120b",
    "gpt-5.4-nano": "GPT-5.4-nano",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "gpt-5.4": "GPT-5.4",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
}

# Weakest to strongest, which is also the ellipse colour order.
MODEL_ORDER = [
    "claude-haiku-4-5-20251001",
    "gpt-oss-20b",
    "gpt-5.4-nano",
    "gemini-3.1-flash-lite-preview",
    "gpt-5.4-mini",
    "gpt-oss-120b",
    "gemini-3.1-pro-preview",
    "claude-opus-4-1",
    "claude-opus-4-6",
    "gpt-5.4",
]

# Paul Tol "muted" plus a vibrant orange: more separable than tab10, roughly CVD-safe.
COLORS = ["#332288", "#88CCEE", "#44AA99", "#117733", "#8B4513",
          "#999933", "#CC6677", "#882255", "#AA4499", "#EE7733"]

COMP_COLORS = {
    "Generator":       "#D4A574",
    "Sample":          "#7BA7C9",
    "Generator noise": "#8FBE8F",
    "Judge noise":     "#CC7070",
}

# Hand-placed so the labels miss the ellipses. Data units.
LABEL_OFFSETS = {
    "claude-haiku-4-5-20251001":     (6, -9),
    "gpt-oss-20b":                   (-9, 8),
    "gpt-5.4-nano":                  (8, -13),
    "gemini-3.1-flash-lite-preview": (10, -9),
    "gpt-5.4-mini":                  (-3, 9),
    "gpt-oss-120b":                  (-6, 10),
    "gemini-3.1-pro-preview":        (10, -7),
    "claude-opus-4-1":               (-5, -13),
    "claude-opus-4-6":               (7, -7),
    "gpt-5.4":                       (6, 8),
}


# --- decomposition (Appendix C.1) ---------------------------------------------------
# A and B share generated outputs and differ only in the judge call, so Var(A-B)/2 is
# judge noise. A and C share tasks but regenerate, so Cov(A,B) - Cov(A,C) is generator
# noise. Cov(A,C) splits into between-generator and within-generator (sample) parts.

def decompose_4way(a, b, c, gen_labels):
    var_judge = np.var(a - b) / 2
    var_gen_noise = max(0, np.cov(a, b, ddof=0)[0, 1] - np.cov(a, c, ddof=0)[0, 1])
    unique = sorted(set(gen_labels))
    means_a = np.array([a[gen_labels == g].mean() for g in unique])
    means_c = np.array([c[gen_labels == g].mean() for g in unique])
    var_gen = np.cov(means_a, means_c, ddof=0)[0, 1]
    within = [np.cov(a[gen_labels == g], c[gen_labels == g], ddof=0)[0, 1] for g in unique]
    return [max(0, var_gen), max(0, np.mean(within)), var_gen_noise, var_judge]


def decompose_cov_3way(gx, wy, wy_c, gen_labels):
    """Covariance of the two scores. Judge noise cancels, so only three components."""
    unique = sorted(set(gen_labels))
    mx = np.array([gx[gen_labels == g].mean() for g in unique])
    my = np.array([wy[gen_labels == g].mean() for g in unique])
    cov_gen = np.cov(mx, my, ddof=0)[0, 1]
    within_ac = np.mean([np.cov(gx[gen_labels == g], wy_c[gen_labels == g], ddof=0)[0, 1]
                         for g in unique])
    within_aa = np.mean([np.cov(gx[gen_labels == g], wy[gen_labels == g], ddof=0)[0, 1]
                         for g in unique])
    return [float(cov_gen), float(within_ac), float(within_aa - within_ac)]


def bootstrap(fn, arrays, gen_labels, n_boot=2000, seed=42):
    """Resample papers within each generator, preserving the panel structure."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(gen_labels))
    idx_by_gen = {g: np.where(gen_labels == g)[0] for g in unique}
    out = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(idx_by_gen[g], size=len(idx_by_gen[g]), replace=True)
                             for g in unique])
        out.append(fn(*[a[bi] for a in arrays], gen_labels[bi]))
    return np.array(out)


# --- panels -------------------------------------------------------------------------

def value_centered_bins(v):
    """Bin edges placed between observed values, so integer scores land in their own bin."""
    u = np.sort(np.unique(v))
    mids = (u[:-1] + u[1:]) / 2
    return np.concatenate([[u[0] - (mids[0] - u[0])], mids, [u[-1] + (u[-1] - mids[-1])]])


def confidence_ellipse(x, y, ax, n_std=1.0, **kw):
    mx, my = x.mean(), y.mean()
    vals, vecs = np.linalg.eigh(np.cov(x, y))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse((mx, my), w, h, angle=angle, **kw))
    return mx, my


def render_left(ax, data):
    g = np.array([r["g_A"] for r in data])
    w = np.array([r["w_A"] for r in data])
    gens = np.array([r["generator"] for r in data])

    ax.hist2d(g, w, bins=[value_centered_bins(g), value_centered_bins(w)],
              cmap="Greys", cmin=1)
    slope, intercept = np.polyfit(g, w, 1)
    xf = np.linspace(25, 90, 100)
    ax.plot(xf, slope * xf + intercept, ls="--", color="black", alpha=0.6, lw=1.5)
    ax.set_xlim(25, 90)
    ax.set_ylim(50, 100)
    ax.set_xlabel("Ground truth proxy score $g(t,o)$", fontsize=21)
    ax.set_ylabel("Weak score $w_b(t,o)$", fontsize=21)
    ax.tick_params(labelsize=19)
    if USETEX:
        fmt = plt.matplotlib.ticker.FuncFormatter(lambda v, _: rf"\textsf{{{v:g}}}")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)

    ax.text(0.03, 0.97, f"Joint corr. $r={np.corrcoef(g, w)[0, 1]:.2f}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=19, zorder=7,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="black", lw=0.8, alpha=0.95))

    for model, color in zip(MODEL_ORDER, COLORS):
        sel = gens == model
        if not sel.any():
            continue
        gx, wy = g[sel], w[sel]
        mx, my = confidence_ellipse(gx, wy, ax, edgecolor=color, facecolor="none",
                                    lw=2.0, alpha=0.95)
        ax.scatter([mx], [my], color=color, s=45, edgecolor="black", lw=0.6, zorder=5)
        dx, dy = LABEL_OFFSETS.get(model, (0, 0))
        ax.text(mx + dx, my + dy,
                f"{MODEL_LABELS.get(model, model)}\n(r={np.corrcoef(gx, wy)[0, 1]:.2f})",
                color=color, fontsize=18, fontweight="bold", ha="center", va="center",
                zorder=6, bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                    edgecolor="none", alpha=0.6))


def compute_right(data):
    gens = np.array([r["generator"] for r in data])
    ga, gb, gc = (np.array([r[k] for r in data]) for k in ("g_A", "g_B", "g_C"))
    wa, wb, wc = (np.array([r[k] for r in data]) for k in ("w_A", "w_B", "w_C"))
    return gens, (ga, gb, gc), (wa, wb, wc)


def render_right(ax, data, ax_legend=None):
    gens, (ga, gb, gc), (wa, wb, wc) = compute_right(data)

    def to_pct(v):
        v = np.array(v, dtype=float)
        return 100 * v / v.sum()

    groups = [
        ("Ground truth\nproxy score", list(COMP_COLORS), to_pct(decompose_4way(ga, gb, gc, gens))),
        ("Weak score", list(COMP_COLORS), to_pct(decompose_4way(wa, wb, wc, gens))),
        ("Covariance of\nthe two scores", list(COMP_COLORS)[:3], to_pct(decompose_cov_3way(ga, wa, wc, gens))),
    ]

    bar_w = 0.42
    xs = np.arange(len(groups)) * (bar_w + 0.64)
    for x, (_, comps, vals) in zip(xs, groups):
        bottom = 0.0
        for comp, val in zip(comps, vals):
            ax.bar(x, val, bar_w, bottom=bottom, color=COMP_COLORS[comp],
                   edgecolor="white", lw=0.7)
            if val >= 4:
                ax.text(x, bottom + val / 2, pct(val), ha="center", va="center",
                        fontsize=18, fontweight="bold", color="black")
            bottom += val

    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=19, fontweight="bold")
    ax.set_xlim(xs[0] - bar_w, xs[-1] + bar_w)
    ax.set_ylabel("Share of total (co-)variance", fontsize=21)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(decimals=0))
    ax.tick_params(axis="y", labelsize=19)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.5)
    ax.set_axisbelow(True)

    handles = [Patch(fc=c, label=k) for k, c in COMP_COLORS.items()]
    target = ax_legend if ax_legend is not None else ax
    if ax_legend is not None:
        target.axis("off")
    target.legend(handles=handles, loc="center", ncol=2, fontsize=18.5, frameon=False,
                  handlelength=1.6, columnspacing=1.0, handletextpad=0.5)


def print_ci(data, n_boot):
    gens, (ga, gb, gc), (wa, wb, wc) = compute_right(data)
    comps4 = list(COMP_COLORS)
    for label, fn, arrays, comps in [
        ("Ground truth proxy score", decompose_4way, (ga, gb, gc), comps4),
        ("Weak score", decompose_4way, (wa, wb, wc), comps4),
        ("Covariance", decompose_cov_3way, (ga, wa, wc), comps4[:3]),
    ]:
        boot = bootstrap(fn, arrays, gens, n_boot=n_boot)
        shares = 100 * boot / boot.sum(axis=1, keepdims=True)
        point = np.array(fn(*arrays, gens))
        point = 100 * point / point.sum()
        print(f"\n{label}  (n_boot={n_boot})")
        for i, comp in enumerate(comps):
            lo, hi = np.percentile(shares[:, i], [2.5, 97.5])
            print(f"  {comp:16s} {point[i]:5.1f}%   95% CI [{lo:5.1f}, {hi:5.1f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="figures/scorer_agreement/data.json")
    ap.add_argument("--out-base", default="figures/scorer_agreement/scorer_agreement")
    ap.add_argument("--print-ci", action="store_true",
                    help="print bootstrap CIs for the right panel")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    data = json.loads(open(args.data).read())
    print(f"{len(data)} rows, {len(set(r['generator'] for r in data))} generators")

    fig = plt.figure(figsize=(19, 7.0))
    gs = GridSpec(1, 2, width_ratios=[3, 1.75], wspace=0.4, figure=fig,
                  left=0.032, right=0.985, top=0.96, bottom=0.13)
    ax_left = fig.add_subplot(gs[0, 0])
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], height_ratios=[1, 5], hspace=0.05)
    ax_legend = fig.add_subplot(sub[0, 0])
    ax_right = fig.add_subplot(sub[1, 0])

    render_left(ax_left, data)
    render_right(ax_right, data, ax_legend=ax_legend)

    for ext in ("png", "pdf"):
        out = f"{args.out_base}.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close()

    if args.print_ci:
        print_ci(data, args.n_boot)


if __name__ == "__main__":
    main()
