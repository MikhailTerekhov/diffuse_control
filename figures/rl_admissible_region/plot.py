"""The admissible region reached by RL, beside the one reached by prompt optimization.

Every point on this plane is validation-scored on the same papers, with 95% bootstrap CIs
on both axes, so RL checkpoints and prompt-optimization frontier prompts are directly
comparable rather than being train-scored against val-scored.

One panel per weak scorer. The comparison that matters is between panels: whether RL can
open a gap against a scorer that prompt optimization could not.

    uv run python figures/rl_admissible_region/plot.py \
        --naive-adapters  data/val_adapters_naive.json \
        --naive-promptopt data/val_promptopt_naive.json \
        --robust-adapters  data/val_adapters_robust.json \
        --robust-promptopt data/val_promptopt_robust.json

Any of the four is optional; a panel with nothing in it is dropped. Paths are relative to
the repo root or absolute.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RL_COLOR = "#2E7D32"
PF_COLOR = "#A63D40"
PF_EDGE = "#5f2224"
RL_EDGE = "#14401a"


def boot(values, n_boot=10000, seed=42):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return None
    if a.size < 3:
        m = float(a.mean())
        return m, m, m
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(a, a.size, replace=True).mean() for _ in range(n_boot)])
    return float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def load(path):
    """{label: {weak, strong}} -> [(label, (s_mean, s_lo, s_hi), (w_mean, w_lo, w_hi))]"""
    if not path:
        return []
    d = json.loads(Path(path).read_text())
    out = []
    for label, r in d.items():
        s, w = boot(r["strong"]), boot(r["weak"])
        if s and w:
            out.append((label, s, w))
    return sorted(out, key=lambda p: p[1][0])


def draw(ax, adapters, promptopt, title, ylabel, annotate):
    if promptopt:
        ax.plot([s[0] * 100 for _, s, _ in promptopt], [w[0] * 100 for _, _, w in promptopt],
                "--", color=PF_COLOR, alpha=0.45, zorder=3)
    for kind, pts, color, edge, marker, size, legend in (
        ("pf", promptopt, PF_COLOR, PF_EDGE, "D", 8, "prompt optimization"),
        ("rl", adapters, RL_COLOR, RL_EDGE, "*", 16, "RL checkpoints"),
    ):
        for label, (sm, slo, shi), (wm, wlo, whi) in pts:
            ax.errorbar(sm * 100, wm * 100,
                        xerr=[[(sm - slo) * 100], [(shi - sm) * 100]],
                        yerr=[[(wm - wlo) * 100], [(whi - wm) * 100]],
                        fmt=marker, ms=size, color=color, ecolor=color, elinewidth=1.0,
                        capsize=2.5, markeredgecolor=edge, markeredgewidth=1.0,
                        zorder=6 if kind == "rl" else 5)
            if annotate:
                ax.annotate(label, (sm * 100, wm * 100), textcoords="offset points",
                            xytext=(7, 6), fontsize=8, color=edge)
        if pts:
            ax.scatter([], [], s=110, c=color, marker=marker, edgecolors=edge,
                       linewidths=1.0, label=f"{legend} (95% CI)")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Ground truth proxy score $g$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=10, framealpha=0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naive-adapters", default=None)
    ap.add_argument("--naive-promptopt", default=None)
    ap.add_argument("--robust-adapters", default=None)
    ap.add_argument("--robust-promptopt", default=None)
    ap.add_argument("--naive-label", default="default rubric")
    ap.add_argument("--robust-label", default="robustified rubric")
    ap.add_argument("--annotate", action="store_true", help="label each point")
    ap.add_argument("--out-base", default="figures/rl_admissible_region/rl_admissible_region")
    args = ap.parse_args()

    panels = []
    for adapters, promptopt, label in (
        (args.naive_adapters, args.naive_promptopt, args.naive_label),
        (args.robust_adapters, args.robust_promptopt, args.robust_label),
    ):
        a, p = load(adapters), load(promptopt)
        if a or p:
            panels.append((a, p, label))
    if not panels:
        raise SystemExit("nothing to plot: pass at least one adapters or promptopt file")

    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6.5), squeeze=False)
    for ax, (a, p, label) in zip(axes[0], panels):
        print(f"{label}: {len(a)} RL checkpoint(s), {len(p)} prompt(s)")
        draw(ax, a, p, f"Weak scorer: {label}", f"Weak score $w$ ({label})", args.annotate)
    fig.suptitle("Validation-scored admissible region: RL vs prompt optimization", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    Path(args.out_base).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(f"{args.out_base}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Saved {args.out_base}.{ext}")


if __name__ == "__main__":
    main()
