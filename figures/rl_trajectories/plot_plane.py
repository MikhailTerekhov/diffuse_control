"""RL trajectories drawn in the (proxy, weak) plane, one panel per weak scorer.

Each run starts at its anchor prompt's position and moves as RL maximizes the weak score.
Plotting it in the same plane as the prompt-optimization frontier shows whether RL reaches
territory prompt optimization could not, and how the two weak scorers differ: against the
default rubric the trajectory can climb while the proxy falls, against a robustified scorer
it cannot.

    uv run python figures/rl_trajectories/plot_plane.py \
        --naive-runs  "anchor A=grpo_anchA_naiveW" "anchor B=grpo_anchB_naiveW" \
        --robust-runs "anchor A=grpo_anchA_bstarW" "anchor B=grpo_anchB_bstarW" \
        --naive-pf  figures/rl_admissible_region/data/val_promptopt.json:naive \
        --robust-pf figures/rl_admissible_region/data/val_promptopt.json:robust

The --*-pf arguments are optional reference frontiers, given as `file.json:key` pointing at
output from figures/rl_admissible_region/eval_promptopt.py. Trajectories are on-policy
training scores; the frontier points are validation-scored with CIs, so treat the frontier
as a reference rather than a like-for-like comparison. For that, use
figures/rl_admissible_region/.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
PF_COLOR = "#A63D40"
PF_EDGE = "#5f2224"


def boot(values, n_boot=10000, seed=0):
    """(mean, lo, hi) as a 95% bootstrap CI, in 0-1 units."""
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return None
    if a.size == 1:
        m = float(a.mean())
        return m, m, m
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(a, a.size, replace=True).mean() for _ in range(n_boot)])
    return float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def load_pf(spec: str):
    """'file.json:key' -> [((s_mean, s_lo, s_hi), (w_mean, w_lo, w_hi)), ...]"""
    path, _, key = spec.partition(":")
    d = json.loads(Path(path).read_text())
    if key:
        d = d[key]
    pts = [(boot(r["strong"]), boot(r["weak"])) for r in d.values()]
    return sorted([p for p in pts if p[0] and p[1]], key=lambda p: p[0][0])


def run_series(run_dir: Path):
    """(step, proxy, weak) in percent, from metrics.jsonl."""
    f = run_dir / "metrics.jsonl"
    out = []
    for line in (f.read_text().splitlines() if f.exists() else []):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "env/all/weak_score" in r and "env/all/strong_score" in r:
            out.append((int(r.get("progress/step", len(out))),
                        r["env/all/strong_score"] * 100, r["env/all/weak_score"] * 100))
    return sorted(out)


def stitched(runs_root: Path, dirs: list[str]):
    combined, offset = [], 0
    for d in dirs:
        s = run_series(runs_root / d)
        if not s:
            print(f"  warn: no metrics in {runs_root / d}")
            continue
        first = s[0][0]
        combined += [(st - first + offset, g, w) for st, g, w in s]
        offset = combined[-1][0] + 1
    return combined


def gaussian_smooth(y, sigma=3.0):
    if sigma <= 0 or len(y) < 2:
        return y
    radius = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, radius, mode="reflect"), k, mode="valid")


def parse_runs(specs):
    out = []
    for spec in specs or []:
        label, _, dirs = spec.partition("=")
        if not dirs:
            label, dirs = spec, spec
        out.append((label.strip(), [d.strip() for d in dirs.split(",") if d.strip()]))
    return out


def draw_panel(ax, runs_root, runs, pf_spec, title, ylabel, sigma):
    if pf_spec:
        pf = load_pf(pf_spec)
        ax.plot([s[0] * 100 for s, _ in pf], [w[0] * 100 for _, w in pf],
                "--", color=PF_COLOR, alpha=0.45, zorder=3)
        for (sm, slo, shi), (wm, wlo, whi) in pf:
            ax.errorbar(sm * 100, wm * 100,
                        xerr=[[(sm - slo) * 100], [(shi - sm) * 100]],
                        yerr=[[(wm - wlo) * 100], [(whi - wm) * 100]],
                        fmt="D", ms=8, color=PF_COLOR, ecolor=PF_COLOR, elinewidth=1.0,
                        capsize=2.5, markeredgecolor=PF_EDGE, markeredgewidth=1.0, zorder=5)
        ax.scatter([], [], s=90, c=PF_COLOR, marker="D", edgecolors=PF_EDGE, linewidths=1.0,
                   label="prompt-opt frontier (val, 95% CI)")

    for i, (label, dirs) in enumerate(runs):
        s = stitched(runs_root, dirs)
        if not s:
            continue
        color = PALETTE[i % len(PALETTE)]
        g = np.array([gg for _, gg, _ in s])
        w = np.array([ww for _, _, ww in s])
        ax.plot(g, w, "-", color=color, alpha=0.20, lw=1.1, zorder=4)
        if len(s) >= 4:
            ax.plot(gaussian_smooth(g, sigma), gaussian_smooth(w, sigma), "-", color=color,
                    alpha=0.95, lw=2.6, zorder=6, solid_capstyle="round")
        ax.scatter([g[0]], [w[0]], s=150, c=color, marker="o", edgecolors="k", linewidths=1.4,
                   zorder=10, label=f"{label} start ({g[0]:.0f}, {w[0]:.0f})")
        ax.scatter([g[-1]], [w[-1]], s=340, c=color, marker="*", edgecolors="k", linewidths=1.5,
                   zorder=10, label=f"{label} step {s[-1][0]} ({g[-1]:.0f}, {w[-1]:.0f})")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Ground truth proxy score $g$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naive-runs", nargs="+", default=[], help="label=dir[,dir]")
    ap.add_argument("--robust-runs", nargs="+", default=[], help="label=dir[,dir]")
    ap.add_argument("--naive-pf", default=None, help="file.json[:key] reference frontier")
    ap.add_argument("--robust-pf", default=None, help="file.json[:key] reference frontier")
    ap.add_argument("--runs-root", type=Path, default=Path("rl_red_team/runs"))
    ap.add_argument("--naive-label", default="default rubric")
    ap.add_argument("--robust-label", default="robustified rubric")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--out-base", default="figures/rl_trajectories/plane")
    args = ap.parse_args()

    naive, robust = parse_runs(args.naive_runs), parse_runs(args.robust_runs)
    if not naive and not robust:
        raise SystemExit("give at least one --naive-runs or --robust-runs entry (label=dir)")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), sharex=True)
    draw_panel(axes[0], args.runs_root, naive, args.naive_pf,
               f"Weak scorer: {args.naive_label}",
               f"Weak score $w$ ({args.naive_label})", args.sigma)
    draw_panel(axes[1], args.runs_root, robust, args.robust_pf,
               f"Weak scorer: {args.robust_label}",
               f"Weak score $w$ ({args.robust_label})", args.sigma)
    fig.suptitle("Prompt-anchored RL maximizing the weak score, in the score plane", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    Path(args.out_base).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(f"{args.out_base}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Saved {args.out_base}.{ext}")


if __name__ == "__main__":
    main()
