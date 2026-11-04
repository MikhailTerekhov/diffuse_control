"""Training trajectories of prompt-anchored RL, weak and proxy score over steps.

Three panels: weak score under each of the two weak scorers being trained against, then
the ground truth proxy score for all runs on shared axes. The question it answers is what
the proxy does while RL pushes the weak score up: against the default rubric the two come
apart, against the robustified scorer they move together.

Runs are given as label=dir pairs, so this works with whatever you have. A run that crashed
and was resumed can be stitched onto a continuous step axis by joining its directories with
a comma.

    uv run python figures/rl_trajectories/plot_overlay.py \
        --naive  "anchor A=grpo_anchA_naiveW" "anchor B=grpo_anchB_naiveW" \
        --robust "anchor A=grpo_anchA_bstarW" "anchor B=grpo_anchB_bstarW"

    # a crash-resume chain for one run
    --naive "anchor A=grpo_anchA_naiveW,grpo_anchA_naiveW_cont"

Reads rl_red_team/runs/<dir>/metrics.jsonl, which the trainer writes per step.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def run_series(run_dir: Path) -> list[tuple[int, float, float]]:
    """(step, weak, proxy) in score-percent units, from a run's metrics.jsonl."""
    f = run_dir / "metrics.jsonl"
    out = []
    for line in (f.read_text().splitlines() if f.exists() else []):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "env/all/weak_score" in r and "env/all/strong_score" in r:
            out.append((int(r.get("progress/step", len(out))),
                        r["env/all/weak_score"] * 100, r["env/all/strong_score"] * 100))
    return sorted(out)


def stitched(runs_root: Path, dirs: list[str]) -> list[tuple[int, float, float]]:
    """Concatenate a resume chain onto one continuous step axis."""
    combined, offset = [], 0
    for d in dirs:
        s = run_series(runs_root / d)
        if not s:
            print(f"  warn: no metrics in {runs_root / d}")
            continue
        first = s[0][0]
        combined += [(st - first + offset, w, g) for st, w, g in s]
        offset = combined[-1][0] + 1
    return combined


def gaussian_smooth(y, sigma=4.0):
    if sigma <= 0 or len(y) < 2:
        return y
    radius = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, radius, mode="reflect"), k, mode="valid")


def parse_runs(specs: list[str]) -> list[tuple[str, list[str]]]:
    """'label=dir1,dir2' -> ('label', ['dir1', 'dir2'])."""
    out = []
    for spec in specs or []:
        label, _, dirs = spec.partition("=")
        if not dirs:
            label, dirs = spec, spec
        out.append((label.strip(), [d.strip() for d in dirs.split(",") if d.strip()]))
    return out


def draw(ax, runs_root, runs, which, linestyle, suffix="", sigma=4.0):
    """which: 1 = weak, 2 = proxy."""
    for i, (label, dirs) in enumerate(runs):
        s = stitched(runs_root, dirs)
        if not s:
            continue
        step = np.array([st for st, _, _ in s])
        y = np.array([row[which] for row in s])
        color = PALETTE[i % len(PALETTE)]
        ax.plot(step, y, linestyle, color=color, alpha=0.12, lw=1.0)
        ax.plot(step, gaussian_smooth(y, sigma), linestyle, color=color, lw=2.6,
                solid_capstyle="round", label=label + suffix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naive", nargs="+", default=[],
                    help="label=dir[,dir] runs trained against the default weak scorer")
    ap.add_argument("--robust", nargs="+", default=[],
                    help="label=dir[,dir] runs trained against a robustified weak scorer")
    ap.add_argument("--runs-root", type=Path, default=Path("rl_red_team/runs"))
    ap.add_argument("--naive-label", default="default rubric")
    ap.add_argument("--robust-label", default="robustified rubric")
    ap.add_argument("--sigma", type=float, default=4.0, help="smoothing width in steps")
    ap.add_argument("--out-base", default="figures/rl_trajectories/overlay")
    args = ap.parse_args()

    naive, robust = parse_runs(args.naive), parse_runs(args.robust)
    if not naive and not robust:
        raise SystemExit("give at least one --naive or --robust run (label=dir)")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True)
    draw(ax1, args.runs_root, naive, 1, "--", sigma=args.sigma)
    draw(ax2, args.runs_root, robust, 1, "-", sigma=args.sigma)
    draw(ax3, args.runs_root, naive, 2, "--", f" - {args.naive_label}", sigma=args.sigma)
    draw(ax3, args.runs_root, robust, 2, "-", f" - {args.robust_label}", sigma=args.sigma)

    ax1.set_title(f"Weak score $w$, {args.naive_label} (dashed)", fontsize=12)
    ax2.set_title(f"Weak score $w$, {args.robust_label} (solid)", fontsize=12)
    ax3.set_title("Ground truth proxy score $g$", fontsize=12)
    ax1.set_ylabel("score (0-100)", fontsize=12)
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("training step", fontsize=12)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    fig.suptitle("Prompt-anchored RL maximizing the weak score, by weak scorer", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    Path(args.out_base).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(f"{args.out_base}.{ext}", dpi=150, bbox_inches="tight")
        print(f"Saved {args.out_base}.{ext}")


if __name__ == "__main__":
    main()
