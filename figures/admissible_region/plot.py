"""The admissible region of weak vs ground-truth-proxy scores.

Only the subversion frontier is required. Every other element is optional, so the figure
is useful from a single red team run and gains detail as you produce more:

  --pf         REQUIRED  subversion frontier: maximize weak, minimize ground truth proxy
  --goodhart   optional  Goodharting frontier from a strong_weak run, bounding the right
  --max-weak   optional  the single-objective max-weak point, which defines w* and w*-eps
  --baselines  optional  benign generator reference lines (see baselines.py)

Every input is a val_scores.json from eval_pf.py.

    uv run python figures/admissible_region/plot.py --pf red_team/runs/<ts>/generalization/val_scores.json

    uv run python figures/admissible_region/plot.py \
        --pf       red_team/runs/<sb>/generalization/val_scores.json \
        --goodhart red_team/runs/<gh>/generalization/val_scores.json \
        --max-weak figures/admissible_region/data/max_weak/val_scores.json \
        --baselines figures/admissible_region/data/baselines.json

Without --max-weak there is no measured maximum of the weak score, so w* falls back to the
highest-weak point on the frontier and the figure says so. USETEX=1 switches to the paper's
Helvetica + LaTeX rendering.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

USETEX = os.environ.get("USETEX") == "1"
if USETEX:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{helvet}"
                               r"\renewcommand{\familydefault}{\sfdefault}"
                               r"\usepackage{xcolor}\usepackage{bm}",
    })

SB_COLOR = "#A63D40"        # subversion frontier
GH_COLOR = "#3D6B9E"        # Goodharting frontier
MW_COLOR = "#E08E45"        # max-weak point
DEF_COLOR = "#5C5C5C"       # the default (benign) prompt
REF_COLOR = "#6B8E23"       # benign generator reference lines
EPS_COLOR = "#546E7A"       # w* / w*-eps guides


def _mean_pct(values) -> float:
    return 100.0 * float(np.mean(values))


def _bootstrap_lower(values, n=16, n_boot=10000, pct=2.5, seed=42) -> float:
    """Lower CI bound on the mean of n resamples: the paper's low-sample-size w*-eps.

    eps stands for the blue team's uncertainty about the weak score, approximated as an
    N=16 lower confidence bound (paper Section 4).
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(vals, size=n, replace=True).mean() for _ in range(n_boot)])
    return 100.0 * float(np.percentile(boots, pct))


def load_points(path: Path) -> list[dict]:
    """val_scores.json -> per-prompt points in score-percent units."""
    raw = json.loads(Path(path).read_text())
    pts = []
    for name, v in raw.items():
        if not v.get("weak") or not v.get("strong"):
            continue
        pts.append({
            "name": name,
            "w": _mean_pct(v["weak"]),
            "s": _mean_pct(v["strong"]),
            "weak_raw": v["weak"],
            "n": len(v["weak"]),
            "train_w": (100 * v["train_weak"]) if v.get("train_weak") is not None else None,
            "train_s": (100 * v["train_strong"]) if v.get("train_strong") is not None else None,
        })
    return sorted(pts, key=lambda p: p["s"])


def pareto(points: list[dict], maximize_strong: bool) -> list[dict]:
    """Non-dominated subset, in validation coordinates."""
    def dominates(f, e):
        if maximize_strong:
            return (f["w"] >= e["w"] and f["s"] >= e["s"] and (f["w"] > e["w"] or f["s"] > e["s"]))
        return (f["w"] >= e["w"] and f["s"] <= e["s"] and (f["w"] > e["w"] or f["s"] < e["s"]))
    front = [e for e in points if not any(dominates(f, e) for f in points if f is not e)]
    return sorted(front, key=lambda e: e["s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pf", required=True, type=Path,
                    help="val_scores.json for the subversion frontier")
    ap.add_argument("--goodhart", type=Path, default=None,
                    help="val_scores.json for a strong_weak (Goodharting) run")
    ap.add_argument("--max-weak", type=Path, default=None,
                    help="val_scores.json for the max-weak point")
    ap.add_argument("--baselines", type=Path, default=None,
                    help="baselines.json from baselines.py")
    ap.add_argument("--default-prompt-label", default="default",
                    help="entry name in --baselines to mark as the default prompt")
    ap.add_argument("--out-base", default="figures/admissible_region/admissible_region")
    ap.add_argument("--no-frontier-line", action="store_true",
                    help="scatter only, do not connect the frontier")
    args = ap.parse_args()

    sb_all = load_points(args.pf)
    if not sb_all:
        raise SystemExit(f"no scored prompts in {args.pf}")
    sb = pareto(sb_all, maximize_strong=False)
    print(f"subversion: {len(sb_all)} prompts, {len(sb)} on the validation frontier")

    gh = pareto(load_points(args.goodhart), maximize_strong=True) if args.goodhart else []
    if args.goodhart:
        print(f"goodharting: {len(gh)} on the validation frontier")

    mw = None
    if args.max_weak:
        pts = load_points(args.max_weak)
        if pts:
            mw = max(pts, key=lambda p: p["w"])
            print(f"max-weak: {mw['name']} weak={mw['w']:.1f} strong={mw['s']:.1f}")

    baselines = {}
    if args.baselines:
        baselines = json.loads(args.baselines.read_text())
        print(f"baselines: {len(baselines)} entries")

    # w* is the best achievable weak score. The max-weak run measures it directly; without
    # it, fall back to the highest-weak point we have and label the figure honestly.
    if mw is not None:
        w_star, w_star_src = mw["w"], "max-weak run"
        w_star_eps = _bootstrap_lower(mw["weak_raw"])
    else:
        top = max(sb_all, key=lambda p: p["w"])
        w_star, w_star_src = top["w"], "highest-weak frontier point"
        w_star_eps = _bootstrap_lower(top["weak_raw"])
    print(f"w* = {w_star:.1f} ({w_star_src}), w*-eps = {w_star_eps:.1f}")

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # Benign reference lines first, so they sit under the data.
    for label, entry in sorted(baselines.items(), key=lambda kv: kv[1].get("strong", 0)):
        s = 100 * entry["strong"] if entry["strong"] <= 1 else entry["strong"]
        is_default = label == args.default_prompt_label
        ax.axvline(s, color=DEF_COLOR if is_default else REF_COLOR, lw=1.8, alpha=0.85,
                   linestyle=(0, (1.5, 2)), zorder=1)
        ax.text(s, 0.02, label, transform=ax.get_xaxis_transform(), rotation=90,
                ha="right", va="bottom", fontsize=11, fontweight="bold",
                color=DEF_COLOR if is_default else REF_COLOR, zorder=2)

    # w* / w*-eps guides: the band inside which an attack counts as nearly-optimal.
    ax.axhline(w_star, color=EPS_COLOR, lw=1.8, ls="--", alpha=0.9, zorder=2)
    ax.axhline(w_star_eps, color=EPS_COLOR, lw=1.8, ls=(0, (4, 3)), alpha=0.9, zorder=2)
    ax.text(0.985, w_star, r"$w^*$", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=15, color=EPS_COLOR, fontweight="bold")
    ax.text(0.985, w_star_eps, r"$w^*\!-\!\epsilon$", transform=ax.get_yaxis_transform(),
            ha="right", va="top", fontsize=15, color=EPS_COLOR, fontweight="bold")

    if gh:
        if not args.no_frontier_line:
            ax.plot([p["s"] for p in gh], [p["w"] for p in gh], color=GH_COLOR,
                    lw=2.0, alpha=0.9, zorder=3)
        ax.scatter([p["s"] for p in gh], [p["w"] for p in gh], color=GH_COLOR, s=70,
                   edgecolor="black", lw=0.7, zorder=4, label="Goodharting Pareto")

    if not args.no_frontier_line and len(sb) > 1:
        ax.plot([p["s"] for p in sb], [p["w"] for p in sb], color=SB_COLOR,
                lw=2.2, alpha=0.95, zorder=3)
    # Dominated prompts stay visible but recede, so the frontier reads as a boundary.
    dominated = [p for p in sb_all if p not in sb]
    if dominated:
        ax.scatter([p["s"] for p in dominated], [p["w"] for p in dominated],
                   facecolor="none", edgecolor=SB_COLOR, s=45, lw=1.0, alpha=0.45,
                   zorder=3, label="Subversive (dominated)")
    ax.scatter([p["s"] for p in sb], [p["w"] for p in sb], color=SB_COLOR, s=80,
               edgecolor="black", lw=0.7, zorder=5, label="Subversion Pareto")

    if mw is not None:
        ax.scatter([mw["s"]], [mw["w"]], color=MW_COLOR, s=190, marker="o",
                   edgecolor="black", lw=1.0, zorder=6, label="Max-weak")

    ax.set_xlabel("Ground truth proxy score $g(r)$", fontsize=17)
    ax.set_ylabel("Weak score $w(b, r)$", fontsize=17)
    ax.tick_params(labelsize=14)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    note = "" if mw is not None else f"  (w* from the {w_star_src}: no max-weak run supplied)"
    ax.set_title(f"Admissible region, validation papers (n={sb_all[0]['n']}){note}",
                 fontsize=13, loc="left", color="#333333", pad=10)
    # Legend below the axes: the vertical baseline lines can land anywhere inside.
    ncol = 2 + (1 if gh else 0) + (1 if mw is not None else 0)
    ax.legend(fontsize=12, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, -0.13), ncol=ncol, columnspacing=1.6,
              handletextpad=0.5)

    Path(args.out_base).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = f"{args.out_base}.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
