"""Deliverable 2: run a single W phase against an existing bootstrap (F_0).

Usage:
    uv run python -m blue_team.scripts.run_one_w_phase \
        --run-dir blue_team/runs/<ts>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blue_team.diagnostics import (
    metrics_under_w,
    scatter_population,
)
from blue_team.population import Population
from blue_team.w_phase import run_w_phase

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "outer.log", mode="a"),
            logging.StreamHandler(),
        ],
    )
    for noisy in ("httpx", "httpcore", "anthropic", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def main_async(run_dir: Path):
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    population = Population.load(run_dir / "population.json")
    chosen_w, diag = await run_w_phase(config, run_dir, population, cycle=1)

    # Diagnostics: compare population's f_0 + anchors under W_0 vs W_1
    figures_dir = run_dir / "cycle_1" / "w_phase" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    project_figures = PROJECT_ROOT / "figures" / "outer_evolve"
    project_figures.mkdir(parents=True, exist_ok=True)

    for label in ("W_0", "W_1"):
        scatter_population(
            population, label,
            figures_dir / f"scatter_{label}.png",
            title=f"Population under {label} (after cycle 1 W phase)",
        )
        scatter_population(
            population, label,
            project_figures / f"cycle1_w_{label}.png",
            title=f"Population under {label} (after cycle 1 W phase)",
        )

    # Side-by-side
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
    role_colors = {"p_def": "black", "anchor": "tab:green", "F_0": "tab:red",
                   "F_1": "tab:orange", "F_2": "tab:purple", "F_3": "tab:blue"}
    for ax, label in zip(axes, ("W_0", "W_1")):
        for role in ("F_0", "anchor", "p_def"):
            xs, ys = [], []
            for p in population.prompts.values():
                if p.role != role:
                    continue
                s = p.s_mean
                w = p.w_mean(label)
                if s is None or w is None:
                    continue
                xs.append(s); ys.append(w)
            if xs:
                marker = "*" if role == "p_def" else ("D" if role == "anchor" else "o")
                size = 200 if role == "p_def" else (60 if role == "anchor" else 30)
                ax.scatter(xs, ys, c=role_colors[role], marker=marker, s=size,
                           alpha=0.7, edgecolors="black", linewidth=0.5, label=role)
        ax.set_xlabel("strong (s_mean)")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{label}: population scatter")
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("weak (w_mean)")
    fig.tight_layout()
    fig.savefig(figures_dir / "side_by_side_W0_vs_W1.png", dpi=120)
    fig.savefig(project_figures / "cycle1_W0_vs_W1.png", dpi=120)
    plt.close(fig)

    # Metrics
    p_def = population.prompts.get(population.p_def_hash)
    f0_plus_anchors = [p for p in population.prompts.values()
                       if p.role in ("F_0", "anchor", "p_def")]
    metrics = {}
    for label in ("W_0", "W_1"):
        metrics[label] = metrics_under_w(f0_plus_anchors, label, p_def)
    metrics["chosen_index"] = diag["chosen_index"]
    metrics["best_J"] = diag["best_J"]
    (run_dir / "cycle_1" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print("\n=== W phase complete ===")
    print(f"  chosen W_1 candidate: {diag['chosen_index']}  (J = {diag['best_J']:+.5f})")
    print("  Metrics under W_0 vs W_1 (over F_0 ∪ anchors):")
    for label in ("W_0", "W_1"):
        m = metrics[label]
        print(f"    {label}: pearson={m.get('pearson'):.4f} | "
              f"saddle_eps_0.05={m.get('sq_eps_0.05')}, "
              f"saddle_eps_0.10={m.get('sq_eps_0.10')}")
    print(f"  figures: {figures_dir / 'side_by_side_W0_vs_W1.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    setup_logging(run_dir)
    asyncio.run(main_async(run_dir))


if __name__ == "__main__":
    main()
