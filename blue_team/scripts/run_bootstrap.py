"""Phase 0 CLI: build initial population.

Usage:
    uv run python -m blue_team.scripts.run_bootstrap
    uv run python -m blue_team.scripts.run_bootstrap --config base.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blue_team.bootstrap import run_bootstrap
from blue_team.diagnostics import (
    scatter_anchor_candidates,
    scatter_bootstrap_overview,
)
from blue_team.population import Population

OUTER_DIR = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "outer.log"),
            logging.StreamHandler(),
        ],
    )
    for noisy in ("httpx", "httpcore", "anthropic", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="base.yaml")
    parser.add_argument("--out", default=None,
                        help="Output run directory (default: blue_team/runs/<ts>/)")
    args = parser.parse_args()

    cfg_path = OUTER_DIR / "config" / args.config
    config = yaml.safe_load(cfg_path.read_text())

    if args.out:
        output_dir = Path(args.out).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTER_DIR / "runs" / ts

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

    setup_logging(output_dir)
    logger.info(f"Run directory: {output_dir}")

    population = asyncio.run(run_bootstrap(config, output_dir))

    # Build anchor-candidate scatter plot from the full candidate list (kept + dropped)
    from blue_team.population import PromptEntry, CachedSample
    cand_dump_path = output_dir / "bootstrap" / "candidate_entries.json"
    if cand_dump_path.exists():
        candidate_entries = []
        for d in json.loads(cand_dump_path.read_text()):
            e = PromptEntry(
                prompt_hash=d["prompt_hash"],
                role=d["role"],
                prompt_text=d["prompt_text"],
                summary=d.get("summary", ""),
                samples=[CachedSample(**s) for s in d.get("samples", [])],
            )
            # Re-tag for plotting purposes
            if d["role"] == "p_def_candidate":
                e.role = "p_def"
            elif d["role"] == "anchor_candidate":
                e.role = "anchor" if d.get("kept") else "anchor_dropped"
            candidate_entries.append(e)
    else:
        candidate_entries = list(population.prompts.values())

    p_def = next((e for e in candidate_entries if e.role == "p_def"), None)
    if p_def is None:
        p_def = population.prompts.get(population.p_def_hash)
    if p_def is not None and p_def.s_mean is not None:
        threshold = config.get("anchor_s_threshold", 0.10)
        figure_path = PROJECT_ROOT / "figures" / "outer_evolve" / "bootstrap_anchor_spread.png"
        run_figure_path = output_dir / "bootstrap" / "anchor_spread.png"
        for fp in (figure_path, run_figure_path):
            scatter_anchor_candidates(
                candidate_entries=candidate_entries,
                s_def=p_def.s_mean,
                threshold=threshold,
                out_path=fp,
                title="Phase 0 anchors (kept vs dropped by |s - s_def| filter)",
            )
        # Overview combining F_0 + anchors
        ov_path = PROJECT_ROOT / "figures" / "outer_evolve" / "bootstrap_overview.png"
        run_ov_path = output_dir / "bootstrap" / "overview.png"
        for fp in (ov_path, run_ov_path):
            scatter_bootstrap_overview(
                population=population,
                candidate_entries=candidate_entries,
                s_def=p_def.s_mean,
                threshold=threshold,
                out_path=fp,
            )

    n_anchors = sum(1 for p in population.prompts.values() if p.role == "anchor")
    n_f0 = len(population.fronts.get("F_0", []))
    print(f"\n=== Bootstrap complete ===")
    print(f"  anchors: {n_anchors}")
    print(f"  F_0:     {n_f0}")
    print(f"  manifest: {output_dir / 'population.json'}")
    print(f"  figures: {PROJECT_ROOT / 'figures' / 'outer_evolve' / 'bootstrap_anchor_spread.png'}")


if __name__ == "__main__":
    main()
