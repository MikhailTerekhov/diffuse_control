"""Validate W phase against source-run F_0 alone (no anchors needed).

Builds a minimal population from `source_run`'s rung-2 PF (with cached
proposals + cached weak scores under W_0) plus the default initial_prompt
treated as a synthetic "p_def" entry (with no cached proposals — we set
its s/w means manually from a representative source-run sample).

Then runs ONE W candidate to verify the full pipeline (Opus → schema parse →
Haiku scoring → J computation) works end-to-end before the real bootstrap finishes.

Usage:
    uv run python -m blue_team.scripts.validate_w_phase \
        --source-run red_team/runs/20260418_223910 \
        --out blue_team/runs/_w_validate
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

OUTER_DIR = Path(__file__).resolve().parent.parent

from blue_team.bootstrap import register_source_pf
from blue_team.population import Population
from blue_team.schema import WPrompt, default_rubric_schema
from blue_team.w_phase import run_w_phase

logger = logging.getLogger(__name__)


def setup_logging(out: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    for noisy in ("httpx", "httpcore", "anthropic", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def main_async(source_run: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((OUTER_DIR / "config" / "base.yaml").read_text())
    # Cap connections lower since we share with other processes
    config["max_connections"] = 64
    config["n_w_candidates"] = 1   # just 1 candidate to validate pipeline
    config["papers_per_prompt_w_phase"] = 8

    # Build a tiny population from source PF
    population = Population(manifest_path=out / "population.json")
    register_source_pf(population, source_run, w0_label="W_0")

    # Save W_0
    bootstrap_dir = out / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    w0 = WPrompt(label="W_0", summary="Standard rubric judge with 8 quality dimensions and two penalty fields.",
                 prompt_text="(W_0 system prompt — see source for full text)",
                 score_schema=default_rubric_schema())
    # Use real W_0 text
    from eval_harness.experiments_quality_eval import JUDGE_SYSTEM_BLIND
    w0.prompt_text = JUDGE_SYSTEM_BLIND
    w0_path = bootstrap_dir / "W_0.json"
    w0.save(w0_path)
    population.add_weak_prompt("W_0", str(w0_path), w0.summary)

    # Take 5 PF prompts (stratified-ish: just take indices 0, len/4, len/2, 3len/4, len-1)
    f0_hashes = population.fronts.get("F_0", [])
    n = len(f0_hashes)
    if n > 5:
        idxs = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        keep = [f0_hashes[i] for i in idxs]
    else:
        keep = f0_hashes
    population.fronts["F_0"] = keep
    # Also drop other prompts from manifest
    keep_set = set(keep)
    population.prompts = {h: p for h, p in population.prompts.items() if h in keep_set}

    # Add summaries (stub — to avoid Sonnet calls)
    for p in population.prompts.values():
        p.summary = f"[stub summary for {p.role}]"

    # Manually set p_def_hash to first prompt (as a stand-in — only used for diagnostics here)
    population.p_def_hash = keep[0]
    population.prompts[keep[0]].role = "p_def"

    population.save()
    logger.info(f"Built mini-population: {len(population.prompts)} prompts under W_0")

    # Now run a single W candidate
    chosen_w, diag = await run_w_phase(config, out, population, cycle=1)
    print(f"\n=== Validation result ===")
    print(f"  chosen J = {diag['best_J']:+.5f}")
    print(f"  chosen W summary: {chosen_w.summary[:200]}")
    print(f"  chosen W prompt[:200]: {chosen_w.prompt_text[:200]}")
    print(f"  candidates: {len(diag['candidates'])}")
    for c in diag["candidates"]:
        print(f"    [{c.get('index')}] J={c.get('J')} fail_rate={c.get('fail_rate')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default="red_team/runs/20260418_223910")
    parser.add_argument("--out", default="blue_team/runs/_w_validate")
    args = parser.parse_args()
    src = (PROJECT_ROOT / args.source_run).resolve()
    out = (PROJECT_ROOT / args.out).resolve()
    setup_logging(out)
    asyncio.run(main_async(src, out))


if __name__ == "__main__":
    main()
