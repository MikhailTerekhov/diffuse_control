"""One PF phase: drive red_team's main_loop with a custom W and seeds.

Approach: write seeded trials as JSON files (decent anchors + previous-cycle PF) that
v2's run.py knows how to load via --seed-prompts, configure weak_scorer_type=outer_evolve
pointing at the W_k JSON, and call main_loop(...). After completion, harvest F_k from
the cycle's checkpoint and refresh population manifest.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_pf_from_checkpoint(cycle_run_dir: Path):
    """Load rung-2 Pareto front prompts + per-paper sample_results."""
    from blue_team.bootstrap import load_source_pf
    return load_source_pf(cycle_run_dir)


def _build_seed_trial_json(prompt_text: str, w_label: str, samples: list,
                            rung_samples: list[int]) -> dict:
    """Construct a seed-trial JSON in the format _load_seed_trial expects."""
    import uuid
    valid = [s for s in samples
             if s.get("weak_score") is not None and s.get("strong_score") is not None]
    rung_scores = {}
    cum_n = 0
    for i, n in enumerate(rung_samples):
        sub = valid[:n]
        if not sub:
            continue
        w = sum(s["weak_score"] for s in sub) / len(sub)
        st = sum(s["strong_score"] for s in sub) / len(sub)
        rung_scores[str(i)] = [w, st, len(sub)]
        cum_n = n

    if not rung_scores:
        return None

    target_rung = max(int(k) for k in rung_scores)
    return {
        "trial_id": uuid.uuid4().hex[:12],
        "prompt_text": prompt_text,
        "justification": f"seeded from outer cycle ({w_label})",
        "parent_trial_id": None,
        "generation": -1,
        "island_id": 0,
        "current_rung": target_rung,
        "sample_results": valid,
        "rung_scores": {k: list(v) for k, v in rung_scores.items()},
        "status": "finished" if target_rung == len(rung_samples) - 1 else "stopped",
    }


async def _rejudge_under_w(
    population, w, w_label: str, max_connections: int,
) -> None:
    """Score every prompt-paper pair in `population` under W; cache as w_scores[w_label].

    Uses cached proposals (no generator calls). Skips pairs already scored under w_label.
    """
    from red_team import evaluator as v2_eval
    from blue_team.judge import score_one
    if not v2_eval._eval_config:
        v2_eval.init_models({
            "weak_judge": "anthropic/claude-haiku-4-5-20251001",
            "weak_judge_reasoning_tokens": 16384,
            "weak_scorer_type": "quality",
            "n_proposals": 10,
        }, max_connections)
    weak_judge = v2_eval._get_weak_judge()
    semaphore = asyncio.Semaphore(max_connections)

    pairs = []   # (prompt_hash, sample, proposal_path)
    for ph, p in population.prompts.items():
        for s in p.samples:
            if s.w_scores.get(w_label) is not None:
                continue
            pairs.append((ph, s, s.proposal_path))

    if not pairs:
        return

    logger.info(f"Re-judging {len(pairs)} (prompt, paper) pairs under {w_label}")
    coros = [score_one(weak_judge, semaphore, w, path, log_subrole="weak_judge_rejudge")
             for _, _, path in pairs]
    results = await asyncio.gather(*coros)
    for (ph, sample, _), (score, _sub) in zip(pairs, results):
        if score is not None:
            sample.w_scores[w_label] = score
    population.save()


async def run_pf_phase(
    config: dict,
    output_dir: Path,
    population,
    cycle: int,
    chosen_w_path: Path,
    chosen_w,
) -> dict:
    """Run one PF phase. Returns a summary dict.

    Refreshes anchor + previous-PF scores under W_k, then drives main_loop with
    seed prompts pointing at the rejudged trials.
    """
    cycle_dir = output_dir / f"cycle_{cycle}" / "pf_phase"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir = cycle_dir / "seed_trials"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    w_label = f"W_{cycle}"
    max_conn = config["max_connections"]
    new_target = config.get("pf_phase_target_highest_rung", 10)

    # Re-judge anchors + previous PFs under W_k (uses cached proposals).
    # Normally already done at the end of w_phase; this is a no-op safety net.
    await _rejudge_under_w(population, chosen_w, w_label, max_conn)

    # 2. Build seed-trial JSONs for previous-cycle PF + anchors
    rung_samples = list(config.get("pf_phase_rung_samples", [8, 32, 128]))
    seed_paths: list[Path] = []
    seed_pool = []
    seed_pool.extend(population.anchors())
    if cycle - 1 >= 0:
        seed_pool.extend(population.front(f"F_{cycle - 1}"))
    seen = set()
    for entry in seed_pool:
        if entry.prompt_hash in seen:
            continue
        seen.add(entry.prompt_hash)
        # Construct sample_results with weak_score under W_k
        sample_dicts = []
        for s in entry.samples:
            wv = s.w_scores.get(w_label)
            if wv is None or s.s_score is None:
                continue
            sample_dicts.append({
                "paper_id": s.paper_id,
                "weak_score": wv,
                "strong_score": s.s_score,
            })
        seed_json = _build_seed_trial_json(entry.prompt_text, w_label, sample_dicts, rung_samples)
        if seed_json is None:
            continue
        seed_path = seeds_dir / f"{entry.prompt_hash}.json"
        seed_path.write_text(json.dumps(seed_json, indent=2))
        seed_paths.append(seed_path)

    # Seeds count toward target_highest_rung in v2's scheduler. To get N NEW
    # rung-2 trials, set target = n_seeds + N.
    n_seeds = len(seed_paths)
    target_highest_rung = n_seeds + new_target
    logger.info(f"Cycle {cycle} PF phase: seeding with {n_seeds} prompts; "
                f"target rung-2 = {target_highest_rung} ({new_target} new + {n_seeds} seeds)")

    # 3. Construct v2 config for this cycle
    v2_config = {
        "task": "quality",
        "moasha": {
            "rung_samples": rung_samples,
            "eta": config.get("pf_phase_eta", 4),
            "target_highest_rung": target_highest_rung,
            "max_connections": max_conn,
            "n_islands": 3,
            "feature_bins": 4,
            "mode": "sandbagging",
            "migration_interval": 640,
            "migration_rate": 0.1,
            "population_size": 80,
            "archive_size": 16,
            "exploration_ratio": 0.3,
            "exploitation_ratio": 0.6,
            "promotion_window": 10,
        },
        "checkpoint_interval": 10,
        "split": config.get("split", {"train_ratio": 0.7, "seed": 42}),
        "optimizer": {
            "model": config["models"]["optimizer"],
            "reasoning_effort": config["models"].get("optimizer_reasoning_effort", "xhigh"),
            "n_inspiration_prompts": 3,
            "show_rubric": True,
            # Ablation: when true, the PF optimizer sees the full W prompt text
            # instead of only its summary. Off by default.
            "show_weak_judge_full_prompt": config.get(
                "show_weak_judge_full_prompt", False
            ),
        },
        "eval_models": {
            "generator": config["models"]["generator"],
            "generator_reasoning_effort": config["models"].get("generator_reasoning_effort", "high"),
            "weak_judge": config["models"]["weak_judge"],
            "weak_judge_reasoning_tokens": config["models"].get("weak_judge_reasoning_tokens", 16384),
            "n_proposals": 10,
            "weak_scorer_type": "outer_evolve",
            "outer_evolve_w_path": str(chosen_w_path),
            "strong_judge": config["models"]["strong_judge"],
            "strong_judge_reasoning_effort": config["models"].get("strong_judge_reasoning_effort", "high"),
            "strong_judge_has_paper": True,
            "strong_scorer_type": "quality",
            "_outer_w_label": w_label,
            "_outer_w_summary": chosen_w.summary,
        },
    }
    (cycle_dir / "v2_config.yaml").write_text(json.dumps(v2_config, indent=2))

    # 4. Reset evaluator caches and call main_loop
    from red_team import evaluator as v2_eval
    v2_eval._eval_config = {}
    v2_eval._models = {}
    v2_eval.reset_outer_evolve_w_cache()

    from red_team.run import main_loop, generate_split
    # Reuse the outer split.json
    split_path = output_dir / "split.json"
    target_split = cycle_dir / "split.json"
    if not target_split.exists():
        target_split.symlink_to(split_path)

    # Mid-PF-phase resume: if we have a partial v2 checkpoint, hand it to main_loop
    # so the scheduler picks up where it left off rather than restarting.
    cp_root = cycle_dir / "checkpoints"
    resume_from = None
    if cp_root.exists():
        cps = [p for p in cp_root.iterdir()
               if p.is_dir() and p.name.startswith("checkpoint_")]
        if cps:
            cps.sort(key=lambda p: int(p.name.rsplit("_", 1)[-1]))
            resume_from = cps[-1]
            logger.info(f"Cycle {cycle} PF phase: resuming from {resume_from.name}")

    await main_loop(
        config=v2_config,
        output_dir=cycle_dir,
        dry_run=False,
        resume_from=resume_from,
        seed_prompt_paths=seed_paths,
    )

    # 5. Harvest the new PF
    cp_dir = cycle_dir / "checkpoints" / "final"
    if not cp_dir.exists():
        cps = sorted([p for p in (cycle_dir / "checkpoints").iterdir() if p.is_dir()],
                     key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)
        cp_dir = cps[-1] if cps else None
    if cp_dir is None:
        raise RuntimeError(f"PF phase {cycle}: no checkpoints found")

    pf = _load_pf_from_checkpoint(cycle_dir)
    logger.info(f"Cycle {cycle}: harvested {len(pf)} PF prompts")

    # 6. Add new PF to population (with cached proposals from this cycle)
    from blue_team.population import CachedSample, PromptEntry, prompt_hash
    eval_details_dir = cycle_dir / "logs" / "eval_details"
    new_hashes: list[str] = []
    seed_hashes = {entry.prompt_hash for entry in seed_pool}
    for e in pf:
        ph = prompt_hash(e["prompt"])
        if ph in seed_hashes:
            continue   # skip seeds — they're already in population
        if ph in population.prompts:
            new_hashes.append(ph)
            continue
        samples = []
        for s in e["samples"]:
            paper_id = s["paper_id"]
            proposal_path = eval_details_dir / f"{e['trial_id']}_{paper_id}.json"
            if not proposal_path.exists():
                continue
            samples.append(CachedSample(
                paper_id=paper_id,
                proposal_path=str(proposal_path),
                s_score=s["strong_score"],
                w_scores={w_label: s["weak_score"]},
            ))
        if not samples:
            continue
        entry = PromptEntry(
            prompt_hash=ph,
            role=f"F_{cycle}",
            prompt_text=e["prompt"],
            summary="",   # filled later by Sonnet summary
            samples=samples,
        )
        population.add_prompt(entry)
        new_hashes.append(ph)

    # Front F_k = new prompts + retained seed prompts (for completeness)
    population.add_front(f"F_{cycle}", new_hashes)
    population.save()

    return {
        "cycle": cycle,
        "n_seeded": len(seed_paths),
        "n_pf": len(pf),
        "n_new_in_pf": len(new_hashes),
    }
