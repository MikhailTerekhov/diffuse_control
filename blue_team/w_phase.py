"""One W phase: propose 3 candidates, score each on P*, pick best by J."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from blue_team.outer_state import OuterState
from blue_team.population import (
    Population,
    PromptEntry,
    sample_papers_for_w_phase,
    select_p_star,
)
from blue_team.schema import WPrompt
from blue_team.w_evaluator import (
    PARSE_FAIL_THRESHOLD,
    aggregate_per_prompt,
    compute_J,
    eval_w_on_population,
    evaluate_candidate,
    pair_per_prompt,
)
from blue_team.w_optimizer import WOptimizer

logger = logging.getLogger(__name__)


async def run_w_phase(
    config: dict,
    output_dir: Path,
    population: Population,
    cycle: int,
    outer_state: OuterState | None = None,
) -> tuple[WPrompt, dict]:
    """Run one W phase. Returns (chosen W_k, candidate diagnostics dict).

    cycle = 1 for the very first W phase (selecting W_1 from W_0+F_0).
    """
    phase_dir = output_dir / f"cycle_{cycle}" / "w_phase"
    phase_dir.mkdir(parents=True, exist_ok=True)

    prev_w_label = f"W_{cycle - 1}"
    prev_w = WPrompt.load(population.weak_prompts[prev_w_label]["path"])
    prev_w.summary = population.weak_prompts[prev_w_label]["summary"] or prev_w.summary

    # Assemble full weak-judge history [W_0, ..., W_{cycle-1}] for the optimizer.
    # We later attach per-prompt raw scores (raw_w, raw_s) once we know p_star
    # and the paper subsample.
    weak_history: list[dict] = []
    for i in range(cycle):
        label = f"W_{i}"
        wp = population.weak_prompts.get(label)
        if not wp:
            continue
        wprompt = WPrompt.load(wp["path"])
        weak_history.append({
            "label": label,
            "summary": wp.get("summary") or wprompt.summary,
            "prompt_text": wprompt.prompt_text,
        })

    # Build P*
    p_star = select_p_star(
        population, cycle,
        n_top_per_past=config.get("n_top_per_past_pf", 2),
        n_top_current=config.get("n_top_current_pf", 2),
        n_sparse_current=config.get("n_sparse_current_pf", 5),
        n_sparse_past=config.get("n_sparse_past_pf", 3),
        w_label_for_topw=prev_w_label,
    )
    logger.info(f"Cycle {cycle} W phase: P* has {len(p_star)} prompts")

    # Sample 32 papers per prompt (fixed across all candidates in this phase)
    papers_per = config.get("papers_per_prompt_w_phase", 32)
    paper_subsample = sample_papers_for_w_phase(p_star, papers_per, seed=cycle)
    (phase_dir / "p_star.json").write_text(json.dumps([
        {"prompt_hash": p.prompt_hash, "role": p.role, "summary": p.summary,
         "s_mean": p.s_mean, "w_mean_prev": p.w_mean(prev_w_label),
         "n_papers_subsample": len(paper_subsample.get(p.prompt_hash, []))}
        for p in p_star
    ], indent=2))

    # Init Haiku judge
    from red_team import evaluator as v2_eval
    if not v2_eval._eval_config:
        v2_eval.init_models({
            "weak_judge": config["models"]["weak_judge"],
            "weak_judge_reasoning_tokens": config["models"].get("weak_judge_reasoning_tokens", 16384),
            "weak_scorer_type": "quality",
            "n_proposals": 10,
        }, config["max_connections"])
    weak_judge = v2_eval._get_weak_judge()

    semaphore = asyncio.Semaphore(config["max_connections"])
    optimizer = WOptimizer(config=config, log_dir=phase_dir / "optimizer_calls")

    # Raw per-prompt-per-paper s_scores (shared across all W's — strong is fixed).
    # Dedupe by paper_id since cached samples can have duplicates (v2 MO-ASHA stores
    # cumulative samples across rungs).
    sub_set_per_prompt = {ph: set(pids) for ph, pids in paper_subsample.items()}

    def _dedup_pairs(pairs):
        seen, out = set(), []
        for pid, v in pairs:
            if pid in seen:
                continue
            seen.add(pid)
            out.append((pid, v))
        return out

    raw_s = {
        p.prompt_hash: _dedup_pairs(
            (s.paper_id, s.s_score) for s in p.samples
            if s.paper_id in sub_set_per_prompt.get(p.prompt_hash, set())
            and s.s_score is not None
        )
        for p in p_star
    }

    # Attach raw_w to each weak_history entry from cached population scores
    for h in weak_history:
        label = h["label"]
        h["raw_w"] = {
            p.prompt_hash: _dedup_pairs(
                (s.paper_id, s.w_scores.get(label)) for s in p.samples
                if s.paper_id in sub_set_per_prompt.get(p.prompt_hash, set())
                and s.w_scores.get(label) is not None
            )
            for p in p_star
        }

    # Baseline J = prev_W's J on this phase's P*. Any candidate with J >= this
    # is rejected; if none beat it, we roll back and keep prev_W as W_cycle.
    prev_raw_w = next((h["raw_w"] for h in weak_history if h["label"] == prev_w_label), {})
    baseline_paired = pair_per_prompt(p_star, prev_raw_w, raw_s)
    baseline_J = compute_J(p_star, baseline_paired)
    logger.info(f"Cycle {cycle} W phase: baseline J = {baseline_J:.5f} "
                f"({prev_w_label} on this P*); candidates must beat this")

    prior: list[dict] = []
    candidate_records = []
    if cycle == 1 and "n_w_candidates_first_cycle" in config:
        n_cands = config["n_w_candidates_first_cycle"]
    else:
        n_cands = config.get("n_w_candidates", 3)

    def _write_state(current_idx: int, status: str, done: int = 0, total: int = 0):
        if outer_state is None:
            return
        outer_state.write_w_phase_state({
            "cycle": cycle, "n_candidates": n_cands,
            "current_candidate": current_idx, "status": status,
            "p_star_size": len(p_star), "papers_per_prompt": papers_per,
            "completed_candidates": [
                {"index": r["index"], "J": r.get("J"),
                 "fail_rate": r.get("fail_rate"),
                 "ideation_time": r.get("ideation_time"),
                 "scoring_time": r.get("scoring_time"),
                 "summary": (r.get("summary") or "")[:300]}
                for r in candidate_records
            ],
            "haiku_calls_done": done,
            "haiku_calls_total": total,
        })

    for i in range(1, n_cands + 1):
        new_label = f"W_{cycle}_cand_{i}"
        cand_path = phase_dir / f"candidate_{i}.json"
        cand_w_path = phase_dir / f"candidate_{i}_W.json"
        # Resume: if this candidate was already scored in a prior crashed run,
        # reload it instead of re-running.
        if cand_path.exists() and cand_w_path.exists():
            try:
                d = json.loads(cand_path.read_text())
                w_cand_resume = WPrompt.load(cand_w_path)
                if d.get("J") is not None:
                    cand_raw_w = {ph: [(pid, w) for pid, w in pp.items() if w is not None]
                                  for ph, pp in d.get("per_pair_scores", {}).items()}
                    prior.append({
                        "index": i,
                        "summary": d.get("summary", w_cand_resume.summary),
                        "J": d["J"],
                        "prompt_text": w_cand_resume.prompt_text,
                        "raw_w": cand_raw_w,
                    })
                    candidate_records.append({
                        "index": i, "label": new_label,
                        "summary": d.get("summary", w_cand_resume.summary),
                        "J": d["J"], "fail_rate": d.get("fail_rate"),
                        "n_total": d.get("n_total"), "n_failed": d.get("n_failed"),
                        "ideation_time": d.get("ideation_time"),
                        "scoring_time": d.get("scoring_time"),
                    })
                    logger.info(f"Cycle {cycle} W candidate {i}/{n_cands}: "
                                f"resumed from disk (J={d['J']:+.5f})")
                    continue
            except Exception as ex:
                logger.warning(f"Failed to resume candidate {i} from disk: {ex}; redoing")

        logger.info(f"Cycle {cycle} W candidate {i}/{n_cands}: proposing...")
        _write_state(i, "proposing")
        t_ideation = time.monotonic()
        w_cand, raw_text, err = await optimizer.propose(
            p_star=p_star, prev_w=prev_w, prev_w_label=prev_w_label,
            prior_candidates=prior, candidate_index=i, total_candidates=n_cands,
            new_w_label=new_label, weak_history=weak_history, raw_s=raw_s,
            baseline_J=baseline_J,
        )
        ideation_time = time.monotonic() - t_ideation
        if w_cand is None:
            logger.warning(f"Candidate {i} failed: {err}")
            prior.append({"index": i, "summary": f"(failed: {err})",
                          "J": float("inf"), "prompt_text": "", "raw_w": {}})
            candidate_records.append({"index": i, "error": err, "J": None,
                                       "raw_response": raw_text[:1000],
                                       "ideation_time": ideation_time,
                                       "scoring_time": None})
            continue
        # Score
        _write_state(i, "scoring", done=0, total=len(p_star) * papers_per)

        def _cb(done: int, total: int, _i=i):
            _write_state(_i, "scoring", done=done, total=total)

        t_scoring = time.monotonic()
        cand_eval = await evaluate_candidate(
            w_cand, weak_judge, semaphore, p_star, paper_subsample,
            progress_cb=_cb,
        )
        scoring_time = time.monotonic() - t_scoring
        logger.info(f"Candidate {i}: J = {cand_eval.J:+.5f}, "
                    f"fail_rate = {cand_eval.fail_rate:.2%} "
                    f"({cand_eval.n_failed}/{cand_eval.n_total})")
        # Build raw_w dict for this candidate from per_pair_scores
        cand_raw_w = {ph: [(pid, w) for pid, w in pp.items() if w is not None]
                      for ph, pp in cand_eval.per_pair_scores.items()}
        prior.append({
            "index": i,
            "summary": w_cand.summary,
            "J": cand_eval.J,
            "prompt_text": w_cand.prompt_text,
            "raw_w": cand_raw_w,
        })
        # Persist candidate + eval (including raw per-paper scores)
        w_cand.save(cand_w_path)
        cand_path.write_text(json.dumps({
            "index": i, "label": new_label, "summary": w_cand.summary,
            "J": cand_eval.J, "fail_rate": cand_eval.fail_rate,
            "n_total": cand_eval.n_total, "n_failed": cand_eval.n_failed,
            "per_prompt_w_mean": cand_eval.per_prompt_w_mean,
            "per_pair_scores": cand_eval.per_pair_scores,
            "ideation_time": ideation_time,
            "scoring_time": scoring_time,
        }, indent=2))
        candidate_records.append({
            "index": i, "label": new_label, "summary": w_cand.summary,
            "J": cand_eval.J, "fail_rate": cand_eval.fail_rate,
            "n_total": cand_eval.n_total, "n_failed": cand_eval.n_failed,
            "ideation_time": ideation_time,
            "scoring_time": scoring_time,
        })
        # Flush API events per candidate so we don't lose data on crash
        from blue_team import api_log
        await api_log.flush(label=f"cycle{cycle}_w_cand{i}")

    # Pick best (J is now a minimization target — lower is better)
    best_J = float("inf")
    best_idx = None
    for rec in candidate_records:
        if rec.get("J") is None:
            continue
        if rec["J"] < best_J:
            best_J = rec["J"]
            best_idx = rec["index"]

    new_w_label = f"W_{cycle}"
    final_path = phase_dir / "chosen.json"

    if best_idx is None or best_J >= baseline_J:
        # Rollback: keep prev_W as this cycle's W. Copy w_scores under prev_w_label
        # to the new label so downstream stages see the same scores under the new
        # W's name.
        if best_idx is None:
            logger.warning(f"Cycle {cycle}: ALL {n_cands} candidates failed; "
                           f"rolling back to {prev_w_label} (J={baseline_J:.5f})")
        else:
            logger.warning(f"Cycle {cycle}: best candidate J={best_J:.5f} did not beat "
                           f"{prev_w_label} J={baseline_J:.5f}; rolling back to {prev_w_label}")
        from blue_team.schema import ScoreSchema
        chosen_w = WPrompt(
            label=new_w_label,
            summary=prev_w.summary,
            prompt_text=prev_w.prompt_text,
            score_schema=ScoreSchema.from_dict(prev_w.score_schema.to_dict()),
        )
        chosen_w.save(final_path)
        # Propagate prev_w_label scores to new_w_label for every sample.
        for entry in population.prompts.values():
            for s in entry.samples:
                v = s.w_scores.get(prev_w_label)
                if v is not None:
                    s.w_scores[new_w_label] = v
        population.add_weak_prompt(new_w_label, str(final_path), chosen_w.summary)
        population.save()
        diag = {
            "cycle": cycle,
            "chosen_index": "rollback",
            "rollback_to": prev_w_label,
            "best_J": baseline_J,
            "best_candidate_J": best_J if best_idx is not None else None,
            "baseline_J": baseline_J,
            "candidates": candidate_records,
            "p_star_hashes": [p.prompt_hash for p in p_star],
            "papers_per_prompt": papers_per,
        }
        (phase_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
        return chosen_w, diag

    chosen_w = WPrompt.load(phase_dir / f"candidate_{best_idx}_W.json")
    chosen_w.label = new_w_label
    chosen_w.save(final_path)
    logger.info(f"Cycle {cycle}: W_{cycle} = candidate {best_idx} "
                f"(J = {best_J:.5f}, baseline {prev_w_label} = {baseline_J:.5f})")

    # Update population: backfill per-paper w_scores under W_k from the chosen candidate
    chosen_eval = json.loads((phase_dir / f"candidate_{best_idx}.json").read_text())
    raw_scores = chosen_eval.get("per_pair_scores", {})
    for ph, by_paper in raw_scores.items():
        entry = population.prompts.get(ph)
        if entry is None:
            continue
        for s in entry.samples:
            v = by_paper.get(s.paper_id)
            if v is not None:
                s.w_scores[new_w_label] = v
    population.add_weak_prompt(new_w_label, str(final_path), chosen_w.summary)
    population.save()

    # Full population re-judge under W_k (cheap — uses cached proposals via
    # Haiku). Done here rather than at the start of pf_phase so the after-W
    # diagnostic plot has w(W_k) for every prompt.
    from blue_team.pf_phase import _rejudge_under_w
    await _rejudge_under_w(population, chosen_w, new_w_label, config["max_connections"])

    # Build diagnostics dict for caller
    diag = {
        "cycle": cycle,
        "chosen_index": best_idx,
        "best_J": best_J,
        "baseline_J": baseline_J,
        "candidates": candidate_records,
        "p_star_hashes": [p.prompt_hash for p in p_star],
        "papers_per_prompt": papers_per,
    }
    (phase_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    return chosen_w, diag


async def evaluate_population_under_W(
    w: WPrompt,
    population: Population,
    config: dict,
    label: str,
) -> dict[str, dict[str, float | None]]:
    """Score every prompt-paper pair in the population under W; backfill into samples.

    Used to refresh population scores under a new W (e.g., after picking W_k).
    """
    from red_team import evaluator as v2_eval
    if not v2_eval._eval_config:
        v2_eval.init_models({
            "weak_judge": config["models"]["weak_judge"],
            "weak_judge_reasoning_tokens": config["models"].get("weak_judge_reasoning_tokens", 16384),
            "weak_scorer_type": "quality",
            "n_proposals": 10,
        }, config["max_connections"])
    weak_judge = v2_eval._get_weak_judge()
    semaphore = asyncio.Semaphore(config["max_connections"])

    p_list = list(population.prompts.values())
    paper_subsample = {p.prompt_hash: [s.paper_id for s in p.samples] for p in p_list}
    raw = await eval_w_on_population(w, weak_judge, semaphore, p_list, paper_subsample)
    for ph, by_paper in raw.items():
        entry = population.prompts.get(ph)
        if not entry:
            continue
        for s in entry.samples:
            v = by_paper.get(s.paper_id)
            if v is not None:
                s.w_scores[label] = v
    population.save()
    return raw
