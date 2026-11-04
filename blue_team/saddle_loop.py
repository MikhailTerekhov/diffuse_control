"""Top-level outer-loop orchestrator: bootstrap + N cycles of (W phase, PF phase)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from blue_team import api_log
from blue_team.diagnostics import (
    metrics_under_w,
    scatter_after_pf,
    scatter_population,
    scatter_w_phase_change,
)
from blue_team.outer_state import OuterState
from blue_team.population import Population
from blue_team.pf_phase import run_pf_phase
from blue_team.summaries import summarize_prompt
from blue_team.w_phase import run_w_phase

logger = logging.getLogger(__name__)


async def _summarize_new_prompts(population: Population, output_dir: Path, config: dict,
                                 semaphore: asyncio.Semaphore | None) -> None:
    """Generate Sonnet summaries for any prompts that lack one."""
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    todo = [(p.prompt_text, p.prompt_hash, p) for p in population.prompts.values()
            if not p.summary]
    if not todo:
        return
    model_name = config["models"].get("summary", "anthropic/claude-sonnet-4-6")
    rt = config["models"].get("summary_reasoning_tokens", 4096)

    async def _one(text, h, entry):
        s = await summarize_prompt(text, "generator", summaries_dir, h,
                                   model_name, rt, semaphore)
        entry.summary = s

    await asyncio.gather(*[_one(t, h, e) for t, h, e in todo])
    population.save()


async def run_saddle_loop(config: dict, output_dir: Path, n_cycles: int) -> None:
    population = Population.load(output_dir / "population.json")
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = diagnostics_dir / "metrics.jsonl"

    outer_state = OuterState(output_dir / "progress")

    sem = asyncio.Semaphore(config["max_connections"])

    p_def = population.prompts.get(population.p_def_hash)

    for cycle in range(1, n_cycles + 1):
        logger.info(f"=== Cycle {cycle} / {n_cycles} ===")
        w_label = f"W_{cycle}"
        chosen_w_path = output_dir / f"cycle_{cycle}" / "w_phase" / "chosen.json"
        # W phase (skip if already done)
        if chosen_w_path.exists() and w_label in population.weak_prompts:
            logger.info(f"Cycle {cycle} W phase already done — skipping; loading W_{cycle}")
            from blue_team.schema import WPrompt
            chosen_w = WPrompt.load(chosen_w_path)
            w_diag = json.loads((output_dir / f"cycle_{cycle}" / "w_phase" / "diagnostics.json").read_text())
        else:
            outer_state.set_phase(cycle, n_cycles, "w_phase")
            outer_state.append_event("w_phase_start", f"cycle {cycle}")
            chosen_w, w_diag = await run_w_phase(
                config, output_dir, population, cycle=cycle, outer_state=outer_state,
            )
            await api_log.flush(label=f"cycle{cycle}_w_phase")
            outer_state.clear_w_phase_state()
            outer_state.append_event("w_phase_done",
                f"cycle {cycle}: J={w_diag['best_J']:+.5f} idx={w_diag['chosen_index']}")

        # Diagnostics after W phase
        all_pop = list(population.prompts.values())
        w_label = f"W_{cycle}"
        m_W = metrics_under_w(all_pop, w_label, p_def)
        m_prev = metrics_under_w(all_pop, f"W_{cycle - 1}", p_def)
        with open(metrics_log, "a") as f:
            f.write(json.dumps({
                "cycle": cycle, "phase": "w_phase_post",
                f"W_{cycle - 1}": m_prev, f"W_{cycle}": m_W,
                "best_J": w_diag["best_J"], "chosen_idx": w_diag["chosen_index"],
            }) + "\n")
        scatter_w_phase_change(
            population, prev_w_label=f"W_{cycle - 1}", new_w_label=w_label,
            out_path=figures_dir / f"cycle_{cycle}_w_phase.png",
            title=f"Cycle {cycle} W phase — w(W_{cycle - 1}) → w({w_label})",
        )

        # PF phase (skip if already done)
        f_label = f"F_{cycle}"
        cycle_pf_dir = output_dir / f"cycle_{cycle}" / "pf_phase"
        pf_already = (cycle_pf_dir / "checkpoints" / "final" / "state.json").exists() and f_label in population.fronts
        if pf_already:
            logger.info(f"Cycle {cycle} PF phase already done — skipping")
            pf_summary = {"cycle": cycle, "skipped": True}
        else:
            outer_state.set_phase(cycle, n_cycles, "pf_phase", cycle_run_dir=cycle_pf_dir)
            outer_state.append_event("pf_phase_start", f"cycle {cycle}")
            # Flush any rejudge calls captured before v2 main_loop replaces transcript
            await api_log.flush(label=f"cycle{cycle}_pf_rejudge")
            pf_summary = await run_pf_phase(config, output_dir, population, cycle, chosen_w_path, chosen_w)
            # v2 main_loop replaced our transcript; reinstall ours for the next phase
            api_log.reinit()
            outer_state.append_event("pf_phase_done",
                f"cycle {cycle}: harvested {pf_summary.get('n_pf', 0)} PF, "
                f"{pf_summary.get('n_new_in_pf', 0)} new")
        await _summarize_new_prompts(population, output_dir, config, sem)
        await api_log.flush(label=f"cycle{cycle}_summaries")

        m_W_post_pf = metrics_under_w(list(population.prompts.values()), w_label, p_def)
        with open(metrics_log, "a") as f:
            f.write(json.dumps({
                "cycle": cycle, "phase": "pf_phase_post",
                f"W_{cycle}": m_W_post_pf, **pf_summary,
            }) + "\n")
        scatter_after_pf(
            population, w_label, new_front_label=f"F_{cycle}",
            out_path=figures_dir / f"cycle_{cycle}_after_pf_under_{w_label}.png",
            title=f"After cycle {cycle} PF phase — population under {w_label}",
            cycle_pf_dir=cycle_pf_dir,
        )
        # Update cycle history for the monitor
        outer_state.update_cycles_history(cycle, {
            "best_J": w_diag.get("best_J"),
            "chosen_index": w_diag.get("chosen_index"),
            "n_candidates": len(w_diag.get("candidates", [])),
            "candidates_J": [c.get("J") for c in w_diag.get("candidates", [])],
            "metrics": {f"W_{cycle - 1}": m_prev, w_label: m_W,
                        f"{w_label}_post_pf": m_W_post_pf},
            "pf_summary": pf_summary,
        })

    outer_state.set_done()
    outer_state.append_event("loop_done", f"{n_cycles} cycles complete")
    logger.info(f"Saddle loop complete. {n_cycles} cycles run.")
