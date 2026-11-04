"""Checkpoint save/load for MO-ASHA v2 state."""

import json
import logging
from pathlib import Path

from red_team.scheduler import (
    MOASHAConfig,
    MOASHAScheduler,
    PromptTrial,
    RungState,
    SampleResult,
    TrialStatus,
)

logger = logging.getLogger(__name__)


def save_state(checkpoint_dir: Path, scheduler: MOASHAScheduler) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trials_data = {}
    for tid, trial in scheduler.trials.items():
        trials_data[tid] = {
            "trial_id": trial.trial_id,
            "prompt_text": trial.prompt_text,
            "justification": trial.justification,
            "parent_trial_id": trial.parent_trial_id,
            "generation": trial.generation,
            "island_id": trial.island_id,
            "current_rung": trial.current_rung,
            "sample_results": [
                {"paper_id": r.paper_id, "weak_score": r.weak_score, "strong_score": r.strong_score}
                for r in trial.sample_results
            ],
            "rung_scores": {str(k): list(v) for k, v in trial.rung_scores.items()},
            "status": trial.status.value,
            "created_at": trial.created_at,
        }

    rungs_data = [
        {
            "rung_index": r.rung_index,
            "n_samples": r.n_samples,
            "completed": r.completed,
            "promoted": list(r.promoted),
        }
        for r in scheduler.rungs
    ]

    state = {
        "n_generated": scheduler.n_generated,
        "trials": trials_data,
        "rungs": rungs_data,
    }

    out_path = checkpoint_dir / "state.json"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(out_path)

    # Save Pareto front prompts as text files
    pareto = scheduler.get_pareto_front()
    prompts_dir = checkpoint_dir / "pareto_prompts"
    prompts_dir.mkdir(exist_ok=True)
    for i, t in enumerate(pareto):
        obj = t.objectives
        if obj:
            fname = f"prompt_{i+1}_w{obj[0]:.3f}_s{obj[1]:.3f}.txt"
        else:
            fname = f"prompt_{i+1}.txt"
        (prompts_dir / fname).write_text(t.prompt_text)

    logger.info(f"Saved checkpoint: {len(scheduler.trials)} trials, {len(pareto)} Pareto")


def load_state(checkpoint_dir: Path, config: MOASHAConfig) -> MOASHAScheduler:
    state_path = checkpoint_dir / "state.json"
    state = json.loads(state_path.read_text())

    scheduler = MOASHAScheduler(config)
    scheduler.n_generated = state["n_generated"]

    for tid, tdata in state["trials"].items():
        trial = PromptTrial(
            trial_id=tdata["trial_id"],
            prompt_text=tdata["prompt_text"],
            justification=tdata.get("justification", ""),
            parent_trial_id=tdata.get("parent_trial_id"),
            generation=tdata.get("generation", 0),
            island_id=tdata.get("island_id", 0),
            current_rung=tdata["current_rung"],
            sample_results=[SampleResult(**sr) for sr in tdata["sample_results"]],
            rung_scores={int(k): tuple(v) for k, v in tdata["rung_scores"].items()},
            status=TrialStatus(tdata["status"]),
            created_at=tdata.get("created_at", 0),
        )
        scheduler.trials[tid] = trial

    for i, rdata in enumerate(state["rungs"]):
        if i < len(scheduler.rungs):
            scheduler.rungs[i].completed = rdata["completed"]
            scheduler.rungs[i].promoted = set(rdata["promoted"])

    # Fix interrupted trials
    fixed = 0
    for trial in scheduler.trials.values():
        if trial.status == TrialStatus.EVALUATING:
            if trial.current_rung > 0 and trial.current_rung - 1 in trial.rung_scores:
                trial.status = TrialStatus.PROMOTED
            else:
                trial.status = TrialStatus.PENDING
            trial._pending_samples = 0
            fixed += 1

    if fixed:
        logger.info(f"Reset {fixed} interrupted trials for re-evaluation")

    logger.info(f"Loaded checkpoint: {len(scheduler.trials)} trials from {checkpoint_dir}")
    return scheduler
