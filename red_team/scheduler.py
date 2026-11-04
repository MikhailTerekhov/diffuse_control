"""MO-ASHA v2 scheduler: multi-objective successive halving with per-sample work queue.

Manages trials through rungs, promotion via NSGA-II ranking, and
prioritized work dispatch at sample granularity.
"""

import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from red_team.pareto import moasha_rank, non_dominated_sort, epsilon_net_order


# --- Data classes ---

@dataclass
class SampleResult:
    paper_id: str
    weak_score: float | None = None
    strong_score: float | None = None


class TrialStatus(str, Enum):
    PENDING = "pending"         # created, not yet evaluated at any rung
    EVALUATING = "evaluating"   # samples being dispatched/in-flight for current rung
    WAITING = "waiting"         # rung eval done, waiting for promotion decision
    PROMOTED = "promoted"       # promoted to next rung, pending eval there
    STOPPED = "stopped"         # not promoted, sitting in pool (may be promoted later)
    FINISHED = "finished"       # completed highest rung
    RETIRED = "retired"         # failed permanently


@dataclass
class PromptTrial:
    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt_text: str = ""
    justification: str = ""
    parent_trial_id: str | None = None
    generation: int = 0
    island_id: int = 0

    current_rung: int = 0
    sample_results: list[SampleResult] = field(default_factory=list)
    rung_scores: dict[int, tuple[float, float, int]] = field(default_factory=dict)
    # rung -> (avg_weak, avg_strong, n_valid)

    status: TrialStatus = TrialStatus.PENDING
    created_at: float = field(default_factory=time.time)

    # Track which samples are dispatched but not yet returned
    _pending_samples: int = 0

    @property
    def objectives(self) -> tuple[float, float] | None:
        """(weak, strong) at highest completed rung."""
        if not self.rung_scores:
            return None
        max_rung = max(self.rung_scores.keys())
        weak, strong, _ = self.rung_scores[max_rung]
        return (weak, strong)

    @property
    def highest_rung(self) -> int:
        return max(self.rung_scores.keys()) if self.rung_scores else -1


@dataclass
class RungState:
    rung_index: int
    n_samples: int
    completed: list[str] = field(default_factory=list)
    promoted: set[str] = field(default_factory=set)


# --- Work items ---

@dataclass
class EvalWorkItem:
    """A single sample evaluation to dispatch."""
    trial_id: str
    rung: int
    paper_sample: dict  # {paper_id, input_text, rq_text, paper_text}
    priority: int = 0   # higher = more urgent (rung * 1000 - creation_order)


@dataclass
class GenerateWorkItem:
    """Signal to generate a new prompt."""
    pass


# --- Config ---

@dataclass
class MOASHAConfig:
    rung_samples: list[int] = field(default_factory=lambda: [8, 32, 128])
    eta: int = 4
    target_highest_rung: int = 80
    max_connections: int = 256
    n_islands: int = 3
    feature_bins: int = 4
    mode: str = "sandbagging"
    objectives: list[str] = field(default_factory=lambda: ["weak_score", "strong_score"])
    promotion_window: int = 10  # only consider last window*eta completions per rung for promotion


# --- Scheduler ---

class MOASHAScheduler:
    """Multi-Objective Asynchronous Successive Halving scheduler."""

    def __init__(self, config: MOASHAConfig):
        self.config = config
        self.trials: dict[str, PromptTrial] = {}
        self.rungs: list[RungState] = [
            RungState(rung_index=i, n_samples=n)
            for i, n in enumerate(config.rung_samples)
        ]
        self.n_generated: int = 0
        self._n_generating: int = 0  # number of concurrent generate calls in flight
        self._dataset: list[dict] | None = None

    @property
    def max_rung(self) -> int:
        return len(self.rungs) - 1

    @property
    def n_highest_rung_complete(self) -> int:
        return len(self.rungs[-1].completed)

    def is_done(self) -> bool:
        return self.n_highest_rung_complete >= self.config.target_highest_rung

    def set_dataset(self, dataset: list[dict]):
        self._dataset = dataset

    def register_trial(self, trial: PromptTrial):
        self.trials[trial.trial_id] = trial
        self.n_generated += 1

    def inject_completed_trial(self, trial: PromptTrial):
        """Insert a pre-evaluated trial as already complete at its highest rung.

        The trial must have rung_scores set at least for trial.current_rung
        (which becomes its highest_rung). It is added to that rung's
        completed list so it appears in the PF, but NOT to its promoted set
        (it doesn't get promoted further). It is marked FINISHED if at the
        max rung, otherwise STOPPED so the scheduler doesn't try to re-eval it.
        """
        if not trial.rung_scores:
            raise ValueError(f"Seeded trial {trial.trial_id} has no rung_scores")
        target_rung = trial.current_rung
        if target_rung not in trial.rung_scores:
            raise ValueError(
                f"Seeded trial {trial.trial_id} missing scores at rung {target_rung}"
            )

        self.trials[trial.trial_id] = trial
        # Don't bump n_generated — seeded trials are pre-existing, not newly generated
        rung_state = self.rungs[target_rung]
        if trial.trial_id not in rung_state.completed:
            rung_state.completed.append(trial.trial_id)

        if target_rung >= self.max_rung:
            trial.status = TrialStatus.FINISHED
        else:
            trial.status = TrialStatus.STOPPED
        trial._pending_samples = 0

    def _objectives_array(self, trial_ids: list[str], rung: int) -> np.ndarray:
        """Build objectives array for NSGA-II ranking.

        For sandbagging: maximize (-strong, weak)
        For strong_weak: maximize (strong, weak)
        """
        rows = []
        for tid in trial_ids:
            trial = self.trials[tid]
            if rung not in trial.rung_scores:
                continue
            weak, strong, _ = trial.rung_scores[rung]
            if self.config.mode in ("sandbagging", "sandbagging_detector"):
                rows.append((-strong, weak))
            else:
                rows.append((strong, weak))
        return np.array(rows) if rows else np.empty((0, 2))

    # --- Rung completion and promotion ---

    def on_sample_complete(self, trial_id: str, result: SampleResult):
        """Called when a single sample finishes for a trial."""
        trial = self.trials[trial_id]
        trial.sample_results.append(result)
        trial._pending_samples = max(0, trial._pending_samples - 1)

        # Check if rung is complete (all samples returned)
        target = self.config.rung_samples[trial.current_rung]
        n_at_rung = len(trial.sample_results)
        if n_at_rung >= target and trial._pending_samples == 0:
            self._complete_rung(trial)

    def _complete_rung(self, trial: PromptTrial):
        """Finalize a trial's rung evaluation and check promotions."""
        rung = trial.current_rung

        # Compute aggregate scores
        valid = [r for r in trial.sample_results
                 if r.weak_score is not None and r.strong_score is not None]
        if valid:
            avg_weak = sum(r.weak_score for r in valid) / len(valid)
            avg_strong = sum(r.strong_score for r in valid) / len(valid)
            trial.rung_scores[rung] = (avg_weak, avg_strong, len(valid))
        else:
            trial.rung_scores[rung] = (0.0, 0.0, 0)

        # Add to rung's completed list
        rung_state = self.rungs[rung]
        if trial.trial_id not in rung_state.completed:
            rung_state.completed.append(trial.trial_id)

        # At max rung, trial is finished
        if rung >= self.max_rung:
            trial.status = TrialStatus.FINISHED
            return

        # Mark as waiting (promotion checked below)
        trial.status = TrialStatus.WAITING

        # Check promotions for this rung
        self._try_promote(rung)

    def _try_promote(self, rung: int):
        """Promote trials from rung using NSGA-II ranking with adaptive rate control.

        Target promotion rate is 1/eta. We track the actual cumulative rate and
        adjust: if we've over-promoted, promote fewer (or none) until the rate
        drops back to target.
        """
        if rung >= self.max_rung:
            return

        rung_state = self.rungs[rung]
        completed_ids = rung_state.completed
        n_completed = len(completed_ids)

        if n_completed < self.config.eta:
            return

        # Adaptive rate: how many TOTAL promotions should we have by now?
        target_rate = 1.0 / self.config.eta
        target_promotions = int(n_completed * target_rate)
        current_promotions = len(rung_state.promoted)

        # How many new promotions can we make?
        k = target_promotions - current_promotions
        if k <= 0:
            # Already at or above target — don't promote, just mark waiting as stopped
            for tid in completed_ids:
                trial = self.trials[tid]
                if tid not in rung_state.promoted and trial.status == TrialStatus.WAITING:
                    trial.status = TrialStatus.STOPPED
            return

        # Use sliding window for ranking (which trials compete for the k slots)
        depth = self.max_rung - rung
        window_size = self.config.promotion_window * self.config.eta * (self.config.eta ** depth)
        windowed_ids = completed_ids[-window_size:]

        # Rank ALL valid trials in window (including already-promoted)
        valid_ids = [tid for tid in windowed_ids if rung in self.trials[tid].rung_scores]
        if not valid_ids:
            return

        objectives = self._objectives_array(valid_ids, rung)
        if len(objectives) == 0:
            return

        # Get full ranking, then walk down it picking the first k unpromoted
        ranked_indices = moasha_rank(objectives, len(valid_ids))
        newly_promoted = []
        for idx in ranked_indices:
            if len(newly_promoted) >= k:
                break
            tid = valid_ids[idx]
            if tid not in rung_state.promoted:
                newly_promoted.append(tid)

        for tid in newly_promoted:
            rung_state.promoted.add(tid)
            trial = self.trials[tid]
            trial.status = TrialStatus.PROMOTED
            trial.current_rung = rung + 1

        # Mark non-promoted trials as stopped
        for tid in valid_ids:
            trial = self.trials[tid]
            if tid not in rung_state.promoted and trial.status == TrialStatus.WAITING:
                trial.status = TrialStatus.STOPPED

    # --- Work queue ---

    def get_work_queue(self, max_items: int) -> list[EvalWorkItem | GenerateWorkItem]:
        """Return prioritized work items to dispatch.

        Priority: higher rung > lower rung > generate new prompt.
        Within a rung, earlier trials first.
        """
        if self._dataset is None:
            return []

        items: list[EvalWorkItem | GenerateWorkItem] = []

        # Collect trials needing samples, sorted by rung descending
        trials_needing_work = []
        for trial in self.trials.values():
            if trial.status in (TrialStatus.PENDING, TrialStatus.PROMOTED, TrialStatus.EVALUATING):
                target = self.config.rung_samples[trial.current_rung]
                dispatched = len(trial.sample_results) + trial._pending_samples
                remaining = target - dispatched
                if remaining > 0:
                    trials_needing_work.append((trial, remaining))

        # Sort: highest rung first, then earliest creation
        trials_needing_work.sort(key=lambda x: (-x[0].current_rung, x[0].created_at))

        for trial, remaining in trials_needing_work:
            if len(items) >= max_items:
                break

            # Pick random samples for this trial
            existing_ids = {r.paper_id for r in trial.sample_results}
            available = [s for s in self._dataset if s["paper_id"] not in existing_ids]
            if not available:
                available = self._dataset

            n_to_dispatch = min(remaining, max_items - len(items))
            samples = random.sample(available, min(n_to_dispatch, len(available)))

            for sample in samples:
                priority = trial.current_rung * 1000 - int(trial.created_at)
                items.append(EvalWorkItem(
                    trial_id=trial.trial_id,
                    rung=trial.current_rung,
                    paper_sample=sample,
                    priority=priority,
                ))
                trial._pending_samples += 1

            if trial.status == TrialStatus.PENDING:
                trial.status = TrialStatus.EVALUATING
            elif trial.status == TrialStatus.PROMOTED:
                trial.status = TrialStatus.EVALUATING

        # Fill spare capacity with generate requests
        # Each generate will produce a trial needing rung_samples[0] eval workers,
        # so estimate how many generates we need to fill the gap
        if not self.is_done():
            spare = max_items - len(items)
            rung0_samples = self.config.rung_samples[0]
            # How many generates to saturate? Each produces rung0_samples work items
            n_generates = max(0, spare // max(1, rung0_samples))
            # Cap by how many are not already generating
            n_generates = min(n_generates, max(1, n_generates) - self._n_generating)
            for _ in range(max(0, n_generates)):
                items.append(GenerateWorkItem())

        return items

    # --- Pareto front queries ---

    def get_pareto_front_at_rung(self, rung: int) -> list[PromptTrial]:
        """Pareto front among trials completed at exactly this rung, using rung-specific scores."""
        rung_state = self.rungs[rung]
        candidates = [self.trials[tid] for tid in rung_state.completed
                      if rung in self.trials[tid].rung_scores]
        if not candidates:
            return []

        objectives = self._objectives_array([t.trial_id for t in candidates], rung)
        if len(objectives) == 0:
            return []

        fronts = non_dominated_sort(objectives)
        if not fronts:
            return []
        ordered = epsilon_net_order(objectives, fronts[0])
        return [candidates[i] for i in ordered]

    def rung_objectives(self, trial: 'PromptTrial', rung: int) -> tuple[float, float] | None:
        """Get (weak, strong) for a trial at a specific rung."""
        if rung not in trial.rung_scores:
            return None
        weak, strong, _ = trial.rung_scores[rung]
        return (weak, strong)

    def get_pareto_front(self) -> list[PromptTrial]:
        """PF at the highest rung that has completions."""
        for rung in range(self.max_rung, -1, -1):
            pf = self.get_pareto_front_at_rung(rung)
            if pf:
                return pf
        return []

    def get_high_rung_pareto(self, n: int = 3) -> list[PromptTrial]:
        """Get PF prompts from highest rung downward. For optimizer inspiration.

        Starts at the highest rung with completions, falls back to lower
        rungs if not enough points.
        """
        for rung in range(self.max_rung, -1, -1):
            pf = self.get_pareto_front_at_rung(rung)
            if len(pf) >= n:
                return pf
        # Not enough at any single rung — return whatever the highest has
        return self.get_pareto_front()

    # --- Summary ---

    def summary(self) -> dict:
        rung_summaries = []
        for r in self.rungs:
            pf = self.get_pareto_front_at_rung(r.rung_index)
            pf_points = []
            for t in pf:
                obj = self.rung_objectives(t, r.rung_index)
                if obj:
                    pf_points.append({"weak": obj[0], "strong": obj[1], "trial_id": t.trial_id})
            rung_summaries.append({
                "rung": r.rung_index, "n_samples": r.n_samples,
                "completed": len(r.completed), "promoted": len(r.promoted),
                "pareto_size": len(pf),
                "pareto_points": pf_points,
            })
        # Active trials with progress
        active_trials = []
        for trial in self.trials.values():
            if trial.status in (TrialStatus.EVALUATING, TrialStatus.PENDING, TrialStatus.PROMOTED):
                target = self.config.rung_samples[trial.current_rung]
                done = len(trial.sample_results)
                active_trials.append({
                    "trial_id": trial.trial_id,
                    "rung": trial.current_rung,
                    "done": done,
                    "target": target,
                    "pending": trial._pending_samples,
                })

        return {
            "n_trials": len(self.trials),
            "n_generated": self.n_generated,
            "n_highest_rung": self.n_highest_rung_complete,
            "target_highest_rung": self.config.target_highest_rung,
            "generating": self._n_generating > 0,
            "n_generating": self._n_generating,
            "rungs": rung_summaries,
            "active_trials": active_trials,
        }
