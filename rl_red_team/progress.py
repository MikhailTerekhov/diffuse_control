"""Progress tracking for the TUI monitor.

Writes state to a JSON file that the monitor reads.
"""

import json
import time
from pathlib import Path


class ProgressTracker:
    def __init__(self, progress_dir: Path | None = None):
        self._dir = progress_dir
        self._run_start = time.time()
        self._step = -1
        self._phase = "init"  # init, eval, sampling, judging, training
        self._max_steps = 0
        self._eval_every = 0

        # Per-phase timing (separate for eval and train)
        self._step_start = 0.0
        self._train_times: list[float] = []
        self._eval_times: list[float] = []
        self._current_is_eval = False

        # Sampling state
        self._sampling_n = 0
        self._sampling_active = 0
        self._sampling_prompt_len = 0
        self._sampling_start = 0.0

        # Judging state
        self._judging_active = 0
        self._judging_peak = 0
        self._judging_total = 0
        self._judging_done = 0

    def set_run_info(self, max_steps: int, groups_per_batch: int, group_size: int,
                     eval_every: int = 0):
        self._max_steps = max_steps
        self._groups_per_batch = groups_per_batch
        self._group_size = group_size
        self._eval_every = eval_every
        self._write()

    def start_step(self, step: int, phase: str = "sampling"):
        self._step = step
        self._phase = phase
        self._step_start = time.time()
        self._current_is_eval = False
        self._reset_batch_state()
        self._write()

    def start_eval(self, step: int):
        self._step = step
        self._phase = "eval"
        self._step_start = time.time()
        self._current_is_eval = True
        self._reset_batch_state()
        self._write()

    def _reset_batch_state(self):
        self._sampling_active = 0
        self._sampling_n = 0
        self._judging_active = 0
        self._judging_peak = 0
        self._judging_done = 0
        self._judging_total = 0

    def set_phase(self, phase: str):
        self._phase = phase
        self._write()

    def finish_step(self):
        elapsed = time.time() - self._step_start
        if self._current_is_eval:
            self._eval_times.append(elapsed)
        else:
            self._train_times.append(elapsed)
        self._phase = "idle"
        self._write()

    # --- Batched strategy interface ---

    def start_sampling(self, n: int, prompt_len: int):
        self._sampling_n = n
        self._sampling_active = n
        self._sampling_prompt_len = prompt_len
        self._sampling_start = time.time()
        self._phase = "sampling"
        self._write()

    def finish_sampling(self, elapsed: float):
        self._sampling_active = 0
        self._phase = "judging"
        self._write()

    # --- Per-env interface ---

    def sampling_started(self):
        self._sampling_active += 1
        self._sampling_n += 1
        if self._sampling_active == 1:
            self._sampling_start = time.time()
        if not self._current_is_eval and self._phase not in ("eval",):
            self._phase = "sampling"
        self._write()

    def sampling_done(self):
        self._sampling_active -= 1
        if self._sampling_active == 0 and not self._current_is_eval:
            self._phase = "judging"
        if self._sampling_active % 5 == 0 or self._sampling_active == 0:
            self._write()

    # --- Judging ---

    def start_judging(self, idx: int = 0):
        self._judging_active += 1
        self._judging_total += 1
        self._judging_peak = max(self._judging_peak, self._judging_active)

    def finish_judging(self, idx: int, elapsed: float):
        self._judging_active -= 1
        self._judging_done += 1
        if self._judging_done % 5 == 0 or self._judging_active == 0:
            self._write()

    def _eta(self) -> dict:
        now = time.time()
        phase_elapsed = now - self._step_start if self._step_start else 0

        # Estimate remaining time in current phase
        if self._current_is_eval and self._eval_times:
            avg_eval = sum(self._eval_times) / len(self._eval_times)
            current_remaining = max(0, avg_eval - phase_elapsed)
        elif not self._current_is_eval and self._train_times:
            avg_train = sum(self._train_times) / len(self._train_times)
            current_remaining = max(0, avg_train - phase_elapsed)
        else:
            current_remaining = 0

        # How many train and eval steps remain
        remaining_steps = max(0, self._max_steps - self._step - 1) if self._max_steps > 0 else 0
        if self._eval_every > 0 and remaining_steps > 0:
            remaining_evals = remaining_steps // self._eval_every
        else:
            remaining_evals = 0
        remaining_trains = remaining_steps

        avg_train = sum(self._train_times) / len(self._train_times) if self._train_times else 0
        avg_eval = sum(self._eval_times) / len(self._eval_times) if self._eval_times else 0

        total_remaining = (
            current_remaining
            + remaining_trains * avg_train
            + remaining_evals * avg_eval
        )

        return {
            "current_phase_remaining": current_remaining,
            "remaining_trains": remaining_trains,
            "remaining_evals": remaining_evals,
            "avg_train_time": avg_train,
            "avg_eval_time": avg_eval,
            "total_seconds": total_remaining if (self._train_times or self._eval_times) else None,
        }

    def _write(self):
        if self._dir is None:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        eta = self._eta()
        state = {
            "ts": time.time(),
            "run_start": self._run_start,
            "uptime": time.time() - self._run_start,
            "step": self._step,
            "max_steps": self._max_steps,
            "phase": self._phase,
            "is_eval": self._current_is_eval,
            "step_elapsed": time.time() - self._step_start if self._step_start else 0,
            "sampling": {
                "n": self._sampling_n,
                "active": self._sampling_active,
                "prompt_len": self._sampling_prompt_len,
                "elapsed": time.time() - self._sampling_start if self._sampling_start else 0,
            },
            "judging": {
                "active": self._judging_active,
                "peak": self._judging_peak,
                "done": self._judging_done,
                "total": self._judging_total,
            },
            "timing": {
                "completed_train_steps": len(self._train_times),
                "completed_eval_steps": len(self._eval_times),
                "avg_train_time": eta["avg_train_time"],
                "avg_eval_time": eta["avg_eval_time"],
            },
            "eta": eta,
        }
        target = self._dir / "progress.json"
        try:
            target.write_text(json.dumps(state, indent=2))
        except Exception:
            pass


# Module-level singleton
_progress: ProgressTracker | None = None


def init_progress(progress_dir: Path) -> ProgressTracker:
    global _progress
    _progress = ProgressTracker(progress_dir)
    return _progress


def get_progress() -> ProgressTracker:
    global _progress
    if _progress is None:
        _progress = ProgressTracker()
    return _progress
