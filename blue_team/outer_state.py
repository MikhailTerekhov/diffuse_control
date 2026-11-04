"""Outer-loop progress writer (atomic JSON to <run_dir>/progress/).

Mirrors red_team.monitor.MonitorState pattern but for the saddle loop.
Files written:
  state.json     — current cycle/phase + timestamps
  cycles.json    — list of completed cycles with metrics
  w_phase.json   — current W phase candidate state (only during W phase)
  events.json    — timestamped event log
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class OuterState:
    def __init__(self, progress_dir: Path):
        self.dir = progress_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._run_started = time.time()
        self._events: list[dict] = []
        # Persistent phase-history list (resumable runs append to it).
        ph = self.dir / "phase_history.json"
        try:
            self._phase_history: list[dict] = json.loads(ph.read_text()) if ph.exists() else []
            if not isinstance(self._phase_history, list):
                self._phase_history = []
        except Exception:
            self._phase_history = []
        # Write PID so the monitor can probe live TCP connections to Anthropic.
        self._atomic(self.dir / "pid.json", {"pid": os.getpid(), "started_at": time.time()})

    def _atomic(self, path: Path, data: Any) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(path)

    def _finalize_current_phase(self) -> None:
        """Close out the in-progress phase entry (if any) with end_ts + duration."""
        if not self._phase_history:
            return
        last = self._phase_history[-1]
        if last.get("ended_at") is None:
            now = time.time()
            last["ended_at"] = now
            last["duration"] = now - last.get("started_at", now)
            self._atomic(self.dir / "phase_history.json", self._phase_history)

    def set_phase(self, cycle: int, n_cycles_total: int, phase: str,
                  cycle_run_dir: Path | None = None) -> None:
        # Finalize the previous phase before starting a new one.
        self._finalize_current_phase()
        now = time.time()
        d = {
            "current_cycle": cycle,
            "n_cycles_total": n_cycles_total,
            "current_phase": phase,
            "phase_started_at": now,
            "run_started_at": self._run_started,
            "ts": now,
        }
        if cycle_run_dir is not None:
            d["cycle_run_dir"] = str(cycle_run_dir)
        self._atomic(self.dir / "state.json", d)
        self._phase_history.append({
            "cycle": cycle, "phase": phase,
            "started_at": now, "ended_at": None, "duration": None,
        })
        self._atomic(self.dir / "phase_history.json", self._phase_history)

    def set_done(self) -> None:
        self._finalize_current_phase()
        d = json.loads((self.dir / "state.json").read_text()) if (self.dir / "state.json").exists() else {}
        d["current_phase"] = "done"
        d["ts"] = time.time()
        self._atomic(self.dir / "state.json", d)

    def append_event(self, event_type: str, detail: str) -> None:
        self._events.append({"ts": time.time(), "type": event_type, "detail": detail})
        self._atomic(self.dir / "events.json", self._events)

    def write_w_phase_state(self, payload: dict) -> None:
        payload = {**payload, "ts": time.time()}
        self._atomic(self.dir / "w_phase.json", payload)

    def clear_w_phase_state(self) -> None:
        p = self.dir / "w_phase.json"
        if p.exists():
            p.unlink()

    def update_cycles_history(self, cycle: int, record: dict) -> None:
        path = self.dir / "cycles.json"
        try:
            history = json.loads(path.read_text()) if path.exists() else []
        except Exception:
            history = []
        # Replace if entry for this cycle already exists
        history = [h for h in history if h.get("cycle") != cycle]
        history.append({"cycle": cycle, **record, "ts": time.time()})
        history.sort(key=lambda h: h["cycle"])
        self._atomic(path, history)
