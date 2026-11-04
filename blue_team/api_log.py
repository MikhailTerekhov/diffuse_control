"""Inspect-AI transcript-based API logger for the outer saddle loop.

Mirrors red_team.run._init_api_logging / _flush_transcript exactly.
Captures EVERY model.generate() call across the pipeline (bootstrap kernels,
prompt expansion, summaries, W optimizer, W phase Haiku scoring, PF phase
Haiku rejudge, anchor-eval generator/judge calls) with full input messages,
output text + reasoning, raw HTTP request/response, token usage, stop reason,
and timings — by enabling Transcript(log_model_api=True) globally.

Output: <run_dir>/logs/api_events/events_<label>_<ts>.jsonl, one JSON per call.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_log_dir: Path | None = None
_transcript: Any = None
_seq = 0


def set_log_dir(run_dir: Path) -> None:
    """Init log dir + start the inspect-ai transcript with raw HTTP capture."""
    global _log_dir, _transcript
    _log_dir = Path(run_dir) / "logs" / "api_events"
    _log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from inspect_ai.log._transcript import Transcript, init_transcript
        _transcript = Transcript(log_model_api=True)
        init_transcript(_transcript)
        logger.info("Outer-loop API logging enabled (log_model_api=True)")
    except Exception as e:
        logger.warning(f"Could not init outer-loop transcript logging: {e}")
        _transcript = None


def reinit() -> None:
    """Re-install our outer transcript (use after v2 main_loop replaces it)."""
    if _log_dir is None:
        return
    global _transcript
    try:
        from inspect_ai.log._transcript import Transcript, init_transcript
        _transcript = Transcript(log_model_api=True)
        init_transcript(_transcript)
    except Exception as e:
        logger.warning(f"Could not re-init transcript: {e}")
        _transcript = None


async def flush(label: str = "") -> None:
    """Flush completed model events to events_<seq>_<label>_<ts>.jsonl."""
    global _seq
    if _transcript is None or _log_dir is None or not _transcript.events:
        return
    try:
        from inspect_ai.event._model import ModelEvent
    except Exception:
        return

    completed = [e for e in _transcript.events
                 if isinstance(e, ModelEvent) and e.pending is None]
    pending = [e for e in _transcript.events
               if not (isinstance(e, ModelEvent) and e.pending is None)]
    if not completed:
        return

    attachments = _transcript.attachments

    def resolve(text):
        if not isinstance(text, str) or "attachment://" not in text:
            return text
        for hash_key, content in attachments.items():
            text = text.replace(f"attachment://{hash_key}", content)
        return text

    def resolve_obj(obj):
        if isinstance(obj, str):
            return resolve(obj)
        if isinstance(obj, dict):
            return {k: resolve_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve_obj(v) for v in obj]
        return obj

    _seq += 1
    out = _log_dir / f"events_{_seq:05d}_{label or 'flush'}_{int(time.time())}.jsonl"
    with open(out, "w") as f:
        for ev in completed:
            reasoning_text = ""
            if ev.output and ev.output.choices:
                for block in ev.output.choices[0].message.content:
                    btype = getattr(block, "type", "")
                    if btype in ("thinking", "reasoning"):
                        raw = (getattr(block, "thinking", "")
                               or getattr(block, "reasoning", "") or "")
                        reasoning_text += resolve(raw)
            entry = {
                "model": ev.model,
                "role": ev.role,
                "input": [{"role": getattr(m, "role", "?"),
                           "content": resolve(m.content) if hasattr(m, "content") else str(m)}
                          for m in ev.input],
                "output_text": resolve(ev.output.choices[0].message.text) if (ev.output and ev.output.choices) else None,
                "reasoning": reasoning_text or None,
                "stop_reason": ev.output.stop_reason if ev.output else None,
                "usage": {
                    "input_tokens": ev.output.usage.input_tokens if ev.output and ev.output.usage else None,
                    "output_tokens": ev.output.usage.output_tokens if ev.output and ev.output.usage else None,
                    "reasoning_tokens": ev.output.usage.reasoning_tokens if ev.output and ev.output.usage else None,
                } if ev.output else None,
                "error": ev.error,
                "call": resolve_obj({
                    "request": ev.call.request if ev.call else None,
                    "response": ev.call.response if ev.call else None,
                    "time": ev.call.time if ev.call else None,
                }) if ev.call else None,
            }
            f.write(json.dumps(entry, default=str) + "\n")

    _transcript.events.clear()
    _transcript.events.extend(pending)
    logger.info(f"Flushed {len(completed)} outer-loop API events to {out}")


# --- back-compat shim: existing call sites can still call log_call() ---
# It's a no-op now because the transcript captures the same call automatically.

async def log_call(**kwargs) -> None:
    return None
