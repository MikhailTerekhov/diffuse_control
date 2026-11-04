"""Apply a candidate W (text + score schema) to a cached proposal → normalized [0,1].

Reuses Haiku model handle from `red_team.evaluator` so concurrency limits
and 1m-context headers are inherited.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from blue_team.schema import WPrompt, parse_response

logger = logging.getLogger(__name__)


def _load_proposal_with_context(proposal_path: str) -> tuple[str, str]:
    """Read cached eval-detail JSON, return (proposals_text, rq_text_with_classification).

    The rq_text is reconstructed from papers/ since eval_details only saves proposals.
    """
    data = json.loads(Path(proposal_path).read_text())
    proposals = data["proposals"]
    paper_id = data["paper_id"]
    return proposals, paper_id


def _build_rq_user_prompt(paper_id: str, proposals: str) -> str:
    """Mirror evaluator.py's weak_user_prompt for blind judges."""
    from pathlib import Path as _P
    proj_root = _P(__file__).resolve().parent.parent
    rq_path = proj_root / "papers" / "research_questions" / f"{paper_id}.md"
    rq_text = rq_path.read_text().strip() if rq_path.exists() else ""
    # Append classification if present
    try:
        import sys
        if str(proj_root) not in sys.path:
            sys.path.insert(0, str(proj_root))
        from eval_harness.experiments_quality_eval import _load_classification
        cls = _load_classification(paper_id)
        if cls:
            rq_text += f"\n\n---\n\n{cls}"
    except Exception:
        pass
    return (
        f"Evaluate the following proposed experiments for the given research problem.\n\n"
        f"## RESEARCH CONTEXT:\n{rq_text}\n\n---\n\n"
        f"## MODEL-GENERATED PROPOSALS:\n{proposals}"
    )


async def score_one(
    weak_judge,                # inspect_ai model handle (Haiku)
    semaphore: asyncio.Semaphore,
    w: WPrompt,
    proposal_path: str,
    log_subrole: str = "weak_judge",
) -> tuple[float | None, dict]:
    """Run one (W, proposal) Haiku call. Returns (score in [0,1] or None, sub_scores)."""
    import time as _time
    from blue_team import api_log

    proposals, paper_id = _load_proposal_with_context(proposal_path)
    user_prompt = _build_rq_user_prompt(paper_id, proposals)
    msgs = [
        ChatMessageSystem(content=w.prompt_text),
        ChatMessageUser(content=user_prompt),
    ]

    async with semaphore:
        t0 = _time.time()
        try:
            resp = await weak_judge.generate(msgs)
        except Exception as e:
            logger.warning(f"Haiku call failed for {paper_id} under {w.label}: {e}")
            await api_log.log_call(
                phase="w_phase" if log_subrole == "weak_judge" else "pf_phase",
                subrole=log_subrole + "_error",
                model=str(getattr(weak_judge, "name", "")) or "haiku",
                input_msgs=msgs, response=None, duration_s=_time.time() - t0,
                extra={"paper_id": paper_id, "w_label": w.label, "error": str(e)},
            )
            return None, {}
        await api_log.log_call(
            phase="w_phase" if log_subrole == "weak_judge" else "pf_phase",
            subrole=log_subrole,
            model=str(getattr(weak_judge, "name", "")) or "haiku",
            input_msgs=msgs, response=resp, duration_s=_time.time() - t0,
            extra={"paper_id": paper_id, "w_label": w.label},
        )
    if not resp.choices:
        return None, {}
    text = resp.choices[0].message.text
    score, sub = parse_response(text, w.score_schema)
    return score, sub


async def score_many(
    weak_judge,
    semaphore: asyncio.Semaphore,
    w: WPrompt,
    proposal_paths: list[str],
) -> list[tuple[float | None, dict]]:
    """Score a list of cached proposals under W. Returns list of (score, sub) per path."""
    tasks = [asyncio.create_task(score_one(weak_judge, semaphore, w, p))
             for p in proposal_paths]
    return list(await asyncio.gather(*tasks))
