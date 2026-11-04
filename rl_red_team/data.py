import json
import logging
from dataclasses import dataclass
from pathlib import Path

from eval_harness.experiments_quality_eval import (
    _load_classification,
    RQ_DIR,
    TEXT_DIR,
)

logger = logging.getLogger(__name__)

from paper_pipeline.utils import RQ_DIR, TEXT_DIR  # noqa: E402


@dataclass(frozen=True)
class PaperSample:
    paper_id: str
    rq_text: str        # research questions + classification
    paper_text: str      # full paper text (for strong judge)


def load_samples(split_file: str, split: str = "train") -> list[PaperSample]:
    with open(split_file) as f:
        split_data = json.load(f)

    paper_ids = split_data[split]
    samples = []
    for pid in paper_ids:
        rq_path = RQ_DIR / f"{pid}.md"
        text_path = TEXT_DIR / f"{pid}.md"
        if not rq_path.exists() or not text_path.exists():
            continue

        rq_text = rq_path.read_text().strip()
        paper_text = text_path.read_text().strip()
        if not rq_text or not paper_text:
            continue

        classification = _load_classification(pid)
        if classification:
            rq_text += f"\n\n---\n\n{classification}"

        samples.append(PaperSample(
            paper_id=pid,
            rq_text=rq_text,
            paper_text=paper_text,
        ))
    return samples
