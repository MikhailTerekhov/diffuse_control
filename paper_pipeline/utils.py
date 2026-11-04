import json
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FUZZY_DATA_DIR", REPO_ROOT / "data" / "papers"))

METADATA_PATH = DATA_DIR / "metadata.json"
TEXT_DIR = DATA_DIR / "text"
TEXT_WITH_FIGS_DIR = DATA_DIR / "text_with_figures"
FIGURES_DESC_DIR = DATA_DIR / "figures_description"
LATEX_SOURCES_DIR = DATA_DIR / "latex_sources"
FIGURE_MAPPINGS_DIR = DATA_DIR / "figure_mappings"
CLASSIFY_DIR = DATA_DIR / "paper_classifications"
RQ_DIR = DATA_DIR / "research_questions"
PAPER_LOGS_DIR = DATA_DIR / "paper_logs"

# The paper's dataset: NeurIPS 2025 + ICLR 2026 only (both post generator knowledge cutoff).
VENUES = [
    ("NeurIPS.cc/2025/Conference", "NeurIPS", 2025),
    ("ICLR.cc/2026/Conference", "ICLR", 2026),
]


def paper_id(conference: str, year: int, openreview_id: str) -> str:
    return f"{conference.lower()}_{year}_{openreview_id}"


_METADATA_LOCK = threading.Lock()


def load_metadata() -> dict:
    with _METADATA_LOCK:
        if METADATA_PATH.exists():
            return json.loads(METADATA_PATH.read_text())
        return {"papers": {}}


def save_metadata(data: dict):
    with _METADATA_LOCK:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=DATA_DIR, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, METADATA_PATH)
        except:
            os.unlink(tmp)
            raise


def check_disk_space(min_gb: float = 50.0) -> bool:
    usage = shutil.disk_usage(DATA_DIR)
    return usage.free / (1024**3) >= min_gb


def get_paper_logger(pid: str) -> logging.Logger:
    """Logger writing to paper_logs/{pid}.log."""
    PAPER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"paper.{pid}")
    if not logger.handlers:
        handler = logging.FileHandler(PAPER_LOGS_DIR / f"{pid}.log")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def close_paper_logger(pid: str):
    logger = logging.getLogger(f"paper.{pid}")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def ensure_dirs():
    for d in [TEXT_DIR, TEXT_WITH_FIGS_DIR, FIGURES_DESC_DIR, LATEX_SOURCES_DIR,
              FIGURE_MAPPINGS_DIR, CLASSIFY_DIR, RQ_DIR, PAPER_LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
