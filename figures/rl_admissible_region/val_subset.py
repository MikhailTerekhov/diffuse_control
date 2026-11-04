"""A fixed random subset of the validation papers, shared by every eval in this figure.

The point of the figure is a like-for-like plane: RL checkpoints and prompt-optimization
prompts scored on the *same* papers, so differences are not sampling noise. Both eval
scripts call val_ids(), and the choice is cached so repeated runs reuse it.

    uv run python figures/rl_admissible_region/val_subset.py        # show the subset
    N=32 uv run python figures/rl_admissible_region/val_subset.py   # smaller, for smoke tests
"""

import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from paper_pipeline.utils import CLASSIFY_DIR, METADATA_PATH, RQ_DIR, TEXT_DIR  # noqa: E402

CACHE = Path(__file__).resolve().parent / "data" / "val_ids.json"
DEFAULT_N = int(os.environ.get("N", 128))
SEED = 42


def _processed(pid: str) -> bool:
    return ((RQ_DIR / f"{pid}.md").exists() and (TEXT_DIR / f"{pid}.md").exists()
            and (CLASSIFY_DIR / f"{pid}.json").exists())


def val_ids(n: int = DEFAULT_N, seed: int = SEED, refresh: bool = False) -> list[str]:
    """The subset, cached. Held-out (spotlight/oral) papers built in the corpus."""
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())
    meta = json.loads(METADATA_PATH.read_text())["papers"] if METADATA_PATH.exists() else {}
    candidates = sorted(pid for pid, info in meta.items()
                        if info.get("decision") in ("spotlight", "oral") and _processed(pid))
    if not candidates:
        raise SystemExit(
            "no held-out papers built. Add some:\n"
            "  cd paper_pipeline && uv run python seed_from_pins.py --split val --limit 5 "
            "&& uv run python run.py 4 5 6 7 8"
        )
    picked = sorted(random.Random(seed).sample(candidates, min(n, len(candidates))))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(picked, indent=2))
    return picked


if __name__ == "__main__":
    ids = val_ids(refresh="--refresh" in sys.argv)
    print(f"{len(ids)} papers -> {CACHE}")
    for pid in ids[:5]:
        print(f"  {pid}")
    if len(ids) > 5:
        print(f"  ... and {len(ids) - 5} more")
