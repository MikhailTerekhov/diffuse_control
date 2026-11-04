"""Generate train/val/test split for Tinker RL training.

- Test: 700 post-cutoff papers (>= 2025-09-01), all spotlight/oral included
- Val: 300 papers from remaining pool
- Train: everything else
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from paper_pipeline.utils import (  # noqa: E402
    RQ_DIR, TEXT_DIR, CLASSIFY_DIR, METADATA_PATH,
)

CUTOFF_DATE = "2025-09-01"


def _load_metadata() -> dict:
    path = METADATA_PATH
    raw = json.loads(path.read_text())
    flat = {}
    for outer in raw.values():
        if isinstance(outer, dict):
            for pid, info in outer.items():
                flat[pid] = info
    return flat


def _has_files(pid: str) -> bool:
    return (
        (RQ_DIR / f"{pid}.md").exists()
        and (TEXT_DIR / f"{pid}.md").exists()
        and (CLASSIFY_DIR / f"{pid}.json").exists()
    )


def generate_split(
    test_n: int = 700,
    val_n: int = 300,
    seed: int = 42,
    output_path: Path | None = None,
) -> dict:
    metadata = _load_metadata()
    rng = random.Random(seed)

    # Partition into post-cutoff and pre-cutoff
    post_spotlight = []
    post_poster = []
    pre_cutoff = []

    for pid, entry in sorted(metadata.items()):
        if not _has_files(pid):
            continue
        fpd = entry.get("first_public_date", "")
        if not fpd:
            continue
        if fpd >= CUTOFF_DATE:
            if entry.get("decision", "poster") in ("spotlight", "oral"):
                post_spotlight.append(pid)
            else:
                post_poster.append(pid)
        else:
            pre_cutoff.append(pid)

    # Test: all spotlight/oral + random posters to fill test_n
    test = list(post_spotlight)
    posters_for_test = max(0, test_n - len(test))
    shuffled_posters = post_poster[:]
    rng.shuffle(shuffled_posters)
    test += shuffled_posters[:posters_for_test]
    remaining_post = shuffled_posters[posters_for_test:]

    # Pool for train+val: remaining post-cutoff + all pre-cutoff
    pool = remaining_post + pre_cutoff
    rng.shuffle(pool)

    val = sorted(pool[:val_n])
    train = sorted(pool[val_n:])
    test = sorted(test)

    result = {
        "seed": seed,
        "train_n": len(train),
        "val_n": len(val),
        "test_n": len(test),
        "total": len(train) + len(val) + len(test),
        "post_cutoff_n": len(post_spotlight) + len(post_poster),
        "pre_cutoff_n": len(pre_cutoff),
        "test_spotlight_oral_n": len(post_spotlight),
        "train": train,
        "val": val,
        "test": test,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(
            f"Split: train={len(train)} val={len(val)} test={len(test)} "
            f"(post-cutoff={len(post_spotlight)+len(post_poster)}, "
            f"pre-cutoff={len(pre_cutoff)})"
        )

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-n", type=int, default=700)
    parser.add_argument("--val-n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="rl_red_team/split.json")
    args = parser.parse_args()
    generate_split(args.test_n, args.val_n, args.seed, Path(args.output))
