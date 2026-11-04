"""Generate train/validation split from post-cutoff papers.

Uses first_public_date >= 2025-09-01 from metadata.json to identify post-cutoff
papers. All spotlight/oral papers go to validation; posters fill the rest to
reach 70/30 train/val.
"""

import json
import random
from pathlib import Path

# One configurable corpus root, shared with the pipeline ($FUZZY_DATA_DIR).
from paper_pipeline.utils import RQ_DIR, TEXT_DIR, CLASSIFY_DIR, METADATA_PATH

CUTOFF_DATE = "2025-09-01"


def _load_metadata() -> dict:
    """Load papers metadata. Returns {paper_id: {...}} dict."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {METADATA_PATH}. Build the corpus first: "
            "see paper_pipeline/PIPELINE.md")
    raw = json.loads(METADATA_PATH.read_text())
    if "papers" in raw and isinstance(raw["papers"], dict):
        return raw["papers"]
    return raw


def _has_required_files(paper_id: str) -> bool:
    return (
        (RQ_DIR / f"{paper_id}.md").exists()
        and (TEXT_DIR / f"{paper_id}.md").exists()
        and (CLASSIFY_DIR / f"{paper_id}.json").exists()
    )


def get_all_eligible_papers(metadata: dict) -> tuple[list[str], list[str]]:
    """Return (spotlight_oral, posters) lists of eligible post-cutoff papers."""
    spotlight_oral = []
    posters = []

    for paper_id, entry in sorted(metadata.items()):
        first_public = entry.get("first_public_date", "")
        if not first_public or first_public < CUTOFF_DATE:
            continue
        if not _has_required_files(paper_id):
            continue
        decision = entry.get("decision", "poster")
        if decision in ("spotlight", "oral"):
            spotlight_oral.append(paper_id)
        else:
            posters.append(paper_id)

    return spotlight_oral, posters


def generate_split(
    train_ratio: float = 0.7,
    seed: int = 42,
    output_path: Path | None = None,
) -> dict:
    metadata = _load_metadata()
    spotlight_oral, posters = get_all_eligible_papers(metadata)
    total = len(spotlight_oral) + len(posters)

    # All spotlight/oral → validation
    val = list(spotlight_oral)

    # Fill remaining validation from posters to reach target 30%
    target_val = int(total * (1 - train_ratio))
    val_from_posters = max(0, target_val - len(val))

    rng = random.Random(seed)
    shuffled_posters = posters[:]
    rng.shuffle(shuffled_posters)

    val += sorted(shuffled_posters[:val_from_posters])
    train = sorted(shuffled_posters[val_from_posters:])
    val = sorted(val)

    result = {
        "seed": seed,
        "train_n": len(train),
        "val_n": len(val),
        "total_eligible": total,
        "spotlight_oral_n": len(spotlight_oral),
        "poster_n": len(posters),
        "train": train,
        "test": val,  # "test" key for backward compat with evaluator
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(
            f"Split: train={len(train)} val={len(val)} "
            f"(spotlight/oral={len(spotlight_oral)}, posters={len(posters)}, "
            f"total={total})"
        )

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="split.json")
    args = parser.parse_args()
    generate_split(args.train_ratio, args.seed, Path(args.output))
