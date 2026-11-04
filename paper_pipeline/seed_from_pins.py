"""Stage 0: seed metadata.json from the shipped paper-ID pin.

Lets you skip stages 1-3 (OpenReview scrape + arXiv matching) and reproduce the
paper's exact corpus instead of whatever OpenReview/arXiv return today. Every pinned
paper already carries its arxiv_id, so stage 4 can start immediately.

Usage:
  python seed_from_pins.py                       # seed all 2636 pinned papers
  python seed_from_pins.py --limit 5             # smoke-test subset
  python seed_from_pins.py --split val --limit 5 # a few held-out validation papers
  python seed_from_pins.py --decision poster     # one decision tier only
"""

import argparse
import csv
from pathlib import Path

from utils import load_metadata, save_metadata, ensure_dirs

PIN_CSV = Path(__file__).resolve().parent.parent / "paper_ids" / "dataset.csv"

# The paper optimizes prompts on posters and validates on the higher-tier papers.
SPLIT_DECISIONS = {"train": {"poster"}, "val": {"spotlight", "oral"}}


def load_pins(path: Path = PIN_CSV) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"pin file not found: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins", type=Path, default=PIN_CSV)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", choices=["train", "val"], default=None,
                    help="train = posters (prompt optimization), "
                         "val = spotlight/oral (held out, what the figures report on)")
    ap.add_argument("--decision", choices=["poster", "spotlight", "oral"], default=None,
                    help="a single decision tier; --split is usually what you want")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reset flags on papers already in metadata.json")
    args = ap.parse_args()

    if args.split and args.decision:
        ap.error("pass --split or --decision, not both")

    rows = load_pins(args.pins)
    if args.split:
        rows = [r for r in rows if r["decision"] in SPLIT_DECISIONS[args.split]]
    elif args.decision:
        rows = [r for r in rows if r["decision"] == args.decision]
    if args.limit:
        rows = rows[:args.limit]

    ensure_dirs()
    metadata = load_metadata()
    papers = metadata.setdefault("papers", {})

    added = reset = skipped = 0
    for r in rows:
        pid = r["paper_id"]
        if pid in papers and not args.overwrite:
            skipped += 1
            continue
        record = {
            "openreview_id": r["openreview_id"],
            "conference": r["conference"],
            "year": int(r["year"]),
            "decision": r["decision"],
            "title": r["title"],
            "arxiv_id": r["arxiv_id"],
            "arxiv_url": f"https://arxiv.org/abs/{r['arxiv_id']}" if r["arxiv_id"] else None,
            "first_public_date": r["first_public_date"],
            # Deliberately unset: the stages must see these as not-yet-done.
            "text_extracted": False,
            "figures_extracted": False,
            "figure_count": None,
            "paper_classified": False,
            "rq_extracted": False,
            "error": None,
        }
        if pid in papers:
            papers[pid].update(record)
            reset += 1
        else:
            papers[pid] = record
            added += 1

    save_metadata(metadata)
    print(f"Seeded {len(rows)} pinned papers: {added} added, {reset} reset, {skipped} left alone")
    print(f"metadata.json now holds {len(papers)} papers")
    print("\nNext: python run.py 4 5 6 7 8")


if __name__ == "__main__":
    main()
