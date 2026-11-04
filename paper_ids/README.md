# Pinned paper set

The exact dataset used in the paper, pinned by ID, so that you can rebuild our dataset rather than whatever OpenReview and arXiv return today.

Seed a fresh `metadata.json` from this and skip stages 1-3 entirely:

```bash
cd ../paper_pipeline && uv run python seed_from_pins.py
```