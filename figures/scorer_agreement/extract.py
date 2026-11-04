"""Collect the scorer-agreement scores out of the inspect logs into one JSON.

Two judges x three runs = six scores per (generator, paper):

  g  ground truth proxy, Sonnet 4.6 with the paper, rubric   (runA, runB, runC)
  w  weak scorer, Haiku 4.5 blind, rubric                    (runA, runB, runC_haiku)

Run A is the base pass, B rescores A's saved proposals with fresh judge calls (judge
noise), C regenerates from scratch (generator noise). See figures/scorer_agreement/README.md.

Only rows where all six scores are present are kept, so the decomposition operates on a
balanced panel.

Usage:
  uv run python figures/scorer_agreement/extract.py
  uv run python figures/scorer_agreement/extract.py --logroot logs/scorer_agreement --out figures/scorer_agreement/data.json
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log

# column -> (log subdir, how the generator is identified, scorer name)
#   "gen"     the log's own eval.model is the generator
#   "rescore" the generator has to be recovered from sample metadata / the source log
SOURCES = {
    "g_A": ("runA", "gen", "quality_judge_paper"),
    "g_B": ("runB", "rescore", "quality_judge_paper"),
    "g_C": ("runC", "gen", "quality_judge_paper"),
    "w_A": ("runA", "gen", "quality_judge_blind"),
    "w_B": ("runB", "rescore", "quality_judge_blind"),
    "w_C": ("runC_haiku", "rescore", "quality_judge_blind"),
}

_BACKTRACK_CACHE = {}


def _generator_from_log(log_path):
    """Follow the gen_logs chain until a log names a real generator model."""
    if log_path in _BACKTRACK_CACHE:
        return _BACKTRACK_CACHE[log_path]
    seen, cur = set(), log_path
    while cur and cur not in seen:
        seen.add(cur)
        try:
            head = read_eval_log(cur, header_only=True)
        except Exception:
            break
        model = head.eval.model or ""
        if model and not model.startswith("none") and model != "unknown":
            _BACKTRACK_CACHE[log_path] = model.split("/")[-1]
            return _BACKTRACK_CACHE[log_path]
        nxt = (head.eval.task_args or {}).get("gen_logs")
        if isinstance(nxt, list) and nxt:
            nxt = nxt[0]
        if isinstance(nxt, str):
            nxt = nxt.split(",")[0].strip()
        cur = nxt
    _BACKTRACK_CACHE[log_path] = "?"
    return "?"


def load(log_glob, kind, scorer_name):
    """{(generator, paper_id): score} for one column."""
    out = {}
    for f in sorted(glob.glob(log_glob)):
        log = read_eval_log(f)
        fallback = ((log.eval.model or "?").split("/")[-1] if kind == "gen"
                    else _generator_from_log(f))
        for s in log.samples or []:
            sc = (s.scores or {}).get(scorer_name)
            if sc is None or not isinstance(sc.value, (int, float)):
                continue
            gen = fallback
            if kind == "rescore":
                meta = s.metadata.get("generator") or ""
                if meta and not meta.startswith("none") and meta != "unknown":
                    gen = meta.split("/")[-1]
            out[(gen, s.id)] = float(sc.value)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", default="logs/scorer_agreement",
                    help="root holding runA/ runB/ runC/ runC_haiku/")
    ap.add_argument("--out", default="figures/scorer_agreement/data.json")
    args = ap.parse_args()

    root = Path(args.logroot)
    columns, missing = {}, []
    for col, (subdir, kind, scorer_name) in SOURCES.items():
        pattern = str(root / subdir / "*.eval")
        files = sorted(glob.glob(pattern))
        if not files:
            missing.append(f"  {col}: no logs at {pattern}")
            continue
        columns[col] = load(pattern, kind, scorer_name)
        print(f"  {col}: {len(columns[col])} scores from {len(files)} log(s)")

    # Hard-fail rather than writing an empty file
    if missing:
        raise SystemExit(
            "Missing inspect logs, cannot build the scorer-agreement data:\n"
            + "\n".join(missing)
            + f"\n\nGenerate them first:  bash figures/scorer_agreement/run_data.sh all"
        )

    common = set.intersection(*(set(c) for c in columns.values()))
    print(f"\n(generator, paper) pairs with all 6 scores: {len(common)}")
    if not common:
        raise SystemExit(
            "No pair has all 6 scores. The runs disagree on generators or papers - check "
            "that every phase used the same --paper_ids and the same generator list."
        )

    rows = [
        {"paper_id": pid, "generator": gen,
         **{col: columns[col][(gen, pid)] for col in SOURCES}}
        for gen, pid in sorted(common)
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    print(f"Wrote {len(rows)} rows to {out}")

    print("\nPer-generator counts:")
    for gen, n in sorted(Counter(r["generator"] for r in rows).items(), key=lambda x: -x[1]):
        print(f"  {gen:38s} n={n}")


if __name__ == "__main__":
    main()
