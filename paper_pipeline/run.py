"""Pipeline orchestrator: build the paper corpus end to end.

Stages
  1  Scrape OpenReview metadata (spotlight/oral across venues)
  2  Scrape ALL accepted papers for the venues in utils.VENUES -- this is where the
     ~2.3k ICLR 2026 posters come from, i.e. the bulk of the dataset
  3  Link papers to arXiv preprints and resolve first-public dates
  4  LaTeX + PDF -> markdown, via a Claude agent
  5  Describe figures with GPT-5.2
  6  Merge figure descriptions into the text (CPU only)
  7  Classify theory/experiment balance and experimental domains
  8  Extract the research questions that form the task input

Stages 4-8 run per venue over every paper with an arxiv_id.

Usage:
  python run.py --status
  python run.py 1 2 3           # metadata only (cheap)
  python run.py 4 5 6 7 8       # the expensive per-paper stages
  python run.py                 # everything

Stages 4-8 make an LLM call per paper, so scope down with run_subset.py --limit first.
"""

import argparse
import asyncio
import sys
import time

from utils import load_metadata, ensure_dirs, VENUES

STAGES = [1, 2, 3, 4, 5, 6, 7, 8]
PER_PAPER_STAGES = [4, 5, 6, 7, 8]


def print_status():
    metadata = load_metadata()
    papers = metadata.get("papers", {})
    total = len(papers)

    if total == 0:
        print("No papers in metadata yet. Run stage 1 first.")
        return

    with_arxiv = sum(1 for p in papers.values() if p.get("arxiv_id"))
    text = sum(1 for p in papers.values() if p.get("text_extracted"))
    figs = sum(1 for p in papers.values() if p.get("figures_extracted"))
    classified = sum(1 for p in papers.values() if p.get("paper_classified"))
    rqs = sum(1 for p in papers.values() if p.get("rq_extracted"))
    errors = sum(1 for p in papers.values() if p.get("error"))
    cost = sum(p.get("agent_cost_usd") or 0 for p in papers.values())

    print(f"\nPipeline Status ({total} papers)")
    print("=" * 40)
    print(f"  Arxiv linked:    {with_arxiv:>5}/{total}")
    print(f"  Text extracted:  {text:>5}/{with_arxiv}")
    print(f"  Figures done:    {figs:>5}/{text}")
    print(f"  Classified:      {classified:>5}/{text}")
    print(f"  Research qs:     {rqs:>5}/{text}")
    print(f"  Errors:          {errors:>5}")
    print(f"  Agent cost:      ${cost:,.2f}")

    by_conf = {}
    for p in papers.values():
        key = f"{p['conference']} {p['year']}"
        d = by_conf.setdefault(key, {"total": 0, "arxiv": 0, "text": 0, "rq": 0})
        d["total"] += 1
        if p.get("arxiv_id"): d["arxiv"] += 1
        if p.get("text_extracted"): d["text"] += 1
        if p.get("rq_extracted"): d["rq"] += 1

    print("\nPer conference:")
    for k in sorted(by_conf):
        v = by_conf[k]
        print(f"  {k}: {v['total']} total, {v['arxiv']} arxiv, {v['text']} text, {v['rq']} rq")


async def run_pipeline(stages):
    ensure_dirs()

    if 1 in stages:
        start = time.time()
        print(f"\n{'='*50}\nStage 1: Scrape spotlight/oral metadata\n{'='*50}")
        from scrape import run
        run()
        print(f"Done in {time.time() - start:.1f}s")

    if 2 in stages:
        start = time.time()
        print(f"\n{'='*50}\nStage 2: Scrape all accepted papers (incl. posters)\n{'='*50}")
        from scrape_all_accepted import run as run_all
        run_all()
        print(f"Done in {time.time() - start:.1f}s")

    if 3 in stages:
        start = time.time()
        print(f"\n{'='*50}\nStage 3: Find arxiv links & dates\n{'='*50}")
        from arxiv_link import run
        run()
        print(f"Done in {time.time() - start:.1f}s")

    per_paper = [s for s in stages if s in PER_PAPER_STAGES]
    if per_paper:
        from run_subset import (
            run_extract, run_figures, run_merge_full, run_classify,
            run_research_questions, filter_papers,
        )

        for _, conf, year in VENUES:
            metadata = load_metadata()
            pids = filter_papers(metadata, conf, year, limit=None)
            if not pids:
                continue

            print(f"\n{'='*50}\n{conf} {year}: {len(pids)} papers\n{'='*50}")

            for stage in per_paper:
                start = time.time()
                if stage == 4:
                    print("\nStage 4: Text extraction")
                    await run_extract(pids)
                elif stage == 5:
                    print("\nStage 5: Figure descriptions")
                    await run_figures(pids)
                elif stage == 6:
                    print("\nStage 6: Merge figures into text")
                    run_merge_full(pids)
                elif stage == 7:
                    print("\nStage 7: Classify papers")
                    await run_classify(pids)
                elif stage == 8:
                    print("\nStage 8: Extract research questions")
                    await run_research_questions(pids)
                print(f"  Done in {time.time() - start:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Paper corpus processing pipeline")
    parser.add_argument("stages", nargs="*", type=int,
                        help=f"Stages to run (default: all). Valid: {STAGES}")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    stages = args.stages if args.stages else list(STAGES)

    for s in stages:
        if s not in STAGES:
            print(f"Unknown stage: {s}. Valid: {STAGES}")
            sys.exit(1)

    asyncio.run(run_pipeline(stages))

    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print_status()


if __name__ == "__main__":
    main()
