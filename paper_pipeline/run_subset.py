"""Run pipeline stages on a subset of papers, fully parallel.

All papers are processed concurrently, bounded by a global paper semaphore
to prevent file descriptor exhaustion. Metadata is saved after every paper.

Usage:
  python run_subset.py --conf ICLR --year 2026 --stages 4 5 6 7 8 --limit 5  # 5 papers
  python run_subset.py --conf ICLR --year 2026 --stages 4 5 6 7 8            # full
"""

import argparse
import asyncio
import shutil
import time

from tqdm.asyncio import tqdm_asyncio

from utils import (
    TEXT_DIR, TEXT_WITH_FIGS_DIR, FIGURES_DESC_DIR, FIGURE_MAPPINGS_DIR,
    CLASSIFY_DIR, RQ_DIR, PAPER_LOGS_DIR,
    load_metadata, save_metadata, ensure_dirs,
)

# At most 32 papers doing anything at once (loggers, downloads, API calls)
SEM_PAPERS = asyncio.Semaphore(32)

# Lock + pending updates for batched metadata writes
METADATA_LOCK = asyncio.Lock()
_pending_updates: dict[str, dict] = {}


async def update_metadata(pid: str, updates: dict):
    """Queue an update and flush to disk. Batches concurrent updates."""
    _pending_updates[pid] = updates
    async with METADATA_LOCK:
        if not _pending_updates:
            return
        batch = dict(_pending_updates)
        _pending_updates.clear()
        metadata = load_metadata()
        for p, u in batch.items():
            metadata["papers"][p].update(u)
        save_metadata(metadata)


def filter_papers(metadata, conf, year, limit, ignore_errors=False):
    matching = [
        pid for pid, info in metadata["papers"].items()
        if info["conference"] == conf and info["year"] == year
        and (ignore_errors or info.get("error") is None)
        and info.get("arxiv_id")
    ]
    return matching[:limit] if limit else matching


def cleanup_paper_outputs(pid: str):
    """Remove all per-paper output files for a paper."""
    for d in [TEXT_DIR, TEXT_WITH_FIGS_DIR, RQ_DIR]:
        p = d / f"{pid}.md"
        if p.exists():
            p.unlink()

    for d in [FIGURE_MAPPINGS_DIR, CLASSIFY_DIR]:
        p = d / f"{pid}.json"
        if p.exists():
            p.unlink()

    fig_dir = FIGURES_DESC_DIR / pid
    if fig_dir.exists():
        shutil.rmtree(fig_dir)

    flat_log = PAPER_LOGS_DIR / f"{pid}.log"
    if flat_log.exists():
        flat_log.unlink()
    log_dir = PAPER_LOGS_DIR / pid
    if log_dir.exists():
        shutil.rmtree(log_dir)


async def run_extract(pids):
    """Stage 4: Claude Code agents with LaTeX + PDF."""
    from extract_agent import process_paper

    metadata = load_metadata()
    todo = [pid for pid in pids if not metadata["papers"][pid].get("text_extracted")]
    if not todo:
        print("  All papers already extracted.")
        return

    print(f"  Converting {len(todo)} papers via Claude Code agents (up to 32 parallel)...")

    async def do_one(pid):
        async with SEM_PAPERS:
            result = await process_paper(pid, metadata["papers"][pid])
        await update_metadata(pid, result)
        cost = result.get("agent_cost_usd", 0)
        status = f"ok (${cost:.2f})" if result.get("text_extracted") else "err"
        return pid, status, cost

    results = await tqdm_asyncio.gather(*[do_one(pid) for pid in todo], desc="  Extract")

    ok = sum(1 for _, s, _ in results if s.startswith("ok"))
    total_cost = sum(c for _, _, c in results)
    print(f"  {ok}/{len(todo)} succeeded, total agent cost: ${total_cost:.2f}")


async def run_figures(pids):
    """Stage 5: figure descriptions from source images."""
    from describe_figures_latex import process_paper

    metadata = load_metadata()
    todo = [pid for pid in pids
            if metadata["papers"][pid].get("text_extracted")
            and not metadata["papers"][pid].get("figures_extracted")]
    if not todo:
        print("  All papers already have figures.")
        return

    print(f"  Describing figures for {len(todo)} papers (parallel)...")

    async def do_one(pid):
        async with SEM_PAPERS:
            try:
                count, usage = await process_paper(pid)
                await update_metadata(pid, {
                    "figures_extracted": True, "figure_count": count, "figures_usage": usage,
                })
            except Exception as e:
                await update_metadata(pid, {"error": f"figures: {e}"})
                count = -1
        return pid, count

    results = await tqdm_asyncio.gather(*[do_one(pid) for pid in todo], desc="  Figures")

    ok = sum(1 for _, c in results if c >= 0)
    total_figs = sum(c for _, c in results if c >= 0)
    print(f"  {ok}/{len(todo)} succeeded, {total_figs} figures total")


def run_merge_full(pids):
    """Stage 6: merge figure descriptions into the paper text (CPU-only, fast)."""
    from merge import merge_into_full_text

    metadata = load_metadata()
    todo = [pid for pid in pids
            if metadata["papers"][pid].get("figures_extracted")]
    if not todo:
        print("  No papers ready for merging.")
        return

    print(f"  Merging figures into full text for {len(todo)} papers...")
    ok = sum(1 for pid in todo if merge_into_full_text(pid))
    print(f"  {ok}/{len(todo)} merged")


async def run_classify(pids):
    """Stage 7: classify papers along theory/experiment and data-type axes."""
    from classify_papers import process_paper

    metadata = load_metadata()
    todo = [pid for pid in pids
            if metadata["papers"][pid].get("text_extracted")
            and not metadata["papers"][pid].get("paper_classified")]
    if not todo:
        print("  All papers already classified.")
        return

    print(f"  Classifying {len(todo)} papers (64 parallel)...")

    async def do_one(pid):
        async with SEM_PAPERS:
            result = await process_paper(pid)
        await update_metadata(pid, result)
        return pid, result.get("paper_classified")

    results = await tqdm_asyncio.gather(*[do_one(pid) for pid in todo], desc="  Classify")

    ok = sum(1 for _, success in results if success)
    print(f"  {ok}/{len(todo)} classified")


async def run_research_questions(pids):
    """Stage 8: extract the research questions that form the task input."""
    from research_questions import process_paper

    metadata = load_metadata()
    todo = [pid for pid in pids
            if metadata["papers"][pid].get("text_extracted")
            and not metadata["papers"][pid].get("rq_extracted")]
    if not todo:
        print("  All papers already have research questions.")
        return

    print(f"  Extracting research questions for {len(todo)} papers...")

    async def do_one(pid):
        async with SEM_PAPERS:
            result = await process_paper(pid)
        await update_metadata(pid, result)
        return pid, result.get("rq_extracted")

    results = await tqdm_asyncio.gather(*[do_one(pid) for pid in todo], desc="  RQs")
    ok = sum(1 for _, success in results if success)
    print(f"  {ok}/{len(todo)} extracted")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Max papers (default: all)")
    parser.add_argument("--stages", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    parser.add_argument("--force", action="store_true", help="Re-run stages even if already completed")
    args = parser.parse_args()

    ensure_dirs()
    metadata = load_metadata()

    pids = filter_papers(metadata, args.conf, args.year, args.limit,
                         ignore_errors=args.force)

    print(f"Selected {len(pids)} {args.conf} {args.year} papers\n")

    if args.force:
        print(f"  Cleaning up outputs for {len(pids)} papers (--force)...")
        for pid in pids:
            cleanup_paper_outputs(pid)
            for key in ["text_extracted", "figures_extracted", "paper_classified",
                        "rq_extracted", "agent_cost_usd", "agent_turns"]:
                metadata["papers"][pid].pop(key, None)
            metadata["papers"][pid].pop("error", None)
        save_metadata(metadata)
        print()

    for stage in args.stages:
        start = time.time()
        print(f"Stage {stage}:")

        if stage == 4:
            await run_extract(pids)
        elif stage == 5:
            await run_figures(pids)
        elif stage == 6:
            run_merge_full(pids)
        elif stage == 7:
            await run_classify(pids)
        elif stage == 8:
            await run_research_questions(pids)
        else:
            print(f"  Unknown stage {stage}")

        print(f"  Done in {time.time() - start:.1f}s\n")

    print("All stages complete.")


if __name__ == "__main__":
    asyncio.run(main())
