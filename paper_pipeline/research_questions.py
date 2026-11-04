"""Stage 8: Extract research questions, affiliations, and experiment costs.

Takes the full paper text and produces a structured output with:
  - Affiliations (standardized, no author names)
  - Research questions / problem statement (intro edited to remove experimental details)
  - Experiment costs (total + split by resource, if mentioned)

Usage:
  python research_questions.py <paper_id>
  python research_questions.py --all
"""

import asyncio
import json
import sys

import anthropic
from tqdm.asyncio import tqdm_asyncio

from utils import (
    RQ_DIR,
    TEXT_DIR, PAPER_LOGS_DIR,
    load_metadata, save_metadata, ensure_dirs, get_paper_logger, close_paper_logger,
)
from pathlib import Path

SEM = asyncio.Semaphore(32)
SEM_PAPERS = asyncio.Semaphore(128)
CLIENT = anthropic.AsyncAnthropic()
MODEL = "claude-opus-4-6"
EFFORT = "high"


EXTRACT_PROMPT = """Below is the full text of an academic paper in markdown.

You will produce a structured JSON response with three fields: affiliations, research_questions, and costs.

## 1. Affiliations

Extract the author-affiliation mapping. Replace author names with anonymous labels that preserve ordering and distinguish lead/equal-contribution authors. Format as a bipartite graph:

"affiliations": [
  {"author": "Lead Author 1", "institutions": ["Dept of CS, Stanford University, Stanford, USA"]},
  {"author": "Lead Author 2", "institutions": ["Google DeepMind, London, UK", "Dept of CS, UCL, London, UK"]},
  {"author": "Author 3", "institutions": ["Dept of CS, Stanford University, Stanford, USA"]},
  ...
]

Rules:
- Use "Lead Author N" for authors marked as equal contribution or first authors. Use "Author N" for the rest. Use "Corresponding Author N" (always numbered) if explicitly marked as corresponding.
- Standardize each institution to: "Department, University/Organization, City, Country" (omit fields not available)
- Preserve the original author ordering from the paper

## 2. Research Questions

Extract and edit the paper's introduction to create a clean problem statement that poses the research questions or problems the paper addresses.

The output should read like a well-motivated introduction that identifies the gap/problem, but does NOT reveal:
- The specific method, approach, or solution proposed by the authors
- Experimental setup, datasets used, or evaluation methodology
- Results, findings, or conclusions
- Figure or table references (remove all [FIGURE N] blocks)
- Any named contributions or contribution lists
- Author names

What to KEEP:
- Background and motivation for the research area
- The problem statement or research questions being addressed
- Why the problem is important or challenging
- Relevant context from prior work that frames the problem (but not the paper's own approach)

Edit the text to flow naturally. It should read as a well-motivated problem statement that could serve as a prompt for: "Given this problem, how would you design experiments to address it?"

## 3. Experiment Costs

Search the ENTIRE paper for mentions of the total computational/financial cost of running the paper's experiments. Look for:
- Dollar amounts spent on compute, API calls, human annotation, etc.
- GPU hours, TPU hours, or similar compute time
- Cloud computing costs
- Human labor costs (crowdsourcing, annotation, etc.)

IMPORTANT: Only report costs if you are fairly confident the reported amounts cover ALL or at least the MOST EXPENSIVE experiments in the paper (including appendix). If the paper only mentions the cost of one sub-experiment, one component, or one attack/baseline — but clearly ran many other expensive experiments whose costs are not reported — set total to null. The goal is to capture the full experiment budget, not partial costs that would be misleading.

Return:
- "total": the total cost as a string (e.g. "$5,000", "1,200 GPU hours"), or null if costs are not comprehensively reported
- "splits": a list of {"resource": "...", "cost": "..."} objects for individual cost items, or empty list if no breakdown is available

## Output format

Return ONLY valid JSON:
{
  "affiliations": [
    {"author": "Lead Author 1", "institutions": ["Dept of CS, Stanford University, Stanford, USA"]},
    {"author": "Author 2", "institutions": ["Google DeepMind, London, UK"]}
  ],
  "research_questions": "The edited introduction text here...",
  "costs": {
    "total": "$5,000" or null,
    "splits": [{"resource": "GPU compute", "cost": "$3,000"}, {"resource": "human annotation", "cost": "$2,000"}]
  }
}"""


async def extract_rq(full_text: str, log) -> dict:
    collected = ""
    async with SEM:
        async with CLIENT.messages.stream(
            model=MODEL, max_tokens=65536,
            messages=[{"role": "user", "content": EXTRACT_PROMPT + "\n\n" + full_text}],
            extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT},
        ) as stream:
            async for event in stream:
                if hasattr(event, 'type') and event.type == 'content_block_delta':
                    if hasattr(event, 'delta') and event.delta.type == 'text_delta':
                        collected += event.delta.text
    text = collected.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    log.info(f"RQ response: {len(text)} chars")
    return json.loads(text)


MAX_RETRIES = 4


async def process_paper(pid: str) -> dict:
    async with SEM_PAPERS:
        for attempt in range(MAX_RETRIES):
            result = await _process_paper(pid)
            err = result.get("error", "")
            if not err or ("429" not in err and "overloaded" not in err.lower()
                           and "rate" not in err.lower() and "error" not in err[:5]):
                return result
            wait = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s
            await asyncio.sleep(wait)
        return result


async def _process_paper(pid: str) -> dict:
    log = get_paper_logger(pid)
    log_dir = PAPER_LOGS_DIR / pid
    log_dir.mkdir(parents=True, exist_ok=True)

    text_path = TEXT_DIR / f"{pid}.md"
    if not text_path.exists():
        log.error("No text file found")
        return {"rq_extracted": False, "error": "no text file"}

    full_text = text_path.read_text()

    try:
        result = await extract_rq(full_text, log)

        # Save raw response
        (log_dir / "rq_response.json").write_text(json.dumps(result, indent=2))

        # Build the research_questions markdown with preamble
        lines = []
        lines.append("## Affiliations\n")
        for entry in result.get("affiliations", []):
            author = entry.get("author", "Author")
            insts = entry.get("institutions", [])
            lines.append(f"- {author}: {'; '.join(insts)}")
        lines.append("")

        costs = result.get("costs", {})
        if costs.get("total"):
            lines.append("## Experiment Costs\n")
            lines.append(f"Total: {costs['total']}")
            for split in costs.get("splits", []):
                lines.append(f"- {split['resource']}: {split['cost']}")
            lines.append("")

        lines.append("## Problem Statement\n")
        lines.append(result.get("research_questions", ""))

        rq_text = "\n".join(lines)

        RQ_DIR.mkdir(parents=True, exist_ok=True)
        (RQ_DIR / f"{pid}.md").write_text(rq_text)

        # Metadata
        costs_known = costs.get("total") is not None
        return {
            "rq_extracted": True,
            "experiment_costs_known": costs_known,
            "error": None,
        }

    except Exception as e:
        log.error(f"RQ extraction failed: {e}", exc_info=True)
        return {"rq_extracted": False, "error": f"rq: {e}"}

    finally:
        close_paper_logger(pid)


async def run_async():
    ensure_dirs()
    RQ_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()

    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        pid = sys.argv[1]
        print(f"Extracting research questions for {pid}...")
        result = await process_paper(pid)
        metadata["papers"][pid].update(result)
        save_metadata(metadata)
        print(f"  rq_extracted={result.get('rq_extracted')}, "
              f"costs_known={result.get('experiment_costs_known')}")
        return

    to_process = [
        pid for pid, info in metadata["papers"].items()
        if info.get("text_extracted")
        and not info.get("rq_extracted")
        and info.get("error") is None
    ]

    if not to_process:
        print("No papers need RQ extraction.")
        return

    print(f"Extracting research questions for {len(to_process)} papers...")

    async def do_one(pid):
        result = await process_paper(pid)
        metadata["papers"][pid].update(result)
        return pid

    await tqdm_asyncio.gather(*[do_one(pid) for pid in to_process], desc="RQ Extract")
    save_metadata(metadata)

    done = sum(1 for p in metadata["papers"].values() if p.get("rq_extracted"))
    with_costs = sum(1 for p in metadata["papers"].values() if p.get("experiment_costs_known"))
    print(f"\nExtracted: {done}, with known costs: {with_costs}")


def run():
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
