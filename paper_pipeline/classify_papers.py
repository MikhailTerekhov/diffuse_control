"""Stage 7: Classify papers along theory/experiment and data-type axes.

Reads full paper text and produces a structured classification:
  - Theory vs experiment balance (4-point scale)
  - Data type: synthetic vs real (4-point scale, or N/A)
  - Application domains (short names)

Usage:
  python classify_papers.py <paper_id>
  python classify_papers.py --all
"""

import asyncio
import json
import random
import sys

import anthropic
from tqdm.asyncio import tqdm_asyncio

from utils import (
    CLASSIFY_DIR,
    TEXT_DIR, PAPER_LOGS_DIR,
    load_metadata, save_metadata, ensure_dirs, get_paper_logger, close_paper_logger,
)
from pathlib import Path

SEM = asyncio.Semaphore(64)
SEM_PAPERS = asyncio.Semaphore(128)
CLIENT = anthropic.AsyncAnthropic()
MODEL = "claude-opus-4-6"


CLASSIFY_PROMPT = """Below is the full text of an academic paper in markdown.

Classify it along the following axes. Read the FULL paper carefully before answering.

## 1. Theory vs Experiment Balance

Which category best describes this paper?

- "theory_only": Only theoretical results (proofs, theorems, bounds). No experimental evaluation whatsoever.
- "mostly_theory": Primarily theoretical contributions, but includes a few supporting experiments or simulations.
- "mostly_experiments": Primarily experimental contributions, with some theoretical analysis or motivation.
- "experiments_only": Only experimental results. No theorems, proofs, or formal theoretical analysis.

## 2. Data Type

If the paper has ANY experimental results, classify the data used:

- "synthetic_only": All experiments use synthetic/simulated data (including RL in simulation, generated datasets, toy examples).
- "mostly_synthetic": Primarily synthetic data, with minor use of real-world data.
- "mostly_real": Primarily real-world data (public benchmarks, collected datasets), with minor synthetic components.
- "real_only": All experiments use real-world collected data or established public benchmarks.
- "not_applicable": Paper has no experimental results (theory_only papers).

Notes:
- Standard real-world ML benchmarks (ImageNet, CIFAR, SWE-bench, MMLU, etc.) count as real data.
- RL in game/simulation environments (Atari, MuJoCo, etc.) counts as synthetic.
- Robotics in simulation counts as synthetic; real robot experiments count as real.
- If the paper uses both real and synthetic data (e.g., trains on real data but evaluates on synthetic test sets, or vice versa), classify as "mostly_real" or "mostly_synthetic" depending on which dominates. Never use "real_only" or "synthetic_only" if both types are present.

## 3. Experimental Domains

List the specific real-world application domains that the paper's experiments target. These should be concrete enough that someone could identify the type of datasets or benchmarks involved — NOT broad ML subfields.

Bad examples (too broad): "computer vision", "NLP", "reinforcement learning", "general ML"
Good examples (specific): "autonomous driving", "protein folding", "financial time series", "medical imaging", "robot manipulation", "code generation", "video surveillance", "weather forecasting", "drug discovery", "chip design", "virtual reality", "document OCR", "satellite imagery"

If the paper runs experiments but they are purely methodological with no specific application domain (e.g., generic image classification on CIFAR/ImageNet, generic language modeling on WikiText, generic RL on Atari/MuJoCo), return an empty list [].
If the paper has no experiments, return an empty list [].

## Output format

Return ONLY valid JSON:
{
  "theory_experiment": "theory_only" | "mostly_theory" | "mostly_experiments" | "experiments_only",
  "data_type": "synthetic_only" | "mostly_synthetic" | "mostly_real" | "real_only" | "not_applicable",
  "domains": ["domain1", "domain2"]
}"""


MAX_RETRIES = 3


async def _call_api(full_text: str) -> str:
    collected = ""
    async with SEM:
        async with CLIENT.messages.stream(
            model=MODEL, max_tokens=16384,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT + "\n\n" + full_text}],
            extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14,context-1m-2025-08-07"},
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        ) as stream:
            async for event in stream:
                if hasattr(event, 'type') and event.type == 'content_block_delta':
                    if hasattr(event, 'delta') and event.delta.type == 'text_delta':
                        collected += event.delta.text
    return collected


async def classify(full_text: str, log) -> dict:
    valid_te = {"theory_only", "mostly_theory", "mostly_experiments", "experiments_only"}
    valid_dt = {"synthetic_only", "mostly_synthetic", "mostly_real", "real_only", "not_applicable"}

    for attempt in range(MAX_RETRIES):
        try:
            text = (await _call_api(full_text)).strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            log.info(f"Classification response (attempt {attempt + 1}): {len(text)} chars")

            result = json.loads(text)
            assert result["theory_experiment"] in valid_te
            assert result["data_type"] in valid_dt
            assert isinstance(result["domains"], list)
            return result

        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            wait = min(120, (2 ** attempt) * 5) * random.uniform(0.5, 1.5)
            log.warning(f"API error (attempt {attempt + 1}): {e}, retrying in {wait:.0f}s")
            await asyncio.sleep(wait)

        except (json.JSONDecodeError, AssertionError, KeyError) as e:
            wait = 2 ** attempt * random.uniform(0.5, 1.5)
            log.warning(f"Parse error (attempt {attempt + 1}): {e}, retrying in {wait:.0f}s")
            await asyncio.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts")


async def process_paper(pid: str) -> dict:
    async with SEM_PAPERS:
        return await _process_paper(pid)


async def _process_paper(pid: str) -> dict:
    log = get_paper_logger(pid)

    text_path = TEXT_DIR / f"{pid}.md"
    if not text_path.exists():
        log.error("No text file found")
        return {"paper_classified": False, "error": "classify: no text file"}

    full_text = text_path.read_text()

    try:
        result = await classify(full_text, log)

        CLASSIFY_DIR.mkdir(parents=True, exist_ok=True)
        (CLASSIFY_DIR / f"{pid}.json").write_text(json.dumps(result, indent=2))

        log.info(f"Classification: {result['theory_experiment']}, {result['data_type']}, domains={result['domains']}")

        return {
            "paper_classified": True,
            "theory_experiment": result["theory_experiment"],
            "data_type": result["data_type"],
            "error": None,
        }

    except Exception as e:
        log.error(f"Classification failed: {e}", exc_info=True)
        return {"paper_classified": False, "error": f"classify: {e}"}

    finally:
        close_paper_logger(pid)


async def run_async():
    ensure_dirs()
    CLASSIFY_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()

    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        pid = sys.argv[1]
        print(f"Classifying {pid}...")
        result = await process_paper(pid)
        metadata["papers"][pid].update(result)
        save_metadata(metadata)
        print(f"  {result}")
        return

    to_process = [
        pid for pid, info in metadata["papers"].items()
        if info.get("text_extracted")
        and not info.get("paper_classified")
        and info.get("error") is None
    ]

    if not to_process:
        print("No papers need classification.")
        return

    print(f"Classifying {len(to_process)} papers (64 parallel)...")

    async def do_one(pid):
        result = await process_paper(pid)
        metadata["papers"][pid].update(result)
        return pid

    await tqdm_asyncio.gather(*[do_one(pid) for pid in to_process], desc="Classify")
    save_metadata(metadata)

    done = sum(1 for p in metadata["papers"].values() if p.get("paper_classified"))
    print(f"\nClassified: {done}")


def run():
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
