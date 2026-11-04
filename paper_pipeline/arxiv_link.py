"""Stage 3: Find arxiv links and first-upload dates for each paper.

Uses a leaky bucket rate limiter (1 req/sec) for arxiv API,
with exponential backoff on 429 errors.
"""

import asyncio
import re
import arxiv
import anthropic
from tqdm.asyncio import tqdm_asyncio
from utils import load_metadata, save_metadata

MODEL = "claude-haiku-4-5-20251001"

SEM = asyncio.Semaphore(128)  # for Claude API calls
CLIENT = anthropic.AsyncAnthropic()


class LeakyBucket:
    """Async leaky bucket rate limiter. Allows 1 request per `interval` seconds."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait = self.interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_running_loop().time()


ARXIV_BUCKET = LeakyBucket(interval=1.0)
MAX_RETRIES = 5


async def search_arxiv_throttled(title: str) -> list[arxiv.Result]:
    """Search arxiv with rate limiting and exponential backoff on 429."""
    for attempt in range(MAX_RETRIES):
        await ARXIV_BUCKET.acquire()
        try:
            results = await asyncio.to_thread(_search_arxiv_sync, title)
            return results
        except Exception as e:
            if "429" in str(e):
                wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s, 80s
                await asyncio.sleep(wait)
                continue
            raise
    return []


def _search_arxiv_sync(title: str) -> list[arxiv.Result]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=f'ti:"{title}"',
        max_results=10,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return list(client.results(search))


def titles_match(a: str, b: str) -> bool:
    def normalize(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())
    return normalize(a) == normalize(b)


async def claude_match_title(paper_title: str, candidates: list[dict]) -> str | None:
    candidates_str = "\n".join(
        f"  [{i}] \"{c['title']}\" (arxiv: {c['arxiv_id']})"
        for i, c in enumerate(candidates)
    )
    prompt = f"""I'm looking for this paper on arxiv:
Title: "{paper_title}"

Here are candidate arxiv results:
{candidates_str}

Which candidate (if any) is the same paper? Titles may differ slightly between OpenReview and arxiv
(revised, shortened, reworded). Respond with ONLY the arxiv ID (e.g., "2401.12345") or "NONE" if no match."""

    async with SEM:
        resp = await CLIENT.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
    text = resp.content[0].text.strip()
    match = re.search(r"\d{4}\.\d{4,5}", text)
    return match.group(0) if match else None


def extract_arxiv_info(result: arxiv.Result) -> dict:
    arxiv_id = result.entry_id.split("/abs/")[-1]
    arxiv_id_clean = re.sub(r"v\d+$", "", arxiv_id)
    return {
        "arxiv_id": arxiv_id_clean,
        "title": result.title,
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id_clean}",
        "published": result.published,
    }


async def process_paper(pid: str, info: dict) -> dict | None:
    title = info["title"]
    if not title:
        return None

    try:
        results = await search_arxiv_throttled(title)
        if not results:
            return None

        candidates = [extract_arxiv_info(r) for r in results]

        for c in candidates:
            if titles_match(title, c["title"]):
                return {
                    "arxiv_id": c["arxiv_id"],
                    "arxiv_url": c["arxiv_url"],
                    "arxiv_first_submitted": c["published"].strftime("%Y-%m-%d"),
                }

        matched_id = await claude_match_title(title, candidates)
        if matched_id:
            for c in candidates:
                if c["arxiv_id"] == matched_id:
                    return {
                        "arxiv_id": c["arxiv_id"],
                        "arxiv_url": c["arxiv_url"],
                        "arxiv_first_submitted": c["published"].strftime("%Y-%m-%d"),
                    }

        return None
    except Exception as e:
        print(f"  Error for {pid}: {e}")
        return None


SAVE_EVERY = 100  # save metadata every N papers


async def run_async():
    metadata = load_metadata()
    to_process = {
        pid: info for pid, info in metadata["papers"].items()
        if info.get("arxiv_id") is None and info.get("error") is None
    }

    if not to_process:
        print("No papers need arxiv linking.")
        return

    print(f"Finding arxiv links for {len(to_process)} papers...")

    found = 0
    processed = 0

    async def process_one(pid, info):
        nonlocal found, processed
        result = await process_paper(pid, info)
        if result:
            metadata["papers"][pid].update(result)
            found += 1
        processed += 1
        if processed % SAVE_EVERY == 0:
            save_metadata(metadata)
        return pid, result is not None

    tasks = [process_one(pid, info) for pid, info in to_process.items()]
    await tqdm_asyncio.gather(*tasks, desc="Arxiv linking")

    save_metadata(metadata)
    print(f"\nFound arxiv links for {found}/{len(to_process)} papers")


def run():
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
