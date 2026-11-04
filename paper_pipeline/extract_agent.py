"""Stage 4: Convert papers to markdown using a Claude agent with LaTeX + PDF access.

Each paper gets a Claude Code agent (Opus 4.6, 1M context) that can read both
the LaTeX sources and the compiled PDF to produce accurate markdown with proper
figure numbering and placement.

Downloads both LaTeX sources and PDF from arxiv to ensure version consistency.
Up to 32 agents run in parallel, each in their own workspace directory.

Logging:
  paper_logs/{pid}/main.log    : pipeline orchestration log
  paper_logs/{pid}/agent.log   : full agent transcript (all actions + reflections)

Usage:
  python extract_agent.py              # all papers needing extraction
  python extract_agent.py <paper_id>   # specific paper
"""

import asyncio
import gzip
import io
import json
import logging
import os
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import httpx
from claude_agent_sdk import (
    query, ClaudeAgentOptions,
    AssistantMessage, ResultMessage, SystemMessage,
    TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock,
)

from utils import (
    TEXT_DIR, LATEX_SOURCES_DIR, FIGURE_MAPPINGS_DIR, PAPER_LOGS_DIR,
    load_metadata, save_metadata, ensure_dirs, close_paper_logger,
)

# Arxiv rate limiting: 1 request per second
ARXIV_LOCK = asyncio.Lock()
LAST_ARXIV_REQUEST = 0.0


async def arxiv_rate_limit():
    global LAST_ARXIV_REQUEST
    async with ARXIV_LOCK:
        now = asyncio.get_running_loop().time()
        wait = max(0, 1.0 - (now - LAST_ARXIV_REQUEST))
        if wait > 0:
            await asyncio.sleep(wait)
        LAST_ARXIV_REQUEST = asyncio.get_running_loop().time()


async def download_source(arxiv_id: str, dest_dir: Path, log) -> bool:
    """Download and extract arxiv source tarball."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    await arxiv_rate_limit()

    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            log.error(f"Download failed: HTTP {resp.status_code}")
            return False
        data = resp.content

    log.info(f"Downloaded {len(data)} bytes from arxiv")
    try:
        if data[:2] == b'\x1f\x8b':
            decompressed = gzip.decompress(data)
            try:
                tar = tarfile.open(fileobj=io.BytesIO(decompressed))
                tar.extractall(path=dest_dir, filter='data')
                tar.close()
                log.info(f"Extracted tar archive to {dest_dir}")
                return True
            except tarfile.TarError:
                (dest_dir / "main.tex").write_bytes(decompressed)
                log.info(f"Single gzipped tex file: {len(decompressed)} bytes")
                return True
        try:
            tar = tarfile.open(fileobj=io.BytesIO(data))
            tar.extractall(path=dest_dir, filter='data')
            tar.close()
            log.info("Extracted plain tar archive")
            return True
        except tarfile.TarError:
            pass
        (dest_dir / "main.tex").write_bytes(data)
        log.info(f"Plain tex file: {len(data)} bytes")
        return True
    except Exception as e:
        log.error(f"Failed to extract source: {e}")
        return False

SEM = asyncio.Semaphore(32)

AGENT_PROMPT = r"""You are converting LaTeX source files of an academic paper into clean markdown. Accuracy and completeness are critical.

In your current folder you will find all the LaTeX source files for the paper as well as the compiled PDF (`paper.pdf`). Convert them into a single clean markdown document. Save it as `paper.md`.

Rules:
1. **Math**: Use `$...$` for inline math and `$$...$$` for display equations. Number display equations: `$$equation \tag{1}$$`. Only use the latex sources to find the equations, DO NOT try to infer latex from the PDF. If non-default latex notation is used, add it to the prelude of the markdown document.
2. **Footnotes**: Embed inline: `[^N: footnote text]`
3. **Figures**: Replace EACH figure with EXACTLY this placeholder (on its own lines, blank lines before and after):

[FIGURE N]
Caption: <the figure caption text>
[/FIGURE N]

Use the PDF to understand where the figure is located in the final text and its index, though try not to put the figure placeholder in a way that would break a paragraph. Do NOT describe figures. Do NOT omit any figures. If the figure in the PDF is not numbered, use FIGURE XN where N is an index among the non-numbered figures.

4. **Tables**: Render simple tables as markdown tables. For complex tables, keep as LaTeX `\begin{table}...\end{table}`. Always include captions.
5. **Section headings**: Use `#`, `##`, `###` etc.
6. **References**: Include the full bibliography/references section. Format each entry uniformly as: `[N] Author1, Author2, ..., AuthorN. Title. *Venue*, Year.` — use the information from the .bbl or .bib files provided.
7. **General**: Preserve ALL text faithfully. Do not summarize, skip, or paraphrase.

To prevent context overflow, write the document in chunks, for example write sections or subsections into individual markdown files, and combine them at the end.

After the markdown, save into `figure_mapping.json` a figure mapping as a JSON object:

{"figure_mapping": {"1": "path/to/image.png", "2": ["figures/plot_a.pdf", "figures/plot_b.pdf"], "X1": "figures/non_numbered_image.svg"}}

The mapping should connect each figure number to its source image filename as specified in \includegraphics commands. Use the path exactly as it appears in the LaTeX source. If a figure has subfigures with multiple \includegraphics, use a json list as shown above. If you cannot determine the image file, use null."""


def get_main_logger(pid: str) -> logging.Logger:
    """Logger that writes to paper_logs/{pid}/main.log."""
    log_dir = PAPER_LOGS_DIR / pid
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"paper_agent.{pid}")
    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)
    handler = logging.FileHandler(log_dir / "main.log")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def close_main_logger(pid: str):
    logger = logging.getLogger(f"paper_agent.{pid}")
    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)


class AgentLogger:
    """Writes full agent transcript to paper_logs/{pid}/agent.log."""

    def __init__(self, pid: str):
        log_dir = PAPER_LOGS_DIR / pid
        log_dir.mkdir(parents=True, exist_ok=True)
        self.f = open(log_dir / "agent.log", "w")

    def _ts(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def log(self, message):
        ts = self._ts()

        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    self.f.write(f"[{ts}] ASSISTANT:\n{block.text}\n\n")
                elif isinstance(block, ThinkingBlock):
                    self.f.write(f"[{ts}] THINKING:\n{block.thinking}\n\n")
                elif isinstance(block, ToolUseBlock):
                    inp = json.dumps(block.input, indent=2, default=str)
                    if len(inp) > 10000:
                        inp = inp[:10000] + "\n... (truncated)"
                    self.f.write(f"[{ts}] TOOL USE: {block.name}\n{inp}\n\n")
                elif isinstance(block, ToolResultBlock):
                    content = str(block.content)
                    if len(content) > 5000:
                        content = content[:5000] + "\n... (truncated)"
                    self.f.write(f"[{ts}] TOOL RESULT:\n{content}\n\n")
                else:
                    self.f.write(f"[{ts}] BLOCK ({type(block).__name__}):\n{block}\n\n")

        elif isinstance(message, ResultMessage):
            self.f.write(f"[{ts}] === RESULT ===\n")
            self.f.write(f"  Status: {message.subtype}\n")
            self.f.write(f"  Error: {message.is_error}\n")
            self.f.write(f"  Turns: {message.num_turns}\n")
            cost = message.total_cost_usd or 0
            self.f.write(f"  Cost: ${cost:.4f}\n")
            self.f.write(f"  Duration: {message.duration_ms / 1000:.1f}s\n")
            if message.result:
                self.f.write(f"  Result: {message.result[:2000]}\n")
            self.f.write("\n")

        elif isinstance(message, SystemMessage):
            self.f.write(f"[{ts}] SYSTEM: {message}\n\n")

        else:
            self.f.write(f"[{ts}] {type(message).__name__}\n\n")

        self.f.flush()

    def close(self):
        self.f.close()


async def download_arxiv_pdf(arxiv_id: str, dest_path: Path, log) -> bool:
    """Download compiled PDF from arxiv. Returns True on success."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    await arxiv_rate_limit()

    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            log.error(f"Arxiv PDF download failed: HTTP {resp.status_code}")
            return False
        dest_path.write_bytes(resp.content)

    log.info(f"Downloaded arxiv PDF: {len(resp.content)} bytes")
    return True


async def setup_workspace(pid: str, arxiv_id: str, log) -> Path:
    """Create agent workspace with fresh arxiv sources + PDF."""
    workspace = PAPER_LOGS_DIR / pid / "claude_workspace"

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    # Download fresh latex sources from arxiv
    source_dir = LATEX_SOURCES_DIR / pid
    if source_dir.exists():
        shutil.rmtree(source_dir)
    ok = await download_source(arxiv_id, source_dir, log)
    if not ok:
        return workspace

    # Copy sources into workspace
    shutil.copytree(source_dir, workspace, dirs_exist_ok=True)

    # Download arxiv PDF directly (not OpenReview)
    await download_arxiv_pdf(arxiv_id, workspace / "paper.pdf", log)

    return workspace


async def run_agent(workspace: Path, main_log, agent_logger: AgentLogger):
    """Run Claude Code agent. Returns ResultMessage."""
    result_msg = None
    main_log.info("Agent starting")

    clean_env = {k: "" for k in ["CLAUDECODE"] if k in os.environ}

    try:
        async with asyncio.timeout(3600):
            async for message in query(
                prompt=AGENT_PROMPT,
                options=ClaudeAgentOptions(
                    allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task"],
                    disallowed_tools=["WebSearch", "WebFetch"],
                    permission_mode="bypassPermissions",
                    model="claude-opus-4-6",
                    cwd=str(workspace),
                    max_turns=80,
                    env=clean_env,
                    betas=["context-1m-2025-08-07"],
                ),
            ):
                agent_logger.log(message)
                if isinstance(message, ResultMessage):
                    result_msg = message
    except TimeoutError:
        main_log.error("Agent timed out after 1 hour")

    return result_msg


MAX_RETRIES = 4


async def process_paper(pid: str, info: dict) -> dict:
    async with SEM:
        for attempt in range(MAX_RETRIES):
            result = await _process_paper(pid, info)
            err = result.get("error", "")
            if not err or ("529" not in err and "Overloaded" not in err and "Command failed with exit code" not in err):
                return result
            wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
            log = logging.getLogger(f"paper_agent.{pid}")
            log.warning(f"Overloaded, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            await asyncio.sleep(wait)
        return result


async def _process_paper(pid: str, info: dict) -> dict:
    main_log = get_main_logger(pid)
    agent_logger = AgentLogger(pid)

    try:
        arxiv_id = info.get("arxiv_id")
        if not arxiv_id:
            main_log.error("No arxiv_id")
            return {"text_extracted": False, "error": "no arxiv_id"}

        # Download sources + PDF from arxiv
        workspace = await setup_workspace(pid, arxiv_id, main_log)

        if not (workspace / "paper.pdf").exists():
            main_log.error("Failed to download arxiv PDF")
            return {"text_extracted": False, "latex_downloaded": True,
                    "error": "arxiv pdf download failed"}

        if not any(workspace.rglob("*.tex")):
            main_log.error("No tex files in downloaded sources")
            return {"text_extracted": False, "latex_downloaded": False,
                    "error": "no tex files in arxiv source"}

        main_log.info(f"Workspace ready: {workspace}")

        # Run agent
        result = await run_agent(workspace, main_log, agent_logger)

        if result is None or result.is_error:
            error_msg = result.result if result else "agent returned no result"
            main_log.error(f"Agent failed: {error_msg}")
            return {"text_extracted": False, "latex_downloaded": True,
                    "error": f"agent: {error_msg}"}

        cost = result.total_cost_usd or 0
        main_log.info(f"Agent done: {result.num_turns} turns, ${cost:.4f}, "
                      f"{result.duration_ms / 1000:.1f}s")

        # Read agent outputs
        paper_md = workspace / "paper.md"
        if not paper_md.exists():
            main_log.error("Agent did not produce paper.md")
            return {"text_extracted": False, "latex_downloaded": True,
                    "error": "no paper.md output"}

        markdown = paper_md.read_text()
        main_log.info(f"paper.md: {len(markdown)} chars")
        (TEXT_DIR / f"{pid}.md").write_text(markdown)

        # Figure mapping
        figure_mapping = {}
        fm_path = workspace / "figure_mapping.json"
        if fm_path.exists():
            try:
                raw = json.loads(fm_path.read_text())
                figure_mapping = raw.get("figure_mapping", raw)
                main_log.info(f"Figure mapping: {len(figure_mapping)} entries")
            except json.JSONDecodeError as e:
                main_log.warning(f"Bad figure_mapping.json: {e}")

        FIGURE_MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
        (FIGURE_MAPPINGS_DIR / f"{pid}.json").write_text(
            json.dumps(figure_mapping, indent=2))

        return {
            "text_extracted": True,
            "latex_downloaded": True,
            "agent_cost_usd": cost,
            "agent_turns": result.num_turns,
            "error": None,
        }

    except Exception as e:
        main_log.error(f"Failed: {e}", exc_info=True)
        return {"text_extracted": False, "latex_downloaded": True,
                "error": f"extract_agent: {e}"}

    finally:
        agent_logger.close()
        close_main_logger(pid)
        close_paper_logger(pid)


async def run_async():
    ensure_dirs()
    metadata = load_metadata()

    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        pid = sys.argv[1]
        info = metadata["papers"].get(pid, {})
        print(f"Processing {pid} (agent)...")
        result = await process_paper(pid, info)
        metadata["papers"][pid].update(result)
        save_metadata(metadata)
        print(f"  text_extracted={result['text_extracted']}, "
              f"cost=${result.get('agent_cost_usd', 0):.4f}")
        return

    to_process = {
        pid: info for pid, info in metadata["papers"].items()
        if info.get("arxiv_id") and not info.get("text_extracted")
        and info.get("error") is None
    }

    if not to_process:
        print("No papers need agent extraction.")
        return

    print(f"Processing {len(to_process)} papers with agents...")

    async def do_one(pid, info):
        result = await process_paper(pid, info)
        metadata["papers"][pid].update(result)
        return pid

    from tqdm.asyncio import tqdm_asyncio
    await tqdm_asyncio.gather(*[do_one(pid, info) for pid, info in to_process.items()],
                              desc="Agent extract")

    save_metadata(metadata)
    extracted = sum(1 for p in metadata["papers"].values() if p.get("text_extracted"))
    total_cost = sum(p.get("agent_cost_usd", 0) for p in metadata["papers"].values())
    print(f"\nExtracted: {extracted}, total agent cost: ${total_cost:.2f}")


def run():
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
