"""Stage 5: Describe figures using the actual source images + GPT-5.2.

Reads the figure mapping written by stage 4 to find source images in latex_sources/.
Converts PDF/EPS images to PNG, then sends to GPT-5.2 for description.

Usage:
  python describe_figures_latex.py <paper_id>
  python describe_figures_latex.py --all
"""

import asyncio
import base64
import io
import json
import re
import sys
from pathlib import Path

import fitz  # pymupdf
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from utils import (
    TEXT_DIR, FIGURES_DESC_DIR,
    LATEX_SOURCES_DIR, FIGURE_MAPPINGS_DIR,
    load_metadata, save_metadata, ensure_dirs, get_paper_logger, close_paper_logger,
)

SEM = asyncio.Semaphore(32)       # concurrent OpenAI calls
SEM_PAPERS = asyncio.Semaphore(8)  # concurrent papers (bounds thread/memory usage)
CLIENT = AsyncOpenAI()
MODEL = "gpt-5.2-2025-12-11"

PROMPT = """You are describing an academic paper figure for use in a research benchmark. Your description will be used INSTEAD of the actual figure in subsequent analysis, so it must be comprehensive and capture every visual detail.

Describe the figure in detail, including:
- The type of plot/visualization (bar chart, line plot, scatter, diagram, etc.)
- All axis labels, units, and scales (tick marks, ranges)
- Every data series shown: their labels, colors, line styles, markers
- All numerical values you can read from the figure
- Trends, patterns, and relative comparisons between series
- Legends, annotations, inset panels
- Layout details (number of subplots, arrangement)

IMPORTANT: Do NOT analyze or interpret the data. Do NOT draw conclusions about what the results mean. Only describe what is visually present in the figure."""


def _flatten_alpha(img):
    """Composite image onto white background if it has transparency."""
    from PIL import Image
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.convert('RGBA').split()[3])
        return background
    return img


def image_to_png_bytes(image_path: Path) -> bytes | None:
    """Convert any supported image format to PNG bytes. Flattens transparency to white."""
    suffix = image_path.suffix.lower()

    if suffix == '.png':
        from PIL import Image
        img = _flatten_alpha(Image.open(image_path))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    if suffix in ('.jpg', '.jpeg', '.gif', '.bmp', '.tiff'):
        from PIL import Image
        img = _flatten_alpha(Image.open(image_path))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    if suffix == '.pdf':
        try:
            doc = fitz.open(str(image_path))
            page = doc[0]
            pix = page.get_pixmap(dpi=250)
            png_bytes = pix.tobytes("png")
            doc.close()
            return png_bytes
        except Exception:
            return None

    if suffix == '.eps':
        try:
            from PIL import Image
            img = Image.open(image_path)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()
        except Exception:
            return None

    return None


def resolve_image_path(source_dir: Path, image_ref: str) -> Path | None:
    """Resolve an image reference from LaTeX to an actual file path."""
    candidates = [source_dir / image_ref]

    # Try with common extensions
    for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.eps', '.PNG', '.JPG', '.PDF']:
        candidates.append(source_dir / (image_ref + ext))

    # Also try just the filename in case of nested paths
    basename = Path(image_ref).name
    for ext in ['', '.png', '.jpg', '.jpeg', '.pdf', '.eps']:
        candidates.append(source_dir / (basename + ext))

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Last resort: search recursively by stem
    name_no_ext = Path(image_ref).stem
    for f in source_dir.rglob(f"{name_no_ext}.*"):
        if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.pdf', '.eps', '.svg'}:
            return f

    return None


def parse_figure_tags(text: str) -> set[str]:
    """Extract figure numbers/IDs from [FIGURE N] tags in markdown."""
    return set(re.findall(r'\[FIGURE ([^\]]+)\]', text))


async def describe_image(img_bytes: bytes) -> tuple[str, dict]:
    """Send image to GPT-5.2 for description. Returns (description, usage)."""
    b64 = base64.b64encode(img_bytes).decode()

    async with SEM:
        resp = await CLIENT.chat.completions.create(
            model=MODEL,
            max_completion_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }],
        )

    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }
    return resp.choices[0].message.content, usage


async def process_paper(pid: str) -> tuple[int, dict]:
    """Process all figures for a paper. Returns (count, total_usage)."""
    async with SEM_PAPERS:
        return await _process_paper(pid)


async def _process_paper(pid: str) -> tuple[int, dict]:
    log = get_paper_logger(pid)
    source_dir = LATEX_SOURCES_DIR / pid
    desc_dir = FIGURES_DESC_DIR / pid
    desc_dir.mkdir(parents=True, exist_ok=True)

    total_usage = {"input_tokens": 0, "output_tokens": 0}

    # Load figure mapping from stage 4
    mapping_path = FIGURE_MAPPINGS_DIR / f"{pid}.json"
    if not mapping_path.exists():
        log.warning("No figure mapping found, skipping figure descriptions")
        return 0, total_usage

    figure_mapping = json.loads(mapping_path.read_text())
    if not figure_mapping:
        log.info("Empty figure mapping, no figures to describe")
        return 0, total_usage

    # Get figure numbers from extracted text to cross-check
    text_path = TEXT_DIR / f"{pid}.md"
    if text_path.exists():
        fig_nums_in_text = parse_figure_tags(text_path.read_text())
    else:
        fig_nums_in_text = set(figure_mapping.keys())

    described = 0

    async def do_figure(fig_num: str, refs: list[str]):
        nonlocal described

        if not refs:
            log.warning(f"Figure {fig_num}: no image reference")
            return

        descriptions = []
        fig_usage = {"input_tokens": 0, "output_tokens": 0}

        for ref in refs:
            img_path = resolve_image_path(source_dir, ref)
            if not img_path:
                log.warning(f"Figure {fig_num}: could not resolve image '{ref}'")
                continue

            png_bytes = await asyncio.to_thread(image_to_png_bytes, img_path)
            if not png_bytes:
                log.warning(f"Figure {fig_num}: could not convert {img_path.suffix} to PNG")
                continue

            try:
                desc, usage = await describe_image(png_bytes)
                descriptions.append(desc)
                fig_usage["input_tokens"] += usage["input_tokens"]
                fig_usage["output_tokens"] += usage["output_tokens"]
            except Exception as e:
                log.error(f"Figure {fig_num}: API error: {e}")

        if descriptions:
            safe_num = fig_num.replace(".", "_")
            if len(descriptions) > 1:
                parts = "\n\n".join(f"Element {i+1}:\n{d}" for i, d in enumerate(descriptions))
                combined = f"This figure contains {len(descriptions)} elements, described individually below.\n\n{parts}"
            else:
                combined = descriptions[0]
            (desc_dir / f"fig_{safe_num}.txt").write_text(combined)
            log.info(f"Figure {fig_num}: described ({len(combined)} chars)")
            described += 1

        total_usage["input_tokens"] += fig_usage["input_tokens"]
        total_usage["output_tokens"] += fig_usage["output_tokens"]

    # Process figures in parallel
    tasks = []
    for fig_num, image_ref in figure_mapping.items():
        if fig_num in fig_nums_in_text or str(fig_num) in fig_nums_in_text:
            # Normalize: lists stay as lists, strings get split on comma
            if isinstance(image_ref, list):
                refs = image_ref
            elif image_ref and image_ref != "null":
                refs = [r.strip() for r in image_ref.split(",")]
            else:
                refs = []
            tasks.append(do_figure(str(fig_num), refs))

    if tasks:
        await asyncio.gather(*tasks)

    print(f"    {pid}: {described}/{len(tasks)} figures described", flush=True)
    log.info(f"Described {described}/{len(tasks)} figures")
    close_paper_logger(pid)
    return described, total_usage


async def run_async():
    ensure_dirs()

    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        pid = sys.argv[1]
        print(f"Processing {pid} (LaTeX figures)...")
        count, usage = await process_paper(pid)
        inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        cost = inp / 1e6 * 2.50 + out / 1e6 * 10.00
        print(f"\n{count} figures | {inp:,} in, {out:,} out | ${cost:.4f}")
        return

    metadata = load_metadata()
    to_process = [
        pid for pid, info in metadata["papers"].items()
        if info.get("latex_downloaded") and info.get("text_extracted")
        and not info.get("figures_extracted") and info.get("error") is None
    ]

    print(f"Describing figures for {len(to_process)} papers (LaTeX pipeline)...")
    grand = {"input_tokens": 0, "output_tokens": 0, "papers": 0, "figures": 0}

    for pid in to_process:
        try:
            count, usage = await process_paper(pid)
            grand["papers"] += 1
            grand["figures"] += count
            grand["input_tokens"] += usage.get("input_tokens", 0)
            grand["output_tokens"] += usage.get("output_tokens", 0)
            metadata["papers"][pid]["figures_extracted"] = True
            metadata["papers"][pid]["figure_count"] = count
        except Exception as e:
            get_paper_logger(pid).error(f"Figure description failed: {e}")
            metadata["papers"][pid]["error"] = f"figures_latex: {e}"

    save_metadata(metadata)
    cost = grand["input_tokens"] / 1e6 * 2.50 + grand["output_tokens"] / 1e6 * 10.00
    print(f"\nTotal: {grand['figures']} figures, {grand['papers']} papers")
    print(f"Tokens: {grand['input_tokens']:,} in, {grand['output_tokens']:,} out | ${cost:.2f}")


def run():
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
