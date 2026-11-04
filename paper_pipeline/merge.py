"""Stage 6: Merge figure descriptions into the paper text.

Produces text_with_figures/{pid}.md -- the canonical paper text used by the eval.

Usage:
  python merge.py              # all papers
  python merge.py <paper_id>   # specific paper
"""

import re
import sys
from tqdm import tqdm

from utils import (
    TEXT_DIR, TEXT_WITH_FIGS_DIR, FIGURES_DESC_DIR,
    load_metadata, ensure_dirs, get_paper_logger, close_paper_logger,
)


def merge_figures_into_text(text: str, pid: str, log) -> str:
    """Replace [FIGURE N]...[/FIGURE N] blocks with figure descriptions."""
    desc_dir = FIGURES_DESC_DIR / pid

    def replace_figure(match):
        fig_id = match.group(1)
        caption = match.group(2).strip() if match.group(2) else ""
        safe_id = fig_id.replace(".", "_")
        desc_path = desc_dir / f"fig_{fig_id}.txt"
        if not desc_path.exists():
            desc_path = desc_dir / f"fig_{safe_id}.txt"

        if not desc_path.exists():
            log.warning(f"No description file for Figure {fig_id}")
            return match.group(0) + "\n[Description not available]\n"

        desc = desc_path.read_text().strip()
        block = f"[FIGURE {fig_id}]\n"
        if caption:
            block += f"Caption: {caption}\n\n"
        block += f"Description:\n{desc}\n"
        block += f"[/FIGURE {fig_id}]"
        return block

    pattern = r"\[FIGURE ([^\]]+)\]\s*(?:Caption:\s*(.*?))?\s*\[/FIGURE \1\]"
    result, count = re.subn(pattern, replace_figure, text, flags=re.DOTALL)
    log.info(f"Merged {count} figure descriptions into text")
    return result


def merge_into_full_text(pid: str) -> bool:
    """Merge figure descriptions into the paper text. Returns success."""
    log = get_paper_logger(pid)
    source = TEXT_DIR / f"{pid}.md"
    if not source.exists():
        return False
    merged = merge_figures_into_text(source.read_text(), pid, log)
    (TEXT_WITH_FIGS_DIR / f"{pid}.md").write_text(merged)
    close_paper_logger(pid)
    return True


def run():
    ensure_dirs()
    metadata = load_metadata()

    if len(sys.argv) > 1:
        pid = sys.argv[1]
        print(f"Merging {pid}...")
        merge_into_full_text(pid)
        print("  done")
        return

    to_process = [
        pid for pid, info in metadata["papers"].items()
        if info.get("text_extracted") and info.get("error") is None
    ]

    if not to_process:
        print("No papers ready for merging.")
        return

    print(f"Merging figures into text for {len(to_process)} papers...")
    for pid in tqdm(to_process, desc="Merging"):
        merge_into_full_text(pid)

    print(f"\nMerged: {len(to_process)}")


if __name__ == "__main__":
    run()
