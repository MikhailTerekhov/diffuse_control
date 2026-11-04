"""Render an optimizer call JSON as nicely-formatted markdown.

Handles two schemas:
  - W-optimizer (outer): list of {role, content} messages
  - PF/v2-optimizer: dict with {call_number, model, input, raw_output,
    parsed_prompt, justification, stop_reason, parent_trial_id, ...}

The save sites in w_optimizer.py and red_team/generator.py call
`render_data(data, md_path, stem)` after writing the JSON, so each .json has
a .md sibling automatically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _render_messages(msgs: list[dict]) -> list[str]:
    parts = []
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        content = m.get("content", "")
        parts.append(f"\n---\n\n## Message {i+1} — `{role}` ({len(content):,} chars)\n")
        parts.append(content.rstrip() + "\n")
    return parts


def render_data(data, output_md: Path, stem: str = "") -> Path:
    """Render an in-memory optimizer call payload to markdown."""
    output_md = Path(output_md)
    parts = [f"# {stem or output_md.stem}\n"]

    if isinstance(data, list):
        parts.extend(_render_messages(data))
    elif isinstance(data, dict):
        meta = []
        for k in ("call_number", "model", "parent_trial_id", "stop_reason"):
            if k in data:
                meta.append(f"- **{k}**: `{data[k]}`")
        if meta:
            parts.append("\n".join(meta) + "\n")
        if isinstance(data.get("input"), list):
            parts.append("\n## Input\n")
            parts.extend(_render_messages(data["input"]))
        for section, label in [
            ("raw_output", "Raw output"),
            ("justification", "Justification (extracted)"),
            ("parsed_prompt", "Parsed prompt (extracted)"),
        ]:
            v = data.get(section)
            if isinstance(v, str) and v:
                parts.append(f"\n---\n\n## {label} ({len(v):,} chars)\n")
                parts.append(v.rstrip() + "\n")
    else:
        raise ValueError(f"Unexpected payload type: {type(data).__name__}")

    output_md.write_text("\n".join(parts))
    return output_md


def render(input_json: Path, output_md: Path | None = None) -> Path:
    data = json.loads(Path(input_json).read_text())
    if output_md is None:
        output_md = Path(input_json).with_suffix(".md")
    return render_data(data, output_md, stem=Path(input_json).stem)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: format_optimizer_call.py <input.json> [output.md]")
        sys.exit(1)
    inp = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
    p = render(inp, out)
    print(f"Wrote {p} ({p.stat().st_size:,} bytes)")
