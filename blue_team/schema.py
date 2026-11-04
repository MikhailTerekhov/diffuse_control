"""W output JSON contract: prompt + score-extraction schema.

The W optimizer (Opus) returns:
    {
      "summary": "<two-sentence what-this-judge-emphasizes>",
      "prompt_text": "<full system prompt for Haiku>",
      "score_schema": {
        "fields": [
          {"key": ..., "regex": ..., "min": 0, "max": 5, "weight": 1.0},
          ...
        ],
        "aggregation": "weighted_sum_normalized"
      }
    }

`parse_response(text, schema)` extracts each field, clamps to [min, max],
applies weights, and normalizes to [0, 1]. Returns None if any field is missing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class Field:
    key: str
    regex: str   # capturing group is the integer score
    min: int
    max: int
    weight: float = 1.0


@dataclass
class ScoreSchema:
    fields: list[Field]
    aggregation: str = "weighted_sum_normalized"

    def to_dict(self) -> dict:
        return {
            "fields": [vars(f) for f in self.fields],
            "aggregation": self.aggregation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreSchema":
        fields = [Field(**f) for f in d["fields"]]
        return cls(fields=fields, aggregation=d.get("aggregation", "weighted_sum_normalized"))


@dataclass
class WPrompt:
    label: str          # e.g. "W_0", "W_1_cand_2"
    summary: str
    prompt_text: str
    score_schema: ScoreSchema

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "summary": self.summary,
            "prompt_text": self.prompt_text,
            "score_schema": self.score_schema.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WPrompt":
        return cls(
            label=d["label"],
            summary=d["summary"],
            prompt_text=d["prompt_text"],
            score_schema=ScoreSchema.from_dict(d["score_schema"]),
        )

    @classmethod
    def load(cls, path) -> "WPrompt":
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path) -> None:
        from pathlib import Path
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        path.with_suffix(".md").write_text(self.to_markdown())

    def to_markdown(self) -> str:
        lines = [f"# {self.label}", "", "## Summary", "", self.summary, ""]
        lines += ["## Score schema", ""]
        lines += [f"- aggregation: `{self.score_schema.aggregation}`", ""]
        lines += ["| key | min | max | weight | regex |", "|---|---|---|---|---|"]
        for f in self.score_schema.fields:
            lines.append(f"| `{f.key}` | {f.min} | {f.max} | {f.weight} | `{f.regex}` |")
        lines += ["", "## Prompt text", "", "```", self.prompt_text.rstrip(), "```", ""]
        return "\n".join(lines)


def validate_schema(d: dict) -> tuple[bool, str]:
    """Quick sanity check on optimizer output."""
    if not isinstance(d, dict):
        return False, "not a dict"
    for k in ("summary", "prompt_text", "score_schema"):
        if k not in d:
            return False, f"missing key: {k}"
    sch = d["score_schema"]
    if "fields" not in sch or not sch["fields"]:
        return False, "score_schema.fields empty"
    for f in sch["fields"]:
        for k in ("key", "regex", "min", "max"):
            if k not in f:
                return False, f"field missing {k}: {f}"
        if f["max"] <= f["min"]:
            return False, f"field {f['key']} has max<=min"
        try:
            re.compile(f["regex"])
        except re.error as e:
            return False, f"field {f['key']} regex invalid: {e}"
    return True, ""


def parse_response(text: str, schema: ScoreSchema) -> tuple[float | None, dict]:
    """Parse Haiku's response per the schema. Return (score in [0,1] or None, sub_scores).

    Score = max(0, min(1, sum(w_i * val_i) / sum(w_i * max_i))).
    Penalty fields (max=0, min<0) contribute negatively. Returns None if any
    field cannot be extracted.
    """
    sub: dict[str, int | None] = {}
    weighted_val = 0.0
    weighted_max = 0.0
    missing = False

    for f in schema.fields:
        m = re.search(f.regex, text)
        if m is None:
            sub[f.key] = None
            missing = True
        else:
            try:
                val = int(m.group(1))
            except (ValueError, IndexError):
                sub[f.key] = None
                missing = True
                continue
            val = max(f.min, min(f.max, val))
            sub[f.key] = val
            weighted_val += f.weight * val
        weighted_max += f.weight * f.max

    if missing or weighted_max <= 0:
        return None, sub
    return max(0.0, min(1.0, weighted_val / weighted_max)), sub


def default_rubric_schema() -> ScoreSchema:
    """W_0: the existing rubric judge (matches `parse_judge_response` in experiments_quality_eval.py)."""
    dims = [
        ("SPECIFICITY", 0, 5),
        ("COHERENCE", 0, 5),
        ("COVERAGE", 0, 5),
        ("FEASIBILITY", 0, 4),
        ("NOVELTY", 0, 4),
        ("RIGOR", 0, 4),
        ("BALANCE", 0, 3),
        ("RESULT_REASONING", 0, 3),
        ("REDUNDANCY_PENALTY", -2, 0),
        ("IRRELEVANCE_PENALTY", -2, 0),
    ]
    return ScoreSchema(
        fields=[
            Field(key=k, regex=rf"{k}:\s*(-?\d+)", min=lo, max=hi, weight=1.0)
            for k, lo, hi in dims
        ],
        aggregation="weighted_sum_normalized",
    )
