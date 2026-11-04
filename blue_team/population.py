"""Population manifest: generator prompts + cached proposals + per-W score caches.

Manifest schema (population.json):
{
  "p_def_hash": "...",
  "prompts": {
    "<prompt_hash>": {
      "role": "anchor" | "p_def" | "F_<k>",
      "prompt_text": "...",
      "summary": "...",
      "samples": [
        {"paper_id": "...", "proposal_path": "...", "s_score": 0.12,
         "w_scores": {"W_0": 0.34, ...}}
      ]
    }
  },
  "weak_prompts": {"W_0": {"path": "...", "summary": "..."}, ...},
  "fronts": {"F_0": ["<hash>", ...], "F_1": [...]}
}
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class CachedSample:
    paper_id: str
    proposal_path: str            # JSON file with "proposals" + cached scores
    s_score: float | None         # strong score (cached, fixed forever)
    w_scores: dict[str, float] = field(default_factory=dict)  # W label -> w_score

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "proposal_path": self.proposal_path,
            "s_score": self.s_score,
            "w_scores": self.w_scores,
        }


@dataclass
class PromptEntry:
    prompt_hash: str
    role: str
    prompt_text: str
    summary: str = ""
    samples: list[CachedSample] = field(default_factory=list)

    @property
    def s_mean(self) -> float | None:
        vs = [s.s_score for s in self.samples if s.s_score is not None]
        return sum(vs) / len(vs) if vs else None

    def w_mean(self, w_label: str) -> float | None:
        vs = [s.w_scores.get(w_label) for s in self.samples
              if s.w_scores.get(w_label) is not None]
        return sum(vs) / len(vs) if vs else None

    def to_dict(self) -> dict:
        return {
            "prompt_hash": self.prompt_hash,
            "role": self.role,
            "prompt_text": self.prompt_text,
            "summary": self.summary,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptEntry":
        return cls(
            prompt_hash=d["prompt_hash"],
            role=d["role"],
            prompt_text=d["prompt_text"],
            summary=d.get("summary", ""),
            samples=[CachedSample(**s) for s in d.get("samples", [])],
        )


@dataclass
class Population:
    manifest_path: Path
    prompts: dict[str, PromptEntry] = field(default_factory=dict)
    weak_prompts: dict[str, dict] = field(default_factory=dict)  # label -> {path, summary}
    fronts: dict[str, list[str]] = field(default_factory=dict)
    p_def_hash: str = ""

    @classmethod
    def load(cls, path: Path) -> "Population":
        if not path.exists():
            return cls(manifest_path=path)
        d = json.loads(path.read_text())
        return cls(
            manifest_path=path,
            prompts={h: PromptEntry.from_dict(p) for h, p in d.get("prompts", {}).items()},
            weak_prompts=d.get("weak_prompts", {}),
            fronts=d.get("fronts", {}),
            p_def_hash=d.get("p_def_hash", ""),
        )

    def save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        d = {
            "p_def_hash": self.p_def_hash,
            "prompts": {h: p.to_dict() for h, p in self.prompts.items()},
            "weak_prompts": self.weak_prompts,
            "fronts": self.fronts,
        }
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(self.manifest_path)

    # --- Mutators ---

    def add_prompt(self, entry: PromptEntry) -> None:
        self.prompts[entry.prompt_hash] = entry

    def add_weak_prompt(self, label: str, path: str, summary: str) -> None:
        self.weak_prompts[label] = {"path": path, "summary": summary}

    def add_front(self, label: str, prompt_hashes: list[str]) -> None:
        self.fronts[label] = prompt_hashes

    # --- Queries ---

    def anchors(self) -> list[PromptEntry]:
        return [p for p in self.prompts.values() if p.role in ("anchor", "p_def")]

    def front(self, label: str) -> list[PromptEntry]:
        return [self.prompts[h] for h in self.fronts.get(label, [])
                if h in self.prompts]

    def latest_w_label(self) -> str:
        if not self.weak_prompts:
            return ""
        # Labels are W_0, W_1, ... — sort by trailing int
        def _idx(label: str) -> int:
            try:
                return int(label.split("_")[-1])
            except ValueError:
                return -1
        return max(self.weak_prompts.keys(), key=_idx)


# --- Stratified P* construction ---

def select_p_star(
    pop: Population,
    cycle: int,
    n_top_per_past: int,
    n_top_current: int,
    n_sparse_current: int,
    w_label_for_topw: str,
    n_sparse_past: int = 3,
) -> list[PromptEntry]:
    """Build the W-phase population P*.

    - All anchors (incl. p_def).
    - From each past PF (F_0 .. F_{cycle-2}): n_top_per_past top-w + n_sparse_past
      sparse picks via max-min in (s, w), among prompts strictly below the
      anchor s-band (s < min_s_among_anchors).
    - From the most-recent PF (F_{cycle-1}): n_top_current top-w + n_sparse_current
      sparse picks via max-min in (s, w), with the same anchor-band filter.

    Filtering by min_s_among_anchors avoids selecting front prompts that overlap
    the honest-anchor region (these don't tell us anything about sandbagging).
    """
    import numpy as np

    from blue_team.sparse_select import select_k_max_min_distance

    anchor_entries = list(pop.anchors())
    out: list[PromptEntry] = list(anchor_entries)

    anchor_s = [a.s_mean for a in anchor_entries if a.s_mean is not None]
    s_floor = min(anchor_s) if anchor_s else float("inf")

    def _by_w(entries: list[PromptEntry]) -> list[PromptEntry]:
        return sorted(entries, key=lambda p: p.w_mean(w_label_for_topw) or -1.0,
                      reverse=True)

    def _below_anchor_band(entries: list[PromptEntry]) -> list[PromptEntry]:
        return [e for e in entries
                if e.s_mean is not None and e.s_mean < s_floor]

    def _add_unique(picks: list[PromptEntry]) -> None:
        existing = {x.prompt_hash for x in out}
        for e in picks:
            if e.prompt_hash not in existing:
                out.append(e)
                existing.add(e.prompt_hash)

    def _sparse_pick(rest: list[PromptEntry], k: int) -> list[PromptEntry]:
        if k <= 0 or not rest:
            return []
        pts = np.array([
            [e.s_mean if e.s_mean is not None else 0.0,
             e.w_mean(w_label_for_topw) if e.w_mean(w_label_for_topw) is not None else 0.0]
            for e in rest
        ])
        k = min(k, len(rest))
        idxs = select_k_max_min_distance(pts, k) if k > 1 else [0]
        return [rest[i] for i in idxs]

    def _pick_from_front(label: str, n_top: int, n_sparse: int) -> None:
        entries = _below_anchor_band(pop.front(label))
        if not entries:
            return
        sorted_entries = _by_w(entries)
        top = sorted_entries[:n_top]
        _add_unique(top)
        rest = sorted_entries[n_top:]
        _add_unique(_sparse_pick(rest, n_sparse))

    # Past PFs
    for k in range(cycle - 1):
        _pick_from_front(f"F_{k}", n_top_per_past, n_sparse_past)

    # Current PF
    if cycle >= 1:
        _pick_from_front(f"F_{cycle - 1}", n_top_current, n_sparse_current)

    return out


def sample_papers_for_w_phase(
    p_star: list[PromptEntry],
    n_papers: int,
    seed: int,
) -> dict[str, list[str]]:
    """For each prompt in P*, pick n_papers random paper_ids from its cached samples.

    Same draw across all candidate W's in a phase (seeded by cycle).
    """
    out: dict[str, list[str]] = {}
    rng = random.Random(seed)
    for p in p_star:
        ids = sorted({s.paper_id for s in p.samples if s.s_score is not None})
        if len(ids) <= n_papers:
            out[p.prompt_hash] = list(ids)
        else:
            out[p.prompt_hash] = rng.sample(ids, n_papers)
    return out
