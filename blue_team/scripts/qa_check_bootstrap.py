"""QA helper: print a summary of bootstrap output for a Sonnet QA agent to inspect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blue_team.population import Population


def print_summary(run_dir: Path):
    pop = Population.load(run_dir / "population.json")
    bootstrap = run_dir / "bootstrap"

    kernels = json.loads((bootstrap / "idea_kernels.json").read_text())
    cand_path = bootstrap / "candidate_entries.json"
    candidates = json.loads(cand_path.read_text()) if cand_path.exists() else []

    p_def = pop.prompts.get(pop.p_def_hash)
    s_def = p_def.s_mean if p_def else None
    w_def = p_def.w_mean("W_0") if p_def else None

    print("=" * 80)
    print("BOOTSTRAP DELIVERABLE 1 SUMMARY")
    print("=" * 80)
    print(f"\nRun dir:       {run_dir}")
    print(f"# kernels:     {len(kernels)}")
    print(f"# candidates:  {len(candidates)} (= 20 anchors + 1 p_def expected)")
    print(f"\np_def:")
    print(f"  s_mean (under strong scorer, 128 papers): {s_def}")
    print(f"  w_mean (under W_0 = rubric):              {w_def}")

    anchors_kept = [p for p in pop.prompts.values() if p.role == "anchor"]
    print(f"\n# anchors kept: {len(anchors_kept)} (target ~10, threshold |s - s_def| <= 0.10)")
    print(f"\nKept anchors (with summaries):")
    for p in anchors_kept:
        print(f"  s={p.s_mean:.3f} w_W0={p.w_mean('W_0'):.3f}  | {p.summary[:100] if p.summary else '(no summary yet)'}")

    print(f"\n# F_0 prompts: {len(pop.fronts.get('F_0', []))}")
    f0 = pop.front("F_0")
    print(f"F_0 (s, w_W0) ranges:")
    if f0:
        s_vals = [p.s_mean for p in f0 if p.s_mean is not None]
        w_vals = [p.w_mean('W_0') for p in f0 if p.w_mean('W_0') is not None]
        print(f"  s:  min={min(s_vals):.3f}, max={max(s_vals):.3f}, n={len(s_vals)}")
        print(f"  w0: min={min(w_vals):.3f}, max={max(w_vals):.3f}, n={len(w_vals)}")

    print(f"\nW_0:")
    w0_info = pop.weak_prompts.get("W_0", {})
    print(f"  path:    {w0_info.get('path', '(missing)')}")
    print(f"  summary: {w0_info.get('summary', '(empty)')[:200]}")

    print(f"\nKept anchor s_means relative to s_def (threshold 0.10):")
    for p in anchors_kept:
        if p.s_mean is not None and s_def is not None:
            delta = p.s_mean - s_def
            mark = "✓" if abs(delta) <= 0.10 else "✗"
            print(f"  {mark} delta={delta:+.3f}  | {p.summary[:80] if p.summary else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print_summary(Path(args.run_dir).resolve())


if __name__ == "__main__":
    main()
