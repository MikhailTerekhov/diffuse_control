"""Score a red team run's Pareto frontier on the held-out validation papers.

Prompt optimization selects on training papers, so the frontier it reports is optimistic.
This re-runs every non-dominated rung-2 prompt on the validation set to get the numbers
the figure actually plots.

Also evaluates a single prompt file, which is how the max-weak point and any benign
reference prompt get their validation scores:

    # a red team run's whole frontier
    uv run python figures/admissible_region/eval_pf.py red_team/runs/<ts>

    # one prompt (e.g. the max-weak winner)
    uv run python figures/admissible_region/eval_pf.py --prompt-file max_weak/runs/<ts>/best_program.txt \
        --out-dir figures/admissible_region/data/max_weak

Writes <out-dir>/val_scores.json (per-prompt weak/strong sample lists, keyed by task name)
and <out-dir>/prompts/index.json (which trial each task came from).

Validation papers default to the spotlight/oral tier, which is the paper's held-out set and
is disjoint from the posters used for training. Override with --paper-ids.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval_harness"))

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.solver import generate, system_message

from eval_harness.experiments_quality_eval import (  # noqa: E402
    build_dataset, quality_judge_blind, quality_judge_paper,
)
from paper_pipeline.utils import CLASSIFY_DIR, METADATA_PATH, RQ_DIR, TEXT_DIR  # noqa: E402
# Same suffix the red team appends during training, imported rather than duplicated so
# validation cannot silently drift from what the prompts were optimized under.
from red_team.evaluator import _build_format_suffix  # noqa: E402

FORMAT_SUFFIX = _build_format_suffix(10)


# Paper defaults, used when there is no run config to read (the --prompt-file case).
DEFAULT_GENERATOR = "anthropic/claude-opus-4-6"
DEFAULT_WEAK_JUDGE = "anthropic/claude-haiku-4-5-20251001"
DEFAULT_STRONG_JUDGE = "anthropic/claude-sonnet-4-6"


def resolve_models(run_dir: Path | None, args) -> dict:
    """Which models to score with.

    Defaults to the models the run itself used, so validation matches training rather than
    silently re-scoring with something else. CLI flags override.
    """
    cfg = {}
    if run_dir and (run_dir / "config.yaml").exists():
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text()).get("eval_models", {}) or {}
    return {
        "generator": args.generator or cfg.get("generator", DEFAULT_GENERATOR),
        "generator_reasoning_effort": cfg.get("generator_reasoning_effort", "high"),
        "weak_judge": args.weak_judge or cfg.get("weak_judge", DEFAULT_WEAK_JUDGE),
        "strong_judge": args.strong_judge or cfg.get("strong_judge", DEFAULT_STRONG_JUDGE),
    }


def _processed(pid: str) -> bool:
    return ((RQ_DIR / f"{pid}.md").exists() and (TEXT_DIR / f"{pid}.md").exists()
            and (CLASSIFY_DIR / f"{pid}.json").exists())


def validation_ids(paper_ids_file: Path | None) -> set[str]:
    """Held-out papers that are actually built in the corpus.

    Deliberately keyed off the corpus rather than a run's split.json: a run records its
    split at launch, so a corpus extended afterwards would look like it has no validation
    papers. Train/val stay disjoint anyway because training uses posters.
    """
    if paper_ids_file:
        wanted = {ln.strip() for ln in Path(paper_ids_file).read_text().split() if ln.strip()}
        source = str(paper_ids_file)
    else:
        meta = json.loads(METADATA_PATH.read_text())["papers"] if METADATA_PATH.exists() else {}
        wanted = {pid for pid, info in meta.items()
                  if info.get("decision") in ("spotlight", "oral")}
        source = "corpus spotlight/oral"
    ids = {pid for pid in wanted if _processed(pid)}
    if not ids:
        raise SystemExit(
            f"no usable validation papers from {source}: {len(wanted)} candidates, none with "
            f"research_questions + text_with_figures + paper_classifications built.\n"
            f"Build some:  cd paper_pipeline && uv run python seed_from_pins.py --split val "
            f"--limit 5 && uv run python run.py 4 5 6 7 8\n"
            f"Or point --paper-ids at a list matching your corpus."
        )
    print(f"Validation papers: {len(ids)} of {len(wanted)} from {source}")
    return ids


def pick_checkpoint(run_dir: Path, override: str | None = None) -> Path:
    """The run's end state.

    A finished run writes checkpoints/final, which holds strictly more than the
    highest-numbered periodic checkpoint. Numeric sorting alone would rank "final" last
    (its suffix is not a digit) and silently read a stale mid-run checkpoint.
    """
    cp_dir = run_dir / "checkpoints"
    if override:
        cp = cp_dir / override if not Path(override).is_absolute() else Path(override)
        if not (cp / "state.json").exists():
            raise SystemExit(f"no state.json in {cp}")
        return cp
    if (cp_dir / "final" / "state.json").exists():
        return cp_dir / "final"
    cps = sorted([p for p in cp_dir.iterdir() if p.is_dir()],
                 key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)
    if not cps:
        raise SystemExit(f"no checkpoints in {cp_dir}")
    print(f"  note: no checkpoints/final, falling back to {cps[-1].name} "
          f"(run may be unfinished)")
    return cps[-1]


def load_rung2_pf(run_dir: Path, top_rung_only: bool, checkpoint: str | None = None) -> list[dict]:
    """Non-dominated rung-2 trials from the run's end state, sorted by strong score."""
    cp = pick_checkpoint(run_dir, checkpoint)
    print(f"Checkpoint: {cp.name}")
    state = json.loads((cp / "state.json").read_text())
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    mode = cfg.get("moasha", {}).get("mode", "sandbagging")

    # Samples accumulate across rungs. To score a prompt on top-rung evaluations only,
    # skip the ones inherited from lower rungs (derived from this run's rung sizes, not
    # a hardcoded constant, so it holds for any rung configuration).
    skip = sum(cfg.get("moasha", {}).get("rung_samples", [8, 32, 128])[:-1]) if top_rung_only else 0

    entries = []
    for tid in state["rungs"][2]["completed"]:
        t = state["trials"][tid]
        samples = t["sample_results"]
        chosen = samples[skip:] if len(samples) > skip else samples
        valid = [s for s in chosen
                 if s.get("weak_score") is not None and s.get("strong_score") is not None]
        if not valid:
            continue
        entries.append({
            "tid": tid,
            "weak": sum(s["weak_score"] for s in valid) / len(valid),
            "strong": sum(s["strong_score"] for s in valid) / len(valid),
            "prompt": t["prompt_text"],
        })

    def dominates(f, e):
        if "sandbagging" in mode:      # maximize weak, minimize strong
            return (f["weak"] >= e["weak"] and f["strong"] <= e["strong"]
                    and (f["weak"] > e["weak"] or f["strong"] < e["strong"]))
        return (f["weak"] >= e["weak"] and f["strong"] >= e["strong"]      # maximize both
                and (f["weak"] > e["weak"] or f["strong"] > e["strong"]))

    pf = [e for e in entries if not any(dominates(f, e) for f in entries if f is not e)]
    pf.sort(key=lambda e: e["strong"])
    print(f"Mode {mode}: {len(entries)} rung-2 trials -> {len(pf)} on the frontier")
    return pf


def make_task(prompt_text, filter_ids, name, metadata):
    return Task(
        dataset=build_dataset(filter_ids=filter_ids, use_classifications=True,
                              post_cutoff_only=True),
        solver=[system_message(prompt_text + FORMAT_SUFFIX),
                generate(extra_headers={"anthropic_beta": "context-1m-2025-08-07"})],
        scorer=[quality_judge_blind(judge_reasoning_effort="high",
                                    judge_reasoning_tokens=16384, role="grader"),
                quality_judge_paper(judge_reasoning_effort="high", role="grader2")],
        fail_on_error=0.1,
        name=name,
        metadata=metadata,
    )


def extract_val_scores(log_dir: Path) -> dict:
    """Per-task weak/strong sample lists, pulled back out of the .eval logs."""
    out = {}
    for f in sorted(log_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(f))
        except Exception as ex:
            print(f"  warn: could not read {f.name}: {ex}")
            continue
        name = (log.eval.task or "").split("/")[-1]
        if log.status != "success":
            print(f"  skipping {name} ({f.name[:24]}...): status={log.status}")
            continue
        weak, strong = [], []
        for s in log.samples or []:
            for scorer_name, sc in (s.scores or {}).items():
                try:
                    v = float(sc.value) / 100.0
                except (TypeError, ValueError):
                    continue
                if "grader2" in scorer_name or "paper" in scorer_name:
                    strong.append(v)
                elif "grader" in scorer_name or "blind" in scorer_name:
                    weak.append(v)
        if weak and strong:
            md = log.eval.metadata or {}
            out[name] = {"weak": weak, "strong": strong, "trial_id": md.get("trial_id"),
                         "train_weak": md.get("train_weak"), "train_strong": md.get("train_strong")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", help="red team run directory")
    ap.add_argument("--prompt-file", type=Path, default=None,
                    help="score a single prompt instead of a run's frontier")
    ap.add_argument("--label", default="prompt", help="task name for --prompt-file")
    ap.add_argument("--out-dir", default=None,
                    help="default: <run_dir>/generalization")
    ap.add_argument("--paper-ids", type=Path, default=None,
                    help="validation paper id list (default: corpus spotlight/oral)")
    ap.add_argument("--generator", default=None,
                    help="override the generator model (default: the run's own)")
    ap.add_argument("--weak-judge", default=None, help="override the weak scorer model")
    ap.add_argument("--strong-judge", default=None,
                    help="override the ground-truth-proxy scorer model")
    ap.add_argument("--max-connections", type=int, default=32)
    ap.add_argument("--max-tasks", type=int, default=6,
                    help="prompts evaluated concurrently (1 = sequential)")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint to read (default: final, else the highest-numbered)")
    ap.add_argument("--top-rung-only", action="store_true",
                    help="average only top-rung samples for the train-side scores")
    ap.add_argument("--skip-eval", action="store_true",
                    help="only re-extract val_scores.json from logs already present")
    args = ap.parse_args()

    if not args.run_dir and not args.prompt_file:
        ap.error("give a run_dir, or --prompt-file")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.run_dir:
        out_dir = Path(args.run_dir) / "generalization"
    else:
        ap.error("--prompt-file needs --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"

    if not args.skip_eval:
        filter_ids = validation_ids(args.paper_ids)
        tasks, index = [], {}
        if args.prompt_file:
            text = args.prompt_file.read_text().strip()
            tasks.append(make_task(text, filter_ids, args.label, {"source": str(args.prompt_file)}))
            index[args.label] = {"source": str(args.prompt_file)}
        else:
            pf = load_rung2_pf(Path(args.run_dir), args.top_rung_only, args.checkpoint)
            prompts_dir = out_dir / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            for i, e in enumerate(pf):
                name = f"pf_{i}"
                (prompts_dir / f"{name}.txt").write_text(e["prompt"])
                tasks.append(make_task(e["prompt"], filter_ids, name, {
                    "trial_id": e["tid"], "train_weak": e["weak"], "train_strong": e["strong"],
                }))
                index[name] = {"trial_id": e["tid"], "train_weak": e["weak"],
                               "train_strong": e["strong"]}
            (prompts_dir / "index.json").write_text(json.dumps(index, indent=2))

        models = resolve_models(Path(args.run_dir) if args.run_dir else None, args)
        print(f"Evaluating {len(tasks)} prompt(s) on {len(filter_ids)} papers -> {log_dir}")
        print(f"  generator     {models['generator']} "
              f"({models['generator_reasoning_effort']} reasoning)")
        print(f"  weak scorer   {models['weak_judge']}        [role: grader]")
        print(f"  proxy scorer  {models['strong_judge']}      [role: grader2]")
        inspect_eval(
            tasks,
            model=models["generator"],
            model_roles={"grader": models["weak_judge"], "grader2": models["strong_judge"]},
            reasoning_effort=models["generator_reasoning_effort"],
            log_dir=str(log_dir),
            max_connections=args.max_connections,
            # Without max_tasks inspect runs the prompts one after another, so the display
            # shows a single task and the rest only start once it finishes.
            max_tasks=min(args.max_tasks, len(tasks)),
            fail_on_error=0.1,
        )
    else:
        index = {}

    val = extract_val_scores(log_dir)
    if not val:
        raise SystemExit(
            f"\nNo scored prompts in {log_dir}. Nothing was written.\n"
            f"Logs with status other than 'success' are skipped: a cancelled run scores "
            f"nothing, so let the eval finish before extracting."
        )
    (out_dir / "val_scores.json").write_text(json.dumps(val, indent=2))
    print(f"\nWrote {len(val)} entries to {out_dir / 'val_scores.json'}")
    for name, v in sorted(val.items()):
        print(f"  {name:10s} n={len(v['weak']):<3} "
              f"val weak={100*sum(v['weak'])/len(v['weak']):.1f} "
              f"strong={100*sum(v['strong'])/len(v['strong']):.1f}")
    print(f"\nNext: uv run python figures/admissible_region/plot.py --pf {out_dir / 'val_scores.json'}")


if __name__ == "__main__":
    main()
