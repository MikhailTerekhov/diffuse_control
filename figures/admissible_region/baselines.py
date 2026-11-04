"""Benign generator performance on the validation papers, for the admissible region's reference lines.

NOT part of the admissible-region pipeline: `plot.py --baselines` is optional and the figure is
complete without it. Run this only when you want the vertical reference lines that say how
a benign model of a given capability scores, which is what makes a claim like "the attack
drags Opus below GPT-OSS-20B" legible.

Each model generates with the *default* prompt, so this measures the model, not an attack.

    # the paper's framing needs at least the untrusted model and a weak reference
    uv run python figures/admissible_region/baselines.py \
        --models anthropic/claude-opus-4-6 openrouter/openai/gpt-oss-20b

    # the full set behind the paper's annotations
    uv run python figures/admissible_region/baselines.py --models \
        anthropic/claude-opus-4-6 anthropic/claude-haiku-4-5-20251001 \
        openrouter/openai/gpt-oss-20b

Writes figures/admissible_region/data/baselines.json:
    {"<label>": {"weak": <0-1>, "strong": <0-1>, "model": ..., "n": ...}, ...}

One entry is labelled "default" (the untrusted model with the default prompt), which
plot.py draws in a distinct colour via --default-prompt-label.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval_harness"))

from inspect_ai import Task, eval as inspect_eval  # noqa: E402
from inspect_ai.solver import generate, system_message  # noqa: E402

from eval_harness.experiments_quality_eval import (  # noqa: E402
    build_dataset, quality_judge_blind, quality_judge_paper, _load_generator_prompt,
)
from red_team.evaluator import _build_format_suffix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pf import (  # noqa: E402
    DEFAULT_STRONG_JUDGE, DEFAULT_WEAK_JUDGE, extract_val_scores, validation_ids,
)

# Short labels for the figure. Anything not here falls back to the model's short name.
LABELS = {
    "anthropic/claude-opus-4-6": "Benign Opus 4.6",
    "anthropic/claude-opus-4-1": "Benign Opus 4.1",
    "anthropic/claude-haiku-4-5-20251001": "Benign Haiku 4.5",
    "openrouter/openai/gpt-oss-20b": "Benign GPT-OSS-20B",
    "openrouter/openai/gpt-oss-120b": "Benign GPT-OSS-120B",
    "openai/gpt-5.4-nano": "Benign GPT-5.4-nano",
    "openai/gpt-5.4-mini": "Benign GPT-5.4-mini",
    "openai/gpt-5.4": "Benign GPT-5.4",
}

REASONING = {
    "anthropic/claude-haiku-4-5-20251001": {"reasoning_tokens": 8192},
    "anthropic/claude-opus-4-1": {"reasoning_tokens": 8192},
}
DEFAULT_REASONING = {"reasoning_effort": "high"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="generator models to measure with the default prompt")
    ap.add_argument("--untrusted", default="anthropic/claude-opus-4-6",
                    help="which model is the untrusted generator, labelled 'default'")
    ap.add_argument("--prompt-style", default="minimal", choices=["minimal", "default"],
                    help="generator prompt; 'minimal' is the red team's optimization seed")
    ap.add_argument("--paper-ids", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("figures/admissible_region/data/baselines.json"))
    ap.add_argument("--log-dir", type=Path, default=Path("logs/admissible_region_baselines"))
    ap.add_argument("--weak-judge", default=DEFAULT_WEAK_JUDGE)
    ap.add_argument("--strong-judge", default=DEFAULT_STRONG_JUDGE)
    ap.add_argument("--max-connections", type=int, default=32)
    args = ap.parse_args()

    filter_ids = validation_ids(args.paper_ids)
    prompt = _load_generator_prompt(args.prompt_style) + _build_format_suffix(10)

    tasks = []
    for model in args.models:
        tasks.append(Task(
            dataset=build_dataset(filter_ids=filter_ids, use_classifications=True,
                                  post_cutoff_only=True),
            solver=[system_message(prompt),
                    generate(extra_headers={"anthropic_beta": "context-1m-2025-08-07"})],
            scorer=[quality_judge_blind(judge_reasoning_effort="high",
                                        judge_reasoning_tokens=16384, role="grader"),
                    quality_judge_paper(judge_reasoning_effort="high", role="grader2")],
            fail_on_error=0.1,
            name=model.split("/")[-1],
            metadata={"model": model},
        ))

    print(f"Measuring {len(tasks)} generator(s) on {len(filter_ids)} papers "
          f"with the {args.prompt_style} prompt")
    print(f"  weak scorer   {args.weak_judge}        [role: grader]")
    print(f"  proxy scorer  {args.strong_judge}      [role: grader2]")
    for model, task in zip(args.models, tasks):
        cfg = REASONING.get(model, DEFAULT_REASONING)
        inspect_eval(
            [task],
            model=model,
            model_roles={"grader": args.weak_judge, "grader2": args.strong_judge},
            log_dir=str(args.log_dir),
            max_connections=args.max_connections,
            fail_on_error=0.1,
            **cfg,
        )

    scores = extract_val_scores(args.log_dir)
    out = {}
    for model in args.models:
        short = model.split("/")[-1]
        v = scores.get(short)
        if not v:
            print(f"  warn: no scores recovered for {model}")
            continue
        label = "default" if model == args.untrusted else LABELS.get(model, short)
        out[label] = {
            "model": model,
            "weak": sum(v["weak"]) / len(v["weak"]),
            "strong": sum(v["strong"]) / len(v["strong"]),
            "n": len(v["weak"]),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(out)} baselines to {args.out}")
    for label, v in sorted(out.items(), key=lambda kv: kv[1]["strong"]):
        print(f"  {label:24s} weak={100*v['weak']:.1f} strong={100*v['strong']:.1f} n={v['n']}")
    print(f"\nUse with: uv run python figures/admissible_region/plot.py --pf ... --baselines {args.out}")


if __name__ == "__main__":
    main()
