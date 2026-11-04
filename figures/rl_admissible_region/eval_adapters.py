"""Score RL checkpoints on the shared validation subset.

Reproduces the training setup rather than inventing a new one: the checkpoint is served
through inspect's tinker-sampling provider with the same renderer, the same developer
prompt, the same temperature and token cap, and the same judges the run trained against.
Only the papers change, from training to held-out.

Checkpoints are named `run_dir:step`, resolved against that run's checkpoints.jsonl:

    uv run python figures/rl_admissible_region/eval_adapters.py \
        --group naive \
        --checkpoints "GRPO 1:1=grpo_sw1-1:30" "max-weak=grpo_maxweak:120"

    # against a robustified weak scorer, the one those runs trained against
    uv run python figures/rl_admissible_region/eval_adapters.py \
        --group robust --weak-prompt blue_team/prompts/b_star.json \
        --checkpoints "GRPO 1:1=grpo_cyc10w_sw1-1:30"

Writes data/val_adapters_<group>.json: {label: {weak: [...], strong: [...]}}.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval_harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_ai import Task, eval as inspect_eval  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.model import GenerateConfig, get_model  # noqa: E402
from inspect_ai.scorer import Score, mean, scorer  # noqa: E402
from inspect_ai.solver import generate, system_message  # noqa: E402
import tinker_cookbook.eval.inspect_utils  # noqa: F401,E402  registers "tinker-sampling"

import rl_red_team.judges as judges  # noqa: E402
from eval_harness.experiments_quality_eval import _load_classification  # noqa: E402
from paper_pipeline.utils import RQ_DIR, TEXT_DIR  # noqa: E402
from val_subset import val_ids  # noqa: E402

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_RENDERER = "gpt_oss_medium_reasoning"
SYSTEM_PROMPT = (REPO_ROOT / "rl_red_team" / "system_prompt.txt").read_text().strip()


def resolve_checkpoint(runs_root: Path, spec: str) -> tuple[str, str]:
    """'label=run_dir:step' -> (label, tinker sampler path)."""
    label, _, ref = spec.partition("=")
    if not ref:
        label, ref = spec, spec
    run_name, _, step = ref.partition(":")
    ckpts = runs_root / run_name.strip() / "checkpoints.jsonl"
    if not ckpts.exists():
        raise SystemExit(f"no checkpoints.jsonl in {ckpts.parent}")
    rows = [json.loads(l) for l in ckpts.read_text().splitlines() if l.strip()]
    if step:
        want = int(step)
        row = next((r for r in rows if r.get("batch") == want), None)
        if row is None:
            avail = [r.get("batch") for r in rows]
            raise SystemExit(f"{run_name}: no checkpoint at step {want}. Available: {avail}")
    else:
        row = rows[-1]     # latest
    return label.strip(), row["sampler_path"]


def build_dataset(ids: list[str]) -> MemoryDataset:
    samples = []
    for pid in ids:
        rq = (RQ_DIR / f"{pid}.md").read_text().strip()
        cls = _load_classification(pid)
        rq_full = rq + (f"\n\n---\n\n{cls}" if cls else "")
        samples.append(Sample(input=rq_full, id=pid, metadata={
            "rq": rq_full, "paper": (TEXT_DIR / f"{pid}.md").read_text().strip(),
        }))
    return MemoryDataset(samples)


def judge_scorer():
    """The training judges, scoring weak and proxy on the generated proposals."""
    @scorer(name="sandbag", metrics={"weak": [mean()], "strong": [mean()]})
    def _factory():
        async def score(state, target):
            md = state.metadata
            wk, st = await asyncio.gather(
                judges.judge_weak(state.output.completion, md["rq"],
                                  paper_id=str(state.sample_id)),
                judges.judge_strong(state.output.completion, md["rq"], md["paper"],
                                    paper_id=str(state.sample_id)),
            )
            return Score(
                value={"weak": wk.score or 0.0, "strong": st.score or 0.0},
                metadata={"weak": wk.score, "strong": st.score,
                          "parse_ok": wk.score is not None and st.score is not None},
            )
        return score
    return _factory()


def collect(log) -> dict:
    weak, strong = [], []
    for s in log.samples or []:
        sc = (s.scores or {}).get("sandbag")
        if not sc:
            continue
        w, g = (sc.metadata or {}).get("weak"), (sc.metadata or {}).get("strong")
        if w is not None and g is not None:
            weak.append(w)
            strong.append(g)
    return {"weak": weak, "strong": strong}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="label=run_dir[:step]; omit the step for the latest")
    ap.add_argument("--group", default="naive",
                    help="output key, e.g. naive or robust; names the output file")
    ap.add_argument("--weak-prompt", default=None,
                    help="a blue-team W JSON, when the runs trained against one")
    ap.add_argument("--runs-root", type=Path, default=Path("rl_red_team/runs"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--renderer", default=DEFAULT_RENDERER)
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="must match the run's training cap")
    ap.add_argument("--limit", type=int, default=0, help="papers, 0 = the whole subset")
    ap.add_argument("--out-dir", type=Path, default=Path("figures/rl_admissible_region/data"))
    ap.add_argument("--log-dir", type=Path, default=Path("logs/rl_admissible_region/adapters"))
    args = ap.parse_args()

    # Use the same weak scorer the runs were trained against, or the default rubric.
    if args.weak_prompt:
        judges.load_weak_wprompt(args.weak_prompt)
        print(f"weak scorer: {args.weak_prompt}")
    else:
        print("weak scorer: the default rubric")

    ids = val_ids()
    if args.limit:
        ids = ids[:args.limit]
    dataset = build_dataset(ids)
    print(f"{len(ids)} validation papers, {len(args.checkpoints)} checkpoint(s)")

    out = {}
    for spec in args.checkpoints:
        label, sampler_path = resolve_checkpoint(args.runs_root, spec)
        print(f"\n{label}: {sampler_path}")
        model = get_model(f"tinker-sampling/{args.model}", renderer_name=args.renderer,
                          model_path=sampler_path,
                          config=GenerateConfig(max_tokens=args.max_tokens, temperature=1.0))
        task = Task(dataset=dataset, solver=[system_message(SYSTEM_PROMPT), generate()],
                    scorer=judge_scorer(), model=model, fail_on_error=0.2,
                    name=label.replace(" ", "_").replace(":", "-"))
        logs = inspect_eval([task], log_dir=str(args.log_dir))
        got = collect(logs[0])
        if not got["weak"]:
            print(f"  warn: no scores for {label} (status={logs[0].status})")
            continue
        out[label] = got
        n = len(got["weak"])
        print(f"  n={n} weak={100*sum(got['weak'])/n:.1f} strong={100*sum(got['strong'])/n:.1f}")

    if not out:
        raise SystemExit("\nno checkpoint produced scores; nothing written")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / f"val_adapters_{args.group}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(out)} checkpoint(s) to {dest}")


if __name__ == "__main__":
    main()
