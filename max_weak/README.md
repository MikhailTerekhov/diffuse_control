# Max-weak baseline (OpenEvolve)

Single-objective prompt search: maximize the weak score, ignore the ground truth proxy. 
We use OpenEvolve here rather than our own implementation.

## How it works

`run.py` is a hydra wrapper. The config is resolved and saved, and OpenEvolve is then launched as a subprocess with the resolved config passed via `EVOLVE_CONFIG`:

```
run.py  --(hydra)-->  resolved yaml
        --(subprocess)-->  python -m max_weak.openevolve_main initial_prompt.txt evaluator.py
                             ^ registers the "inspect" LLM provider, then defers to
                               upstream's CLI
```

The evolutionary database (MAP-Elites over islands) and the mutation LLM call are handled by OpenEvolve. Everything task-specific is in `evaluator.py`, which is called for each candidate prompt: proposals are generated with the untrusted model and scored with the weak and (optionally) ground truth proxy judges, and the metrics are returned to OpenEvolve. Evaluation is cascaded, so only the prompts that pass cheap 16-paper screens get expensive 128-paper evaluations.

## Running it

```bash
uv run python max_weak/run.py --config-name config_tiny          # smallest real-API run
uv run python max_weak/run.py --config-name config_max_weak_nostrong
uv run python max_weak/run.py max_iterations=10                  # hydra override
```

With `EVOLVE_DRY_RUN=1`, mock scores are used, so you can check the plumbing without API calls.

You will need the dataset (`paper_pipeline/PIPELINE.md`) and `ANTHROPIC_API_KEY`, as well as `OPENAI_API_KEY` / `OPENROUTER_API_KEY` for non-Anthropic generators.

## Configs

The OpenEvolve settings and the shared judge/generator block are in `base.yaml`. The overlays are:

| Config | For |
|---|---|
| `config_max_weak_nostrong.yaml` | the main max-weak run, `skip_strong_judge: true` |
| `config_max_weak_nostrong_gpt54.yaml` | w\* for the GPT-5.4 generator |
| `config_max_weak_nostrong_gptoss120b.yaml` | w\* for the GPT-OSS-120B generator |
| `config_max_weak_nostrong_detector.yaml` | w\* against the detector weak scorer |
| `config_max_weak_nostrong_outer_evolve_cyc10.yaml` | w\* against the blue team's b\* |
| `config_tiny.yaml` | smallest real-API smoke test: 2-paper screens, 4-paper full evals, stops after 2 |
| `config_test.yaml` | plumbing smoke test, but in *sandbagging* mode rather than max-weak |

The scorer knobs are the same as in `red_team/config/base.yaml`: the options for `weak_scorer_type` are `quality`, `quality_norubric`, `detector` and `outer_evolve`, and the proxy's information advantage is set by `strong_judge_has_paper`. For `outer_evolve`, the path should lead to a blue team `chosen.json` (see `blue_team/README.md`).

## Layout

| File | Contents |
|---|---|
| `run.py` | hydra wrapper: config resolution, split, subprocess launch, post-processing |
| `evaluator.py` | the OpenEvolve evaluator: generate, judge, cascade, return metrics |
| `build_system_message.py` | construction of the mutation operator's system message per mode |
| `inspect_llm.py` | Inspect AI backend for OpenEvolve's mutation LLM (the reconstruction) |
| `openevolve_main.py` | entry point: backend registration, then upstream's CLI |
| `initial_prompt.txt` | the seed generator prompt, same as the red team's |
| `optimizer_system_message.txt` | mutation operator prompt template |
