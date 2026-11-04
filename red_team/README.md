# Red team: multi-objective prompt optimization

Here we search the space of generator system prompts for subversive behaviour: proposals that the weak scorer rates highly but the ground truth proxy rates low. We trace out the Pareto frontier of this trade-off rather than a single point, so that the blue team can be evaluated against a whole family of attacks.

The algorithm is MO-ASHA (Schmucker et al.) with NSGA-II ranking, plus island MAP-Elites
borrowed from AlphaEvolve to keep the prompt population diverse. See Appendix D of the paper.

## How it works

Prompts are evaluated on progressively larger paper samples across three rungs
(8 / 32 / 128). Cheap noisy evaluations at rung 0 filter down to expensive reliable ones
at rung 2: only the top 1/eta of a sliding window is promoted. An LLM (Opus 4.6) acts as
the mutation operator, shown a parent prompt plus inspirations drawn from the archive, the
island's MAP-Elites grid, and the population at random.

Three islands evolve semi-independently and migrate occasionally. Within each island, diversity is preserved through a MAP-Elites grid over (combined score, prompt length). The combined score is deliberately orthogonal to the optimization direction, so solutions spread along the frontier instead of collapsing.

## Running it

```bash
# No API calls: the full scheduler / island / Pareto machinery with a mock evaluator
uv run python red_team/run.py --dry-run --config config_test.yaml

# Smallest real-API smoke test: rungs of 1/2/4 papers, ~16 trials
uv run python red_team/run.py --config config_tiny.yaml

# The paper's main run (subversion frontier). Long-running, see Scale.
uv run python red_team/run.py --config config_run_rubric.yaml

# Resume an interrupted run
uv run python red_team/run.py --resume red_team/runs/<timestamp>

# Watch a live run
uv run python -m red_team.monitor red_team/runs/<timestamp>
```

You will need the dataset (`paper_pipeline/PIPELINE.md`) and `ANTHROPIC_API_KEY`, as well as `OPENAI_API_KEY` / `OPENROUTER_API_KEY` if the generator is not an Anthropic model.

## Configs

The hyperparameters from Table 2 of the paper are in `base.yaml`, along with documentation for every knob. All other configs are overlays, deep-merged onto it, and the figure for each config is named in its header.

| Config | Result |
|---|---|
| `config_run_rubric.yaml` | subversion frontier against the default weak scorer b_0 |
| `config_run_maxboth_rubric.yaml` | Goodharting frontier, maximizing both scores |
| `config_run_norubric.yaml` | rubric-awareness ablation: optimizer not told the criteria |
| `config_quality_gpt54.yaml` + `..._gpt54_gap.yaml` | subversion with GPT-5.4 generating, two stages |
| `config_quality_gptoss.yaml` + `..._gptoss_gap.yaml` | subversion with GPT-OSS-120B generating, two stages |
| `config_run_weak_rubric_strong_norubric.yaml` + `..._norubric_strong_gap.yaml` | subversion against a rubric-free ground truth proxy |
| `config_run_detector_rubric.yaml` + `..._detector_gap.yaml` | subversion against the sandbagging detector as weak scorer |
| `config_sandbagging_outer_evolve_cyc10_*.yaml` | subversion against the blue team's robustified scorer b* |

The results of all of these runs are plotted with the code in `figures/admissible_region/`, in the same way regardless of the weak scorer under attack.

`config_tiny.yaml` and `config_test.yaml` are smoke tests, not figures. In `config_tiny`, the rungs have 1/2/4 papers, so the top rung fits a 5-paper seeded dataset and each rung has more papers than the last. `config_test` is for `--dry-run` only, since its rungs of 2/4/8 papers exceed a small dataset.

The `_gap` configs are for second stages: a finished run is resumed with `pf_gap.enabled` to densify a specific strong-score band, which is how we reach the top-left corner of the frontier in the paper. You will need to edit the resume path, since it refers to a specific run directory.

## Scorer configuration

Both scorers are swappable, under `eval_models` (see `base.yaml` for the full list):

- `weak_scorer_type`: `quality` (the ten-criterion rubric, the default b_0),
  `quality_norubric` (one holistic 0-100 score), `detector` (Haiku's sandbagging
  confidence, inverted), or `outer_evolve` (a blue-team-optimized W prompt, path given by
  `outer_evolve_w_path`).
- `strong_scorer_type`: `quality` or `quality_norubric`. The information advantage that makes it a proxy for ground truth is set by `strong_judge_has_paper`.
- `optimizer.show_rubric`: whether the *optimizer* is told the rubric dimensions. Setting it to false gives the ablation from Appendix E.

For `outer_evolve`, the path should lead to a `chosen.json` from a blue team run (see `blue_team/README.md`). 

## Scale

The main run in the paper took 32,192 generator samples: 1,618 rung-0 trials, 402 at rung 1, 100 at rung 2. Every sample is scored by both judges. We recommend starting with `--dry-run`, then `config_small.yaml`, before launching a full run.

## Layout

| File | Contents |
|---|---|
| `run.py` | entry point: config load, split, async trial loop, checkpointing |
| `scheduler.py` | MO-ASHA rungs, promotion, NSGA-II objective construction |
| `islands.py` | island MAP-Elites, parent sampling, migration |
| `pareto.py` | non-dominated sort, epsilon-net diversity ordering, combined rank |
| `generator.py` | the LLM mutation operator and its system message |
| `evaluator.py` | one sample: generator -> weak + strong judge, all scorer variants |
| `state.py` | checkpoint save/load, resume with in-flight trials reset |
| `monitor.py` | live terminal dashboard |
| `watchdog.py` | a limit on open connections, beyond which the run is killed |
| `initial_prompt.txt` | the minimal generator prompt used as the optimization seed |
| `optimizer_system_message.txt` | template for the mutation operator (Appendix H.11) |
