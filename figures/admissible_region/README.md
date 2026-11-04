# The admissible region

The 2D region of (ground truth proxy, weak) score pairs the red team can reach.

Four elements, only the first required:

| Element | Flag | Comes from |
|---|---|---|
| Subversion Pareto frontier | `--pf` (required) | a red team run in `sandbagging` mode |
| Goodharting Pareto frontier | `--goodhart` | a red team run in `strong_weak` mode |
| Max-weak point, defining w\* and w\*-eps | `--max-weak` | the `max_weak/` baseline |
| Benign generator reference lines | `--baselines` | `baselines.py` |

So you can make the figure from a single red team run and add detail as you add more runs.

## Which weak scorer is being attacked

Nothing here is tied to a particular weak scorer: it is set in the config of the red team run. We get both of the paper's admissible region results this way:

| Weak scorer under attack | Red team config | Shape of the region |
|---|---|---|
| default rubric b_0 | `config_run_rubric.yaml` | a flat top: subversion is possible |
| robustified b\* | `config_sandbagging_outer_evolve_cyc10_*.yaml` | near-diagonal: the blue team closed the gap |

You can get both with the same commands, and the contrast between them is the blue team result. Since `blue_team/prompts/b_star.json` is included, you do not need a blue team run for the second one:

```bash
uv run python red_team/run.py --config config_sandbagging_outer_evolve_cyc10_opus.yaml
uv run python figures/admissible_region/eval_pf.py red_team/runs/<ts>
uv run python figures/admissible_region/plot.py \
    --pf red_team/runs/<ts>/generalization/val_scores.json \
    --out-base figures/admissible_region/against_bstar
```

Same for the detector scorer and the rubric-free proxy: pick the config, plot the run.

## Steps

```bash
# 1. Score the run's frontier on the held-out papers. Prompt optimization selects on
#    training papers, so its own numbers are optimistic; this is what gets plotted.
uv run python figures/admissible_region/eval_pf.py red_team/runs/<ts>

# 2. Plot
uv run python figures/admissible_region/plot.py --pf red_team/runs/<ts>/generalization/val_scores.json
```

Adding the optional pieces:

```bash
# Goodharting frontier: a second red team run with mode: strong_weak
uv run python red_team/run.py --config config_run_maxboth_rubric.yaml
uv run python figures/admissible_region/eval_pf.py red_team/runs/<gh>

# max-weak point: score the winning prompt from the max_weak baseline
uv run python figures/admissible_region/eval_pf.py \
    --prompt-file max_weak/runs/<ts>/best_program.txt \
    --out-dir figures/admissible_region/data/max_weak

# benign reference lines (opt-in, see baselines.py)
uv run python figures/admissible_region/baselines.py \
    --models anthropic/claude-opus-4-6 openrouter/openai/gpt-oss-20b

uv run python figures/admissible_region/plot.py \
    --pf        red_team/runs/<sb>/generalization/val_scores.json \
    --goodhart  red_team/runs/<gh>/generalization/val_scores.json \
    --max-weak  figures/admissible_region/data/max_weak/val_scores.json \
    --baselines figures/admissible_region/data/baselines.json
```

## w\* and w\*-eps

w\* is the best weak score any prompt achieves; w\*-eps is a low-sample-size (N=16) lower
confidence bound on it, standing for the blue team's uncertainty. An attack "counts" when it
sits above w\*-eps, so the pair sets the bar the figure is judged against.

With the max-weak run, w\* is measured directly. **Without `--max-weak`, the highest-weak point on the subversion frontier is used as w\***, which is noted in the figure's subtitle. This is a weaker claim: the frontier may not have reached the true maximum, so the bar may sit too low. This is fine for iterating, but not for a reported number.

All frontier prompts are evaluated concurrently (`--max-tasks`, default 6). Otherwise, inspect would run them one at a time, with a single task on the display and the rest only beginning once it finishes.

Logs with a status other than `success` are skipped during extraction, and you get an error rather than an empty `val_scores.json`. A cancelled eval has no scores, so let it finish first.

## Models

In `eval_pf.py`, proposals are scored with the models used in the run itself, read from its `config.yaml`, so validation matches training rather than silently re-scoring with something else. Three roles
are involved: the generator, the weak scorer (inspect role `grader`) and the ground truth
proxy (role `grader2`). Override any of them with `--generator`, `--weak-judge`,
`--strong-judge`. With `--prompt-file` there is no run config, so the paper's defaults apply
(Opus 4.6 generating, Haiku 4.5 weak, Sonnet 4.6 proxy).

## Validation papers

By default, we use the spotlight/oral tier of the dataset, the paper's held-out set, which is disjoint from the posters used for training. It is deliberately read from the **dataset** rather than a run's `split.json`: the split is recorded at launch, so a dataset extended afterwards would look like it has no validation papers. You can override this with `--paper-ids`.

If nothing is built yet:

```bash
cd paper_pipeline
uv run python seed_from_pins.py --split val --limit 5
uv run python run.py 4 5 6 7 8
```

## Notes

In `eval_pf.py`, we reuse the red team's own format suffix via `red_team.evaluator._build_format_suffix`, so validation cannot silently drift from the conditions the prompts were optimized under.

The frontier is read from `checkpoints/final` when a run has finished; otherwise, the highest-numbered periodic checkpoint is used, with a note. There is strictly more data in `final` than in the last periodic checkpoint, and its name does not sort numerically, so this is easy to get wrong. To read a specific checkpoint, pass `--checkpoint`.

By default, train-side scores are averaged over all of a trial's samples. With `--top-rung-only`, only top-rung evaluations are used, and samples inherited from lower rungs are skipped. The number of samples to skip is derived from the run's own `rung_samples`, so this works for any rung configuration.

With `USETEX=1`, the figure is rendered with the paper's Helvetica + LaTeX style, for which you need a working `latex` and `dvipng`. The default is mathtext.

| File | Contents |
|---|---|
| `eval_pf.py` | scoring of a run's frontier, or a single prompt, on the validation papers |
| `plot.py` | the figure; only `--pf` is required |
| `baselines.py` | opt-in: benign generator performance for the reference lines |
