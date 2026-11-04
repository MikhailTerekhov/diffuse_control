# Diffuse AI Control on Fuzzy Tasks

Here you will find the code for the paper *Diffuse AI Control on Fuzzy Tasks* ([arXiv](https://arxiv.org/abs/2606.08892), [blog post](https://alignment.anthropic.com/2026/diffuse-ai-control/)).

We do not provide the dataset of the processed papers, because it is derived from
CC-licensed conference papers that we cannot redistribute. You can, however, rebuild the dataset or a part of it yourself using the code in `paper_pipeline/`. For the same reason, we could not provide the logs of our runs. If you want to reproduce the results, you will need to run both dataset construction and the scripts for the individual experiments.

## Setup

```bash
uv sync
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
export FUZZY_DATA_DIR=/path/to/scratch    # optional; default ./data/papers
```

`FUZZY_DATA_DIR` is the root of the dataset that you will generate.
To run the experiments in `rl_red_team/`, you will additionally need `TINKER_API_KEY`.

## Repository structure

| Directory | Contents |
|---|---|
| `paper_pipeline/` | Code for building the dataset from public sources, in eight stages driven by `run.py`: scraping accepted papers, linking them to arXiv, converting LaTeX and PDF to markdown, describing the figures and merging them into the text, classifying each paper, and extracting the research questions that form the task input. |
| `paper_ids/` | The set of papers we used, pinned by ID, so that you can rebuild our dataset rather than whatever OpenReview returns today. |
| `eval_harness/` | The task itself, used in all of the experiments: proposals are generated from a paper's research questions and scored with the weak scorer and the ground truth proxy scorer. Both scorers are configurable, including from a rubric or from a blue team prompt. |
| `red_team/` | Multi-objective prompt optimization (MO-ASHA plus island MAP-Elites) over generator system prompts. The result is the Pareto frontier of the "high weak score, low proxy score" trade-off rather than a single attack. |
| `max_weak/` | Single-objective baseline, where only the weak score is maximized and the proxy is ignored. We use it to estimate the achievable ceiling w\*, and with it the selection margin w\*-eps that the blue team has to work with. Based on upstream OpenEvolve with a provider shim. |
| `blue_team/` | Saddle-point optimization of the weak scorer: the scorer is patched against every attack found so far, the red team then attacks the patched scorer, and the two steps alternate until new attacks stop working. |
| `blue_team/prompts/b_star.json` | The robustified weak scorer b\* from our run of this loop, included so that you can use the blue team result without rerunning the loop. |
| `rl_red_team/` | The red team with gradient access instead of prompt access: GRPO plus LoRA fine-tuning of an open-weight generator through the Tinker API, starting from a prompt on the frontier. |

The components depend on each other in this order. All of the optimizers score their candidates with the code in `eval_harness/`, for which you need the dataset. The blue team defends against the attacks found with `red_team/` and `max_weak/`, and we then test b\* in `rl_red_team/` and in the figures below.

You can find the commands and the configuration in the README of each directory, and the details of the dataset stages in `paper_pipeline/PIPELINE.md`.

## Reproducing the results

Each of these directories is self-contained, with the scripts for generating the data and a script for plotting it.

| Directory | Result |
|---|---|
| `figures/scorer_agreement/` | How much signal the weak score carries: proxy against weak score per generator, plus a decomposition of where each score's variance comes from. |
| `figures/admissible_region/` | The region of (proxy, weak) score pairs the red team can reach, for any weak scorer. Its flat top is the central claim: there are prompts with a weak score near its ceiling and a much lower proxy score. Untuned baselines, the max-weak point and a Goodharting frontier are optional overlays. |
| `figures/rl_trajectories/` | The proxy score while RL pushes the weak score up, against the training step and in the score plane. If you run the same setup against two weak scorers, the contrast is the blue team result: the two scores come apart under the default rubric and move together under b\*. |
| `figures/rl_admissible_region/` | RL checkpoints beside prompt optimization frontier prompts, all scored on the same held-out papers with bootstrap CIs, so that the two kinds of optimization can be compared directly. |

## Building the dataset

The papers we used are pinned in `paper_ids/dataset.csv`, each with its arXiv ID. If you seed the pipeline from this file, you get our dataset and can skip the scraping stages:

```bash
cd paper_pipeline
uv run python seed_from_pins.py --limit 5      # smoke-test on 5 papers first
uv run python run.py 4 5 6 7 8
uv run python run.py --status

uv run python seed_from_pins.py                # then the full set
uv run python run.py 4 5 6 7 8
```

See `paper_pipeline/PIPELINE.md` for the details, the per-stage credentials and the alternative rescrape-from-scratch path. **Note that dataset construction is expensive; in our experiments, the API costs to generate the clean dataset samples were about $5/paper.**

## Run outputs

The outputs of each optimizer run are stored in a timestamped directory under its component (`red_team/runs/<timestamp>/` and so on) together with the run's config, so you can read the settings back later rather than reconstruct them. Runs in `red_team/` and `rl_red_team/` can also be resumed from these directories after an interruption. These directories, the inspect eval logs and the generated figures are all gitignored.
