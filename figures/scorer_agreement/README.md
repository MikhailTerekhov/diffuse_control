# Scorer agreement and variance decomposition

Left panel: ground truth proxy score vs weak score, with a 1-sigma ellipse per generator.

Right panel: where each score's variance comes from. 

## Dataset prerequisite

In the paper, we evaluate on the held-out spotlight/oral set, so those papers have to be in your dataset first:

```bash
cd paper_pipeline
uv run python seed_from_pins.py --split val --limit 5   # or drop --limit for all 304
uv run python run.py 4 5 6 7 8
```

See `paper_pipeline/PIPELINE.md` for the train/validation seeding details.

If your dataset only has posters, you need to pass a list of papers that matches it. Otherwise, no papers pass the paper ID filter and the eval runs on nothing:

```bash
PAPER_IDS=paper_ids/train_posters.txt N_PAPERS=5 bash figures/scorer_agreement/run_data.sh all
```

## Running it

```bash
# 1. Data. Three passes x 10 generators over the validation papers. Expensive! Check on a few papers first.
N_PAPERS=3 bash figures/scorer_agreement/run_data.sh all
bash figures/scorer_agreement/run_data.sh all

# 2. Collect the logs into one JSON
uv run python figures/scorer_agreement/extract.py

# 3. Plot
uv run python figures/scorer_agreement/plot.py --print-ci
```

You will need the dataset (`paper_pipeline/PIPELINE.md`) and these keys:
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`.

With `USETEX=1`, the text is rendered with the paper's Helvetica + LaTeX style, for which you need a working `latex` and `dvipng`. Without it, mathtext is used, and the plot looks nearly identical.

## Files

| File | Contents |
|---|---|
| `run_data.sh` | the four inspect passes (A, B, C, C_haiku) |
| `extract.py` | extraction of the `.eval` logs into `data.json`, one row per (generator, paper) with all 6 scores |
| `plot.py` | both panels; 95% bootstrap CIs with `--print-ci` |

The eval itself is in `eval_harness/experiments_quality_eval.py`: proposals are generated and scored in the task `experiments_quality`, and saved proposals from a prior log are re-scored in the task `rescore_quality`.
