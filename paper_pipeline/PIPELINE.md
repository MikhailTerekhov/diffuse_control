# Paper dataset pipeline

Here you will find the code for building the experiment planning dataset from public sources. We cannot redistribute the dataset itself, since the papers are CC-licensed, so you will need to rebuild it locally.

The dataset consists of **NeurIPS 2025 + ICLR 2026** papers (both after the generators' knowledge cutoff), as defined in `utils.VENUES`. The exact paper set we used is pinned in `../paper_ids/`.

## Output layout

All outputs are stored under `$FUZZY_DATA_DIR` (default `../data/papers`), never inside the repo:

```
metadata.json                     central state: IDs, titles, decisions, dates, stage flags
text/{pid}.md                     stage 4: markdown from LaTeX+PDF
figures_description/{pid}/         stage 5: one .txt per figure
text_with_figures/{pid}.md        stage 6: CANONICAL paper text used by the eval
paper_classifications/{pid}.json  stage 7: theory/experiment balance + domains
research_questions/{pid}.md       stage 8: CANONICAL task input
latex_sources/, figure_mappings/, paper_logs/    intermediates
```

Only `text_with_figures/`, `research_questions/`, `paper_classifications/` and
`metadata.json` are read downstream.

## Stages

| # | Script | What | Needs |
|---|---|---|---|
| 0 | `seed_from_pins.py` | seed `metadata.json` from `../paper_ids/dataset.csv` (**use this instead of 1-3** to reproduce our exact dataset); `--split train\|val`, `--limit N` | - |
| 1 | `scrape.py` | OpenReview metadata, spotlight/oral | - (public API) |
| 2 | `scrape_all_accepted.py` | **all** accepted papers incl. posters | - |
| 3 | `arxiv_link.py` | match to arXiv preprints, resolve `first_public_date` | `ANTHROPIC_API_KEY` (Haiku title matching) |
| 4 | `extract_agent.py` | LaTeX + PDF => markdown via a Claude agent | `ANTHROPIC_API_KEY`, Claude Agent SDK |
| 5 | `describe_figures_latex.py` | per-figure descriptions (GPT-5.2, `detail=high`) | `OPENAI_API_KEY` |
| 6 | `merge.py` | splice descriptions into the `[FIGURE N]` slots | - |
| 7 | `classify_papers.py` | theory/experiment balance + experimental domains | `ANTHROPIC_API_KEY` |
| 8 | `research_questions.py` | extract the RQs that form the task input | `ANTHROPIC_API_KEY` |

Stages 4-8 are run per venue on every paper with an `arxiv_id`.

## Running it

```bash
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...

# take 5 papers from the set used in the paper
uv run python seed_from_pins.py --limit 5             
uv run python run.py 4 5 6 7 8
uv run python run.py --status

# ...then the full set
uv run python seed_from_pins.py
uv run python run.py 4 5 6 7 8
```

### Seeding train vs validation papers

In the paper, we optimize prompts on the **poster** papers and report results on the held-out
**spotlight/oral** papers. With `--limit` alone, you get whatever comes first alphabetically, which is all posters. A smoke dataset built this way has no validation papers, so no papers would be selected in an eval filtered by `../paper_ids/val_spotlight_oral.txt`.

Use `--split` to pick a side, and seed both if you want to exercise the whole stack:

```bash
# a few of each: posters for the red team, spotlight/oral for the figures
uv run python seed_from_pins.py --split train --limit 5
uv run python seed_from_pins.py --split val   --limit 5
uv run python run.py 4 5 6 7 8
uv run python run.py --status
```

Seeding is additive and papers that are already present are skipped, so the results of the two commands accumulate and it is safe to re-run them. With `--split val`, you get both spotlight and oral papers. To narrow this down to a single tier, use `--decision`.

The corresponding id lists for downstream evals are `../paper_ids/train_posters.txt` and
`../paper_ids/val_spotlight_oral.txt`.

To rescrape from scratch instead of using the pinned IDs:

```bash
uv run python run.py 1 2 3        # metadata only, cheap
uv run python run.py 4 5 6 7 8
```

To run on a single venue, use `run_subset.py`, which also supports `--limit`:
```bash
uv run python run_subset.py --conf ICLR --year 2026 --stages 4 5 6 7 8 --limit 5
```