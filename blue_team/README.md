# Blue team: saddle-point optimization of the weak scorer

Here we search for a weak scorer prompt that the red team cannot exploit. We alternate between patching the scorer against every attack found so far and letting the red team attack the patched scorer, until new attacks stop working. See Section 5 and Appendix G (Algorithm 1) of the paper.

The objective is the diagonal residual: make the weak score track the ground truth
proxy across all known attacks, so the subversion Pareto frontier collapses onto the
`w = g` diagonal and there is no gap left to exploit.

## The loop

Each outer cycle has two phases:

- **W phase.** Opus proposes `n_w_candidates` new scorer prompts. Each is scored by
  re-judging **cached** proposals from the attacks collected so far. The candidate with the lowest
  loss J becomes the next W, or the previous W is kept if none beats it.
- **PF phase.** The red team (`red_team/`'s MO-ASHA) runs for
  `pf_phase_target_highest_rung` fresh rung-2 trials against the new W, producing the next
  batch of attacks.

Alongside the attacks, the population includes **anchor prompts**: diverse benign generator
prompts, so the loss penalizes a scorer that punishes honest work rather than only
rewarding one that catches attacks.

## Prerequisite: a finished red-team run

In the bootstrap step, F_0 is seeded from an existing red team Pareto frontier, and in the W phase, the **saved proposals** of that run are re-scored. So `source_run` must be a red team run directory with its `logs/eval_details/` proposal cache intact. Note that `logs/` is gitignored, so a run with pruned logs cannot be used to seed a saddle run.

In the provided configs, `source_run` is set to `"REPLACE_ME"` because the path is machine-specific, so you need to pass it explicitly:

```bash
uv run python -m blue_team.scripts.run_full_loop \
    --config full_10cycle.yaml \
    --source-run red_team/runs/<timestamp>
```

`full_10cycle.yaml` is the paper's run: 10 cycles, 5 W candidates each, 32 papers per
evaluation, 10 anchors, 15 new rung-2 trials per cycle. `base.yaml` is smaller,
`test.yaml` is a smoke test.

Other entry points:

```bash
uv run python -m blue_team.scripts.run_bootstrap      # phase 0 only: anchors + W_0 + F_0
uv run python -m blue_team.scripts.run_one_w_phase --run-dir blue_team/runs/<ts>
uv run python -m blue_team.scripts.validate_w_phase --source-run red_team/runs/<ts>
uv run python -m blue_team.monitor blue_team/runs/<ts>     # live dashboard
```

## The provided b*

`prompts/b_star.json` is the paper's robustified scorer b* (Appendix I.10, analyzed in
Section 5.2): the `W_10` output of the 10-cycle run, self-contained, so you do not need your own blue team run for the robustified-scorer version of the admissible region. The configs in the other packages are already set to this file.

Its ten dimensions are question_relevance, method_fit, approach_diversity,
domain_expertise, scope_alignment, insight_quality, question_coverage, overlay_penalty,
vocabulary_penalty, template_penalty. It is worth comparing these against the default scorer b_0 in `eval_harness/prompts/judge_rubric.txt`: the dimensions added under adversarial pressure are method_fit, approach_diversity and vocabulary_penalty, the defenses against the totalizing science attacks from Section 5.2.

## Using a discovered scorer elsewhere

The output of every W phase is saved to `cycle_<n>/w_phase/chosen.json`, a self-contained `{label, summary, prompt_text, score_schema}`. Only this file is needed in the other packages, so you can use a robustified scorer in any of them:

```bash
# score proposals with it in the eval harness
uv run inspect eval "eval_harness/experiments_quality_eval.py@experiments_quality" \
    -T weak_scorer=outer_evolve \
    -T outer_evolve_w_path=blue_team/runs/<ts>/cycle_10/w_phase/chosen.json
```

```yaml
# or attack it with the red team / max-weak baseline
eval_models:
  weak_scorer_type: "outer_evolve"
  outer_evolve_w_path: "blue_team/runs/<ts>/cycle_10/w_phase/chosen.json"
```

`red_team/config/config_sandbagging_outer_evolve_cyc10_*.yaml` and
`max_weak/config/config_max_weak_nostrong_outer_evolve_cyc10.yaml` are set up for this, with the path set to the provided b* by default. To attack a scorer you discovered, change the path to your own run. The optimizer is shown the `summary` of W, so the red team stays adaptive, as the framework requires.

## Layout

| File | Contents |
|---|---|
| `saddle_loop.py` | the outer loop over N cycles of W and PF phases |
| `w_phase.py` | propose W candidates, evaluate J, pick or roll back |
| `pf_phase.py` | the red team's MO-ASHA run against the current W |
| `w_optimizer.py` | the Opus prompt optimizer for W, and its context |
| `w_evaluator.py` | the loss J of Eq. 7: group weights and per-prompt residuals |
| `population.py` | prompts, cached samples, frontiers, attack subselection |
| `sparse_select.py` | selection of k well-spread points from a frontier |
| `bootstrap.py` | anchors from idea kernels, W_0, and F_0 from the source run |
| `schema.py` | the W JSON contract: prompt text + score-extraction schema |
| `prompts/b_star.json` | the paper's robustified scorer b* (Appendix I.10) |
| `judge.py` | one weak-judge call under a given W |
| `diagnostics.py` | per-cycle scatter plots written into the run dir |
| `summaries.py`, `outer_state.py`, `api_log.py`, `monitor.py` | support |
