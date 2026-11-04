# Admissible region, RL beside prompt optimization

Every point here is validation-scored on the same papers, with 95% bootstrap CIs on both
axes, so RL checkpoints and prompt-optimization frontier prompts can be compared directly.
That is the difference from `figures/rl_trajectories/`, where the trajectories are on-policy
training scores.

One panel per weak scorer. The question is whether RL reaches territory prompt optimization
could not, and whether that changes when the weak scorer is robustified.

## Steps

```bash
# 1. Fix the shared paper subset (cached, so both evals use the same papers)
uv run python figures/rl_admissible_region/val_subset.py

# 2. RL checkpoints. label=run_dir:step, resolved via that run's checkpoints.jsonl
uv run python figures/rl_admissible_region/eval_adapters.py --group naive \
    --checkpoints "GRPO 1:1=grpo_sw1-1:30" "max-weak=grpo_maxweak:120"

#    for runs trained against a robustified scorer, score with that same scorer
uv run python figures/rl_admissible_region/eval_adapters.py --group robust \
    --weak-prompt blue_team/prompts/b_star.json \
    --checkpoints "GRPO 1:1=grpo_bstar_sw1-1:30"

# 3. Prompt-optimization frontiers, from red team runs
uv run python figures/rl_admissible_region/eval_promptopt.py --group naive \
    --run red_team/runs/<sandbagging-run>
uv run python figures/rl_admissible_region/eval_promptopt.py --group robust \
    --run red_team/runs/<outer-evolve-run>

# 4. Plot. Any input may be omitted; an empty panel is dropped
uv run python figures/rl_admissible_region/plot.py \
    --naive-adapters  figures/rl_admissible_region/data/val_adapters_naive.json \
    --naive-promptopt figures/rl_admissible_region/data/val_promptopt_naive.json \
    --robust-adapters  figures/rl_admissible_region/data/val_adapters_robust.json \
    --robust-promptopt figures/rl_admissible_region/data/val_promptopt_robust.json \
    --robust-label "b* rubric"
```

For a cheap smoke test, pass `--limit N` to either eval. You can shrink the subset itself with `N=32 uv run python val_subset.py --refresh`.