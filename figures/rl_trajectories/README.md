# RL training trajectories

Here we plot the ground truth proxy score while RL pushes the weak score up. If you run the same anchored setup against two different weak scorers, the contrast is the blue team result: against the default rubric, the two scores come apart, and against the robustified scorer, they move together, so there is no gap left to exploit.

Both plots are made from `rl_red_team/runs/<dir>/metrics.jsonl`, which is updated at each training step. No extra evaluation is needed, so you can make them while a run is still going.

## Two views of the same runs

```bash
# score against training step, three panels: weak per scorer, then the proxy
uv run python figures/rl_trajectories/plot_overlay.py \
    --naive  "anchor A=grpo_anchA_naiveW" "anchor B=grpo_anchB_naiveW" \
    --robust "anchor A=grpo_anchA_bstarW" "anchor B=grpo_anchB_bstarW" \
    --robust-label "b* rubric"

# the same runs as paths through the (proxy, weak) plane
uv run python figures/rl_trajectories/plot_plane.py \
    --naive-runs  "anchor A=grpo_anchA_naiveW" "anchor B=grpo_anchB_naiveW" \
    --robust-runs "anchor A=grpo_anchA_bstarW" "anchor B=grpo_anchB_bstarW" \
    --robust-label "b* rubric"
```

Runs are `label=dir` pairs, so the labels are yours to choose. A run that crashed and was
resumed is stitched onto a continuous step axis by joining its directories with a comma:
`--naive "anchor A=grpo_anchA,grpo_anchA_cont"`.

With `plot_plane.py`, you can optionally overlay a prompt optimization frontier for reference, computed with `figures/rl_admissible_region/eval_promptopt.py`:

```bash
    --naive-pf  figures/rl_admissible_region/data/val_promptopt_naive.json \
    --robust-pf figures/rl_admissible_region/data/val_promptopt_robust.json
```

Treat that overlay as a reference only: the trajectories are on-policy training scores while
the frontier points are validation-scored. For a like-for-like comparison use
`figures/rl_admissible_region/`.

## Reading them

The starting point is the anchor prompt's own position, so a trajectory that goes nowhere
means RL found nothing beyond what the prompt already did. The smoothing width is set with `--sigma`, and the raw trace stays visible underneath.
