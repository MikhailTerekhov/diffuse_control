# RL red team

Here we fine-tune the generator with RL instead of optimizing its prompt, to check whether the prompt optimization results survive stronger optimization pressure.

In the rest of the repo we use prompt optimization, because the frontier generator is only available through sampling. With an open-weight generator, we can test whether the flat top of the admissible region is an artifact of a weak search.

## How it works

GRPO (unclipped, no KL) with LoRA on GPT-OSS-120B via the Tinker API. The reward is a
scalarization of the two scores,

    r = alpha * w + beta * (1 - g)

so by sweeping alpha and beta, we trace out the convex hull of the subversion frontier. The shape of the reward is set by `reward_mode`: `max_weak` for weak score only, `additive` with `w1`/`w2` for the sweep,
`max_strong`, `weak_plus_strong`, `gated`.

Training is asynchronous: rollouts are generated and judged by a pool of workers while training proceeds on completed batches, since sampling times vary widely. A batch may therefore
contain rollouts from several weight snapshots, corrected by the importance-sampling ratio
in the loss; anything more than `max_steps_off_policy` snapshots behind is discarded.

## Running it

```bash
export TINKER_API_KEY=... ANTHROPIC_API_KEY=...

# the dataset split used here
uv run python -m rl_red_team.generate_split --output rl_red_team/split.json

# maximize the weak score from a prompt anchor, against the default weak scorer
uv run python -m rl_red_team.train_async \
    reward_mode=max_weak max_steps=700 \
    system_prompt_path=rl_red_team/anchor_prompts/naive_pf3.txt

# the same, against the blue team's robustified scorer
uv run python -m rl_red_team.train_async \
    reward_mode=max_weak max_steps=700 \
    system_prompt_path=rl_red_team/anchor_prompts/naive_pf3.txt \
    weak_prompt_path=blue_team/prompts/b_star.json

# a point on the frontier sweep instead of pure max-weak
uv run python -m rl_red_team.train_async reward_mode=additive w1=1.0 w2=3.0

# watch it
uv run python -m rl_red_team.monitor
```

Configuration is passed as `chz`-style `key=value` arguments on the command line. All fields are documented in `AsyncCLIConfig` in `train_async.py`, with the paper's hyperparameters as defaults: LoRA rank 32,
unclipped GRPO, KL 0, lr 1e-5, 16 tasks per batch, 8 rollouts per task, 2 epochs per batch,
temperature 1.0, `max_steps_off_policy` 2.

The outputs of a run are stored in `rl_red_team/runs/<name>/`: `metrics.jsonl` with per-step metrics (used for the trajectory figures), `checkpoints.jsonl` with sampler paths (used for the validation eval), and the rollouts and logs.

## Anchoring

In `anchor_prompts/`, you can find generator prompts taken from points on a prompt optimization frontier. If RL starts from one of these rather than from the default prompt, training begins where prompt optimization left off, so we can test whether RL pushes further rather than rediscovering the same behaviour. You can pass one as `system_prompt_path`.

Note that the token cap matters: generations with these prompts are long, and truncating them distorts the starting position. For this reason, the default `max_tokens` is 16384.

## Figures

| Directory | Contents |
|---|---|
| `figures/rl_trajectories/` | weak and proxy score over training, per weak scorer |
| `figures/rl_admissible_region/` | validation-scored plane: RL beside prompt optimization |

## Layout

| File | Contents |
|---|---|
| `train_async.py` | entry point: async producer/consumer training loop |
| `env.py` | one rollout: generate proposals, score, reward |
| `judges.py` | the weak and ground-truth-proxy judges, and reward scalarization |
| `data.py` | dataset samples for the RL dataset |
| `strategy.py` | the batched single-turn rollout strategy |
| `generate_split.py` | the train/val/test split used here |
| `progress.py`, `monitor.py`, `status.py` | progress files and live views |
| `anchor_prompts/` | frontier prompts to anchor runs on |
| `system_prompt.txt` | the developer prompt used during training |
