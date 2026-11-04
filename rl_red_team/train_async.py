"""Custom async training loop for RL fine-tuning with Tinker.

Replaces the cookbook's train.main() with:
- N async train workers continuously generating rollouts
- 1 val worker processing evaluations from a queue
- Main loop collecting results, training with multi-epoch CISPO, updating sampler
- Clean failure handling and decoupled validation
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import chz
import tinker
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.metrics import compute_kl_sample_train, incorporate_kl_penalty
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.train import train_step
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log, trace

from rl_red_team.data import PaperSample, load_samples
from rl_red_team.env import (
    SandbagEnvGroupBuilder,
    _get_renderer_cached,
)
from rl_red_team.progress import init_progress
from rl_red_team.strategy import BatchedSingleTurn

logger = logging.getLogger(__name__)


@chz.chz
class AsyncCLIConfig:
    # Model
    model_name: str = "openai/gpt-oss-120b"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Reward weights / shaping
    w1: float = 1.0
    w2: float = 1.0
    reward_mode: str = "additive"  # additive, multiplicative, gated, max_weak, max_strong, weak_plus_strong
    w_floor: float = 0.80          # gated: keep weak >= w_floor
    gate_lambda: float = 6.0       # gated: penalty slope below the floor

    # Data
    split_file: str = "rl_red_team/split.json"
    system_prompt_path: str = "rl_red_team/system_prompt.txt"
    weak_prompt_path: str | None = None  # optional outer_evolve WPrompt json for an alt weak rubric

    # Training
    train_group_size: int = 8
    val_group_size: int = 1
    groups_per_batch: int = 16
    num_epochs: int = 2
    learning_rate: float = 1e-5
    max_tokens: int = 16384
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    loss_fn: LossFnType = "importance_sampling"  # GRPO (group-centered advantages, no clipping)
    clip_low: float = 0.8
    clip_high: float = 1.2
    seed: int = 0

    # Async
    num_workers: int = 16
    max_steps_off_policy: int = 2

    # Eval / checkpointing
    eval_every: int = 10
    save_every: int = 10
    max_steps: int = 300
    val_limit: int = 64
    # Early stop on collapse: if mean train weak_score stays below this for
    # `early_stop_patience` consecutive steps, stop (0 disables).
    early_stop_weak: float = 0.0
    early_stop_patience: int = 2

    # Logging
    log_path: str | None = None
    wandb_project: str = "fuzzy-sandbagging-rl"
    wandb_name: str | None = None

    # Judging
    judge_timeout: int = 1800

    # Task
    task: str = "quality"  # "quality" or "debug"

    # Service
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _make_builder_iter(builders: list, seed: int = 42):
    """Infinite iterator over builders. Each epoch reshuffles differently."""
    epoch = 0
    while True:
        rng = random.Random(seed + epoch)
        shuffled = list(builders)
        rng.shuffle(shuffled)
        yield from shuffled
        epoch += 1


def _make_builders(
    samples: list[PaperSample],
    model_name: str,
    renderer_name: str,
    system_prompt: str,
    w1: float,
    w2: float,
    group_size: int,
    is_eval: bool = False,
    debug_mode: bool = False,
    reward_mode: str = "additive",
    w_floor: float = 0.80,
    gate_lambda: float = 6.0,
) -> list[SandbagEnvGroupBuilder]:
    return [
        SandbagEnvGroupBuilder(
            sample=s,
            model_name=model_name,
            renderer_name=renderer_name,
            system_prompt=system_prompt,
            w1=w1,
            w2=w2,
            group_size=group_size,
            is_eval=is_eval,
            debug_mode=debug_mode,
            reward_mode=reward_mode,
            w_floor=w_floor,
            gate_lambda=gate_lambda,
        )
        for s in samples
    ]


async def async_main(config: AsyncCLIConfig):
    # ── Resolve renderer ──────────────────────────────────────────────────
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=config.model_name,
        explicit_renderer_name=config.renderer_name,
        load_checkpoint_path=config.load_checkpoint_path,
        base_url=config.base_url,
    )

    if config.task == "debug":
        system_prompt = (
            "You are a witty comedian. Given a research topic, respond with "
            "exactly one pun or joke about it. Be concise — one sentence only."
        )
        max_tokens = 128
    else:
        system_prompt = Path(config.system_prompt_path).read_text().strip()
        max_tokens = config.max_tokens

    # ── Run naming ────────────────────────────────────────────────────────
    model_slug = config.model_name.split("/")[-1]
    prefix = {"debug": "debug-", "quality": "sandbag-"}[config.task]
    if config.reward_mode == "gated":
        reward_tag = f"-gated_wf{config.w_floor}_l{config.gate_lambda}"
    elif config.reward_mode == "multiplicative":
        reward_tag = "-mult"
    elif config.reward_mode in ("max_weak", "max_strong"):
        reward_tag = f"-{config.reward_mode}"
    else:
        reward_tag = f"-add_w1_{config.w1}_w2_{config.w2}"
    run_name = (
        f"{prefix}{model_slug}"
        f"{reward_tag}"
        f"-lr{config.learning_rate}"
        f"-g{config.train_group_size}"
        f"-e{config.num_epochs}"
        f"-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    log_path = config.log_path or f"rl_red_team/runs/{run_name}"
    wandb_name = config.wandb_name or run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)
    Path(log_path).mkdir(parents=True, exist_ok=True)

    # ── Progress tracker ──────────────────────────────────────────────────
    progress = init_progress(Path(log_path) / "progress")
    progress.set_run_info(
        max_steps=config.max_steps,
        groups_per_batch=config.groups_per_batch,
        group_size=config.train_group_size,
        eval_every=config.eval_every,
    )

    # ── Judge timeout ─────────────────────────────────────────────────────
    import rl_red_team.judges as judges_module

    judges_module.JUDGE_TIMEOUT = config.judge_timeout
    if config.weak_prompt_path:
        judges_module.load_weak_wprompt(config.weak_prompt_path)

    # ── Debug logging (unbuffered - survives hangs/kills) ────────────────
    debug_handler = logging.FileHandler(f"{log_path}/debug.log", mode="a")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s")
    )
    # Force line-buffered so every log line hits disk immediately
    debug_handler.stream.reconfigure(line_buffering=True)
    for name in ("rl_red_team.env", "rl_red_team.judges", "rl_red_team.strategy",
                  "rl_red_team.train_async", "__main__"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(debug_handler)
        # Keep propagate=True for __main__ so training progress shows on stdout
        if name != "__main__":
            lg.propagate = False

    # ── ML logging ────────────────────────────────────────────────────────
    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        wandb_name=wandb_name,
    )
    ml_logger.log_hparams(config)

    # ── wandb x-axis configuration ───────────────────────────────────────
    # Per-rollout and per-group metrics are logged at higher frequency than
    # per-step metrics. define_metric sets separate x-axes so each panel
    # gets the right counter. We stop using step= in wandb.log to avoid
    # global step monotonicity conflicts.
    try:
        import wandb as _wandb

        if _wandb.run is not None:
            _wandb.define_metric("rollout/count")
            _wandb.define_metric("rollout/*", step_metric="rollout/count")
            _wandb.define_metric("group/count")
            _wandb.define_metric("group/*", step_metric="group/count")
            _wandb.define_metric("progress/step")
            _wandb.define_metric("*", step_metric="progress/step")
    except ImportError:
        _wandb = None  # type: ignore[assignment]

    # ── Tracing (Perfetto) ────────────────────────────────────────────────
    # Instruments Tinker SDK (forward_backward_async, optim_step_async,
    # sample_async) automatically. Flushes to JSONL every 1s via background
    # thread + atexit handler, so traces survive kills.
    trace_path = str(Path(log_path) / "trace_events.jsonl")
    timing_spans_path = Path(log_path) / "timing_spans.jsonl"
    trace.trace_init(output_file=trace_path)
    current_task = asyncio.current_task()
    if current_task is not None:
        current_task.set_name("main")
    logger.info(f"Tracing enabled → {trace_path}")

    split_file = config.split_file

    # ── Load data + build env builders ───────────────────────────────────
    train_samples_q = load_samples(split_file, split="train")
    val_samples_q = load_samples(split_file, split="val")
    if config.task == "debug":
        train_samples_q = [
            PaperSample(s.paper_id, s.rq_text[:100], s.paper_text[:100])
            for s in train_samples_q
        ]
        val_samples_q = [
            PaperSample(s.paper_id, s.rq_text[:100], s.paper_text[:100])
            for s in val_samples_q
        ]
    train_builders = _make_builders(
        train_samples_q, config.model_name, renderer_name, system_prompt,
        config.w1, config.w2, config.train_group_size,
        debug_mode=(config.task == "debug"),
        reward_mode=config.reward_mode, w_floor=config.w_floor,
        gate_lambda=config.gate_lambda,
    )
    val_builders = _make_builders(
        val_samples_q[:config.val_limit], config.model_name, renderer_name,
        system_prompt, config.w1, config.w2, config.val_group_size,
        is_eval=True, debug_mode=(config.task == "debug"),
        reward_mode=config.reward_mode, w_floor=config.w_floor,
        gate_lambda=config.gate_lambda,
    )
    train_iter = _make_builder_iter(train_builders, seed=config.seed)

    # ── Tinker setup ──────────────────────────────────────────────────────
    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Stamp the RESOLVED renderer into training-run metadata so future resumes
    # inherit it (resolve_renderer_name_from_checkpoint_or_default reads this key).
    # We stamp the resolved name, not config.renderer_name, because the latter is
    # often None (defaulted) - without this, resuming a run whose renderer was an
    # explicit non-default silently reverts to the model default on the next leg.
    user_metadata: dict[str, str] = {}
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, renderer_name)

    if config.load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            path=config.load_checkpoint_path,
            user_metadata=user_metadata,
        )
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
            user_metadata=user_metadata,
        )

    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    sampling_step = 0

    # Pre-warm renderer/tokenizer cache (blocks event loop once, avoids
    # synchronous HuggingFace download later when workers start)
    _get_renderer_cached(config.model_name, renderer_name)

    # KL reference (frozen base model for regularization)
    kl_reference_client = None
    if config.kl_penalty_coef > 0:
        kl_reference_client = await service_client.create_sampling_client_async(
            base_model=config.model_name,
        )

    # ── Rollout strategy ──────────────────────────────────────────────────
    strategy = BatchedSingleTurn()

    # ── Queues ────────────────────────────────────────────────────────────
    # Bounded so producers block (backpressure) when the trainer falls behind,
    # instead of accumulating multi-MB TrajectoryGroups without limit (OOM).
    result_queue: asyncio.Queue[tuple[TrajectoryGroup | None, int]] = asyncio.Queue(
        maxsize=2 * config.groups_per_batch
    )
    val_queue: asyncio.Queue[tuple[int, tinker.SamplingClient] | None] = asyncio.Queue()
    done = False

    # ── Train workers ─────────────────────────────────────────────────────
    # Each worker continuously grabs the next paper from train_iter, runs a
    # rollout, and pushes the result to result_queue. Workers see the latest
    # sampling_client via closure - when the main loop updates it, workers
    # pick up the new reference on their next iteration.

    async def train_worker(worker_id: int):
        while not done:
            builder = next(train_iter)
            try:
                policy = TinkerTokenCompleter(
                    sampling_client, max_tokens=max_tokens,
                    temperature=config.temperature,
                )
                async with trace.scope_span("train/rollout"):
                    group = await do_group_rollout(
                        env_group_builder=builder,
                        policy=policy,
                        strategy=strategy,
                    )
            except Exception as e:
                logger.warning(f"Worker {worker_id}: rollout failed: {e}")
                group = None
            await result_queue.put((group, sampling_step))  # blocks when full → backpressure

    workers = [
        asyncio.create_task(train_worker(i), name=f"train_worker_{i}")
        for i in range(config.num_workers)
    ]

    # ── Val worker ────────────────────────────────────────────────────────
    # Pulls (step, sampler) from val_queue, runs all val builders
    # concurrently, logs metrics directly to wandb/disk.

    async def _safe_val_rollout(builder, policy):
        try:
            async with trace.scope_span("val/rollout"):
                return await do_group_rollout(builder, policy, strategy=strategy)
        except Exception as e:
            logger.warning(f"Val rollout failed: {e}")
            return None

    async def val_worker_coro():
        while True:
            item = await val_queue.get()
            if item is None:
                break
            step, sampler = item

            logger.info(f"Val: starting evaluation at step {step}")
            t0 = time.time()

            async with trace.scope_span("val/eval"):
                policy = TinkerTokenCompleter(
                    sampler, max_tokens=max_tokens, temperature=config.temperature,
                )
                results = await asyncio.gather(*[
                    _safe_val_rollout(builder, policy)
                    for builder in val_builders
                ])
            groups = [g for g in results if g is not None]

            if groups:
                taglist = [[config.task] for _ in groups]
                val_metrics = compute_trajectory_metrics(groups, taglist)
                val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                val_metrics["val/time"] = time.time() - t0
                val_metrics["val/num_groups"] = len(groups)
                val_metrics["progress/step"] = step
                ml_logger.log_metrics(val_metrics)
                reward = val_metrics.get("val/env/all/reward/total", "N/A")
                logger.info(f"Val: step {step} done in {time.time() - t0:.0f}s, reward={reward}")
            else:
                logger.warning(f"Val: step {step} - all groups failed")

    val_task = asyncio.create_task(val_worker_coro(), name="val_worker")

    # ── Main training loop ────────────────────────────────────────────────
    # ── Per-rollout/per-group wandb helper ────────────────────────────────
    # Logs directly to wandb (bypasses JSON/console loggers - too noisy).
    def _log_wandb(metrics: dict) -> None:
        try:
            if _wandb is not None and _wandb.run is not None:
                _wandb.log(metrics)
        except Exception:
            pass

    rollout_count = 0
    group_count = 0
    consec_low_weak = 0  # consecutive steps with collapsed weak (for early stop)

    # ── Optional memory profiling (MEMPROF=1): per-step RSS + tracemalloc growth ──
    import os as _os
    _memprof = bool(_os.environ.get("MEMPROF"))
    _prev_snap = None
    if _memprof:
        import tracemalloc
        tracemalloc.start(20)

        def _rss_mb():
            with open("/proc/self/statm") as _f:
                return int(_f.read().split()[1]) * 4096 / 1e6

    # ── Rollout JSONL logger (full request/response pairs) ────────────────
    rollouts_path = Path(log_path) / "rollouts.jsonl"
    _rollouts_file = open(rollouts_path, "a")
    _rollout_tok = get_tokenizer(config.model_name)

    def _decode(toks) -> str:
        try:
            return _rollout_tok.decode(toks)
        except Exception as e:  # logging must never break training
            return f"<decode_error: {type(e).__name__}: {e}>"

    def _log_rollout(step: int, gen_step: int, group: TrajectoryGroup):
        """Append one JSON line per rollout with full logs."""
        for traj, total_reward in zip(
            group.trajectories_G, group.get_total_rewards()
        ):
            for t in traj.transitions:
                record = {
                    "step": step,
                    "gen_step": gen_step,
                    "total_reward": total_reward,
                    "metrics": t.metrics,
                    # Exact post-render tokens we train on, decoded to text:
                    #   prompt_text   = rendered Harmony input (incl. any system
                    #                   preamble like the reasoning header),
                    #   response_text = raw sampled tokens (incl. the reasoning
                    #                   channel, before parse_response strips it).
                    "prompt_text": _decode(t.ob.to_ints()),
                    "response_text": _decode(t.ac.tokens),
                    "logs": {
                        k: v for k, v in t.logs.items()
                        if isinstance(v, (str, int, float))
                    },
                }
                _rollouts_file.write(json.dumps(record) + "\n")
        _rollouts_file.flush()

    logger.info(
        f"Starting async training: {config.max_steps} steps, "
        f"{config.num_workers} workers, {config.num_epochs} epochs/step, "
        f"{config.groups_per_batch} groups/batch"
    )

    for step in range(config.max_steps):
        t_step = time.time()
        progress.start_step(step)
        logger.info(f"=== Step {step}/{config.max_steps} START ===")

        # Queue val if due (non-blocking, frozen sampler snapshot)
        if config.eval_every > 0 and step % config.eval_every == 0:
            val_queue.put_nowait((step, sampling_client))
            logger.info(f"Step {step}: queued validation")

        with trace.trace_iteration(step=step) as window:
            # ── Collect groups_per_batch results from workers ──
            async with trace.scope_span("collect_rollouts"):
                t_collect = time.time()
                logger.info(f"Step {step}: collecting {config.groups_per_batch} groups...")
                groups: list[TrajectoryGroup] = []
                stale_count = 0
                failed_count = 0
                while len(groups) < config.groups_per_batch:
                    group, gen_step = await result_queue.get()
                    if group is None:
                        failed_count += 1
                        logger.debug(f"Step {step}: got failed group ({len(groups)}/{config.groups_per_batch})")
                        continue
                    if step - gen_step > config.max_steps_off_policy:
                        stale_count += 1
                        logger.info(
                            f"Step {step}: discarding stale rollout "
                            f"(gen_step={gen_step}, staleness={step - gen_step})"
                        )
                        continue
                    groups.append(group)
                    _log_rollout(step, gen_step, group)
                    logger.debug(
                        f"Step {step}: collected group {len(groups)}/{config.groups_per_batch} "
                        f"(gen_step={gen_step}, rewards={[f'{r:.2f}' for r in group.get_total_rewards()]})"
                    )

                    # ── Per-group + per-rollout wandb logging ──
                    group_count += 1
                    rewards = group.get_total_rewards()
                    _log_wandb({
                        "group/count": group_count,
                        "group/mean_reward": sum(rewards) / len(rewards),
                        "group/num_rollouts": len(rewards),
                    })
                    for traj in group.trajectories_G:
                        for t in traj.transitions:
                            rollout_count += 1
                            entry: dict = {"rollout/count": rollout_count}
                            m = t.metrics
                            if "reward" in m:
                                entry["rollout/reward"] = m["reward"]
                            if "weak_score" in m:
                                entry["rollout/weak_score"] = m["weak_score"]
                            if "strong_score" in m:
                                entry["rollout/strong_score"] = m["strong_score"]
                            if "judge_parse_fail" in m:
                                entry["rollout/judge_parse_fail"] = m["judge_parse_fail"]
                            _log_wandb(entry)
                t_collect = time.time() - t_collect
                logger.info(
                    f"Step {step}: collected all groups in {t_collect:.1f}s "
                    f"(stale={stale_count}, failed={failed_count})"
                )

            # ── Prepare training data ──
            async with trace.scope_span("prepare_training_data"):
                logger.info(f"Step {step}: preparing training data ({len(groups)} groups)...")
                advantages = compute_advantages(groups)
                datums, _metadata = assemble_training_data(groups, advantages)
                logger.info(f"Step {step}: assembled {len(datums)} datums")

                # Drop any datum whose token sequence contains an out-of-vocab id.
                # gpt-oss occasionally samples a Harmony control token above the
                # model's trainable ceiling (200018); Tinker then rejects the whole
                # forward_backward batch (RequestFailedError "token_ids max ...
                # exceeds max token id 200018") and the run dies. Rare (~1 rollout /
                # few-hundred steps); dropping it just skips that trajectory's grad.
                _MAX_TOK = 200018
                _kept = [(d, md) for d, md in zip(datums, _metadata)
                         if max(d.model_input.to_ints(), default=0) <= _MAX_TOK]
                if len(_kept) < len(datums):
                    logger.warning(f"Step {step}: dropped {len(datums) - len(_kept)} "
                                   f"datum(s) with out-of-vocab token >{_MAX_TOK}")
                    datums = [d for d, _ in _kept]
                    _metadata = [md for _, md in _kept]

                kl_metrics: dict = {}
                if config.kl_penalty_coef > 0 and kl_reference_client is not None:
                    logger.info(f"Step {step}: computing KL penalty...")
                    t_kl = time.time()
                    async with trace.scope_span("kl_penalty"):
                        kl_metrics = await incorporate_kl_penalty(
                            datums,
                            kl_reference_client,
                            config.kl_penalty_coef,
                            kl_discount_factor=0.0,
                        )
                    logger.info(
                        f"Step {step}: KL penalty done in {time.time() - t_kl:.1f}s "
                        f"(kl={kl_metrics.get('kl_policy_base', 'N/A')})"
                    )

            # ── Multi-epoch training ──
            async with trace.scope_span("multi_epoch_train"):
                t_train = time.time()
                progress.set_phase("training")
                logger.info(f"Step {step}: training {config.num_epochs} epochs on {len(datums)} datums...")
                epoch_kl_metrics: dict = {}
                for epoch in range(config.num_epochs):
                    t_epoch = time.time()
                    random.shuffle(datums)
                    async with trace.scope_span(f"train_epoch_{epoch}"):
                        logger.debug(f"Step {step}: epoch {epoch} forward_backward START")
                        loss_fn_config = None
                        if config.loss_fn in ("cispo", "ppo"):
                            loss_fn_config = {
                                "clip_low_threshold": config.clip_low,
                                "clip_high_threshold": config.clip_high,
                            }
                        training_logprobs = await train_step(
                            data_D=datums,
                            training_client=training_client,
                            learning_rate=config.learning_rate,
                            num_substeps=1,
                            loss_fn=config.loss_fn,
                            loss_fn_config=loss_fn_config,
                        )
                        # Track per-epoch KL: how much did the model
                        # change from the sampling policy?
                        epoch_kl = compute_kl_sample_train(datums, training_logprobs)
                        kl_v1 = epoch_kl["optim/kl_sample_train_v1"]
                        kl_v2 = epoch_kl["optim/kl_sample_train_v2"]
                        entropy = epoch_kl["optim/entropy"]
                        logger.info(
                            f"Step {step}: epoch {epoch} done in {time.time() - t_epoch:.1f}s "
                            f"kl_v1={kl_v1:.4f} kl_v2={kl_v2:.4f} entropy={entropy:.4f}"
                        )
                        for k, v in epoch_kl.items():
                            epoch_kl_metrics[f"{k}/epoch_{epoch}"] = v
                t_train = time.time() - t_train
                logger.info(f"Step {step}: training done in {t_train:.1f}s ({config.num_epochs} epochs)")

            # ── Update sampler (workers pick up new reference via closure) ──
            async with trace.scope_span("update_sampler"):
                t_save = time.time()
                if config.save_every > 0 and step > 0 and step % config.save_every == 0:
                    logger.info(f"Step {step}: saving full checkpoint...")
                    path_dict = await checkpoint_utils.save_checkpoint_async(
                        training_client=training_client,
                        name=f"{step:06d}",
                        log_path=log_path,
                        loop_state={"batch": step},
                        kind="both",
                    )
                    sampling_client = await training_client.create_sampling_client_async(
                        path_dict["sampler_path"]
                    )
                    logger.info(f"Step {step}: checkpoint saved in {time.time() - t_save:.1f}s")
                else:
                    logger.debug(f"Step {step}: saving sampler weights...")
                    sampling_client = (
                        await training_client.save_weights_and_get_sampling_client_async()
                    )
                    logger.debug(f"Step {step}: sampler saved in {time.time() - t_save:.1f}s")
                sampling_step = step + 1

        # ── Log train metrics + timing spans ──
        # No step= parameter - wandb uses define_metric("*", step_metric=
        # "progress/step") so the x-axis comes from the dict, not the
        # global step counter. This avoids monotonicity conflicts with
        # the higher-frequency per-rollout/per-group logs.
        taglist = [[config.task] for _ in groups]
        train_metrics = compute_trajectory_metrics(groups, taglist)
        train_metrics.update(kl_metrics)
        train_metrics.update(epoch_kl_metrics)
        train_metrics.update(window.get_timing_metrics())
        train_metrics["time/total"] = time.time() - t_step
        train_metrics.update({
            "progress/step": step,
            "progress/sampling_step": sampling_step,
            "async/stale_discarded": stale_count,
            "async/failed_groups": failed_count,
            "async/rollout_count": rollout_count,
            "async/group_count": group_count,
        })
        ml_logger.log_metrics(train_metrics)
        window.write_spans_jsonl(timing_spans_path, step=step)
        progress.finish_step()

        # Free inspect-ai's global transcript. It records a ModelEvent (full
        # judge prompt + paper + proposals + response) for every generate()
        # call and never clears it outside an eval task → ~100MB/step RSS leak.
        from inspect_ai.log._transcript import transcript as _inspect_transcript
        _it = _inspect_transcript()
        _it._events.clear()
        _it._pending_events.clear()
        getattr(_it, "_model_call_counts", {}).clear()  # else grows per judge call
        getattr(_it, "_kept_event_ids", set()).clear()

        reward_total = train_metrics.get("env/all/reward/total", 0)
        logger.info(
            f"Step {step}: reward={reward_total:.3f} "
            f"collect={t_collect:.0f}s train={t_train:.0f}s "
            f"total={time.time() - t_step:.0f}s "
            f"stale={stale_count} failed={failed_count}"
        )

        if _memprof:
            import tracemalloc, gc
            gc.collect()
            snap = tracemalloc.take_snapshot()
            if _prev_snap is not None:
                top = snap.compare_to(_prev_snap, "lineno")[:10]
                logger.info(f"MEMPROF step {step}: RSS={_rss_mb():.0f}MB  top growths since last step:")
                for st in top:
                    logger.info(f"  {st.size_diff/1e6:+.1f}MB (cum {st.size/1e6:.1f}MB, {st.count_diff:+d} objs) "
                                f"{st.traceback.format()[-1].strip()}")
            else:
                logger.info(f"MEMPROF step {step}: RSS={_rss_mb():.0f}MB (baseline)")
            _prev_snap = snap

        # Early stop on collapse (mean train weak score floored out).
        if config.early_stop_weak > 0:
            weak_now = train_metrics.get("env/all/weak_score", 1.0)
            consec_low_weak = consec_low_weak + 1 if weak_now < config.early_stop_weak else 0
            if consec_low_weak >= config.early_stop_patience:
                logger.info(f"Early stop at step {step}: weak={weak_now:.3f} < "
                            f"{config.early_stop_weak} for {config.early_stop_patience} steps")
                break

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Training loop finished, shutting down...")
    done = True
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    # Signal val worker to stop, then wait for it to finish current eval
    val_queue.put_nowait(None)
    await val_task

    # Final checkpoint
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=log_path,
        loop_state={"batch": config.max_steps, "final": True},
        kind="both",
    )

    _rollouts_file.close()
    trace.trace_shutdown()
    ml_logger.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    cli_config = chz.entrypoint(AsyncCLIConfig)
    asyncio.run(async_main(cli_config))
