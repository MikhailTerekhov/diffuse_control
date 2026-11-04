"""Batched single-turn rollout strategy.

All envs in a GRPO group share the same prompt, so we batch them into
one sample_async(num_samples=N) call for ~1.4x speedup on Tinker sampling.
"""

import asyncio
import time
from dataclasses import dataclass

import tinker

from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.rollout_strategy import RolloutResult, RolloutStrategy
from tinker_cookbook.rl.types import (
    ActionExtra,
    EnvGroupBuilder,
    Trajectory,
    Transition,
)
from tinker_cookbook.utils import logtree, trace

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchedSingleTurn(RolloutStrategy):
    """Batched sampling for single-turn environments.

    Instead of N separate sample_async(num_samples=1) calls, fires one
    sample_async(num_samples=N) call since all envs in a GRPO group
    share the same prompt.
    """

    @property
    def catches_group_errors(self) -> bool:
        return True

    async def execute(
        self,
        env_group_builder: EnvGroupBuilder,
        policy: TokenCompleter,
    ) -> RolloutResult:
        assert isinstance(policy, TinkerTokenCompleter), (
            "BatchedSingleTurn requires TinkerTokenCompleter"
        )

        # Create envs
        envs = await env_group_builder.make_envs()
        n = len(envs)

        # Get initial observations (all should be identical prompts)
        async with trace.scope_span("env_initial_observation"):
            obs_and_stops = await asyncio.gather(
                *[env.initial_observation() for env in envs]
            )
        prompt = obs_and_stops[0][0]
        stop_condition = obs_and_stops[0][1]

        # Batched sampling - progress is tracked by env.initial_observation/step
        # 30-min timeout to detect hung Tinker connections (CLOSE-WAIT)
        SAMPLING_TIMEOUT = 2400  # 40 min - per-group sampling max observed is ~30 min
        async with trace.scope_span("policy_sample_batched"):
            t0 = time.time()
            logger.info(f"Tinker sample_async start: n={n} prompt_len={prompt.length} max_tokens={policy.max_tokens}")
            sample_result = await asyncio.wait_for(
                policy.sampling_client.sample_async(
                    prompt=prompt,
                    num_samples=n,
                    sampling_params=tinker.SamplingParams(
                        stop=stop_condition,
                        max_tokens=policy.max_tokens,
                        temperature=policy.temperature,
                    ),
                ),
                timeout=SAMPLING_TIMEOUT,
            )

        elapsed = time.time() - t0
        logger.info(f"Tinker sample_async done: n={n} elapsed={elapsed:.1f}s tokens={[len(s.tokens) for s in sample_result.sequences]}")

        with logtree.scope_header(f"Batched Sampling ({n} samples)"):
            logtree.table_from_dict({
                "num_samples": n,
                "prompt_tokens": prompt.length,
                "sample_time": f"{time.time() - t0:.1f}s",
            }, caption="Batch stats")

        # Step each env with its sample (concurrent judge calls)
        # Progress is tracked by env.step() via sampling_done/finish_judging
        async def finish_rollout(idx, env, seq, ob):
            async with trace.scope_span("env_step"):
                step_result = await env.step(
                    seq.tokens,
                    extra=ActionExtra(stop_reason=seq.stop_reason),
                )

            ac = TokensWithLogprobs(
                tokens=seq.tokens,
                maybe_logprobs=seq.logprobs,
                stop_reason=seq.stop_reason,
            )
            transition = Transition(
                ob=ob,
                ac=ac,
                reward=step_result.reward,
                episode_done=step_result.episode_done,
                metrics=step_result.metrics,
                logs=step_result.logs,
            )

            with logtree.scope_header(f"Rollout {idx}"):
                logtree.table_from_dict({
                    "tokens": len(seq.tokens),
                    "reward": f"{step_result.reward:.3f}",
                    **{k: f"{v:.3f}" if isinstance(v, float) else str(v)
                       for k, v in step_result.metrics.items()},
                }, caption=f"Rollout {idx} results")

            return Trajectory(
                transitions=[transition],
                final_ob=step_result.next_observation,
            )

        async with trace.scope_span("env_step_all"):
            trajectories = list(await asyncio.gather(*[
                finish_rollout(i, env, seq, obs_and_stops[i][0])
                for i, (env, seq) in enumerate(zip(envs, sample_result.sequences))
            ]))

        return RolloutResult(trajectories=trajectories, envs=envs, errors=[])
