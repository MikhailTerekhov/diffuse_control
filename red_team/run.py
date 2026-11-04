"""MO-ASHA v2 run orchestrator.

Usage:
    uv run python red_team/run.py
    uv run python red_team/run.py --config config_test.yaml
    uv run python red_team/run.py --dry-run
    uv run python red_team/run.py --resume runs/20260401_120000
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RED_TEAM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RED_TEAM_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RED_TEAM_DIR) not in sys.path:
    sys.path.insert(0, str(RED_TEAM_DIR))

import yaml

from red_team.scheduler import (
    MOASHAConfig,
    MOASHAScheduler,
    PromptTrial,
    SampleResult,
    TrialStatus,
    EvalWorkItem,
    GenerateWorkItem,
)
from red_team.islands import IslandConfig, IslandManager
from red_team.state import save_state, load_state
from red_team.monitor import MonitorState

logger = logging.getLogger(__name__)


def load_config(config_path: str | None) -> dict:
    cfg_dir = RED_TEAM_DIR / "config"
    base = yaml.safe_load((cfg_dir / "base.yaml").read_text())
    if config_path:
        p = Path(config_path)
        if not p.is_absolute():
            p = cfg_dir / p
        overlay = yaml.safe_load(p.read_text())
        _deep_merge(base, overlay)
    return base


def _deep_merge(base: dict, overlay: dict) -> dict:
    for k, v in overlay.items():
        # A key written with no value ("optimizer:") parses as None. Treat that as
        # "no override" rather than letting it blank out a populated base section.
        if v is None and base.get(k) is not None:
            continue
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def generate_split(config: dict, out_path: Path):
    from eval_harness.generate_split import generate_split as _generate_split
    _generate_split(
        train_ratio=config.get("split", {}).get("train_ratio", 0.7),
        seed=config.get("split", {}).get("seed", 42),
        output_path=out_path,
    )


def setup_logging(output_dir: Path):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_dir / "moasha_v2.log")],
    )
    try:
        logging.getLogger().addHandler(logging.StreamHandler())
    except Exception:
        pass
    http_handler = logging.FileHandler(log_dir / "http_requests.log")
    http_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    for noisy in ("httpx", "httpcore", "anthropic", "LiteLLM"):
        lg = logging.getLogger(noisy)
        lg.addHandler(http_handler)
        lg.propagate = False


def _init_api_logging():
    """Enable full API request/response logging via inspect's transcript system."""
    try:
        from inspect_ai.log._transcript import Transcript, init_transcript
        t = Transcript(log_model_api=True)
        init_transcript(t)
        logger.info("API logging enabled (log_model_api=True)")
        return t
    except Exception as e:
        logger.warning(f"Could not init API transcript logging: {e}")
        return None


def _flush_transcript(transcript, output_dir: Path, label: str = ""):
    """Flush transcript events to a JSONL file, resolving attachment:// references."""
    if not transcript or not transcript.events:
        return
    from inspect_ai.event._model import ModelEvent
    events_dir = output_dir / "logs" / "api_events"
    events_dir.mkdir(parents=True, exist_ok=True)

    # Only flush completed events (pending=None means complete)
    model_events = [e for e in transcript.events
                    if isinstance(e, ModelEvent) and e.pending is None]
    # Keep incomplete events for next flush
    pending = [e for e in transcript.events if not (isinstance(e, ModelEvent) and e.pending is None)]
    if not model_events:
        return

    attachments = transcript.attachments

    def resolve(text):
        """Replace attachment:// references with actual content."""
        if not isinstance(text, str) or "attachment://" not in text:
            return text
        for hash_key, content in attachments.items():
            text = text.replace(f"attachment://{hash_key}", content)
        return text

    def resolve_obj(obj):
        """Recursively resolve attachment references in dicts/lists/strings."""
        if isinstance(obj, str):
            return resolve(obj)
        if isinstance(obj, dict):
            return {k: resolve_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve_obj(v) for v in obj]
        return obj

    import time
    out = events_dir / f"events_{label}_{time.time():.0f}.jsonl"
    with open(out, "w") as f:
        for event in model_events:
            # Extract reasoning/thinking from content blocks
            reasoning_text = ""
            if event.output and event.output.choices:
                for block in event.output.choices[0].message.content:
                    btype = getattr(block, "type", "")
                    if btype in ("thinking", "reasoning"):
                        raw = getattr(block, "thinking", "") or getattr(block, "reasoning", "") or ""
                        reasoning_text += resolve(raw)

            entry = {
                "model": event.model,
                "role": event.role,
                "input": [{"role": getattr(m, "role", "?"),
                           "content": resolve(m.content) if hasattr(m, "content") else str(m)}
                          for m in event.input],
                "output_text": resolve(event.output.choices[0].message.text) if event.output.choices else None,
                "reasoning": reasoning_text or None,
                "stop_reason": event.output.stop_reason if event.output else None,
                "usage": {
                    "input_tokens": event.output.usage.input_tokens if event.output and event.output.usage else None,
                    "output_tokens": event.output.usage.output_tokens if event.output and event.output.usage else None,
                    "reasoning_tokens": event.output.usage.reasoning_tokens if event.output and event.output.usage else None,
                } if event.output else None,
                "error": event.error,
                "call": resolve_obj({
                    "request": event.call.request if event.call else None,
                    "response": event.call.response if event.call else None,
                    "time": event.call.time if event.call else None,
                }) if event.call else None,
            }
            f.write(json.dumps(entry, default=str) + "\n")

    # Keep only pending events, clear flushed ones
    transcript.events.clear()
    transcript.events.extend(pending)
    logger.info(f"Flushed {len(model_events)} API events to {out}")


def _load_seed_trial(path: Path) -> PromptTrial:
    data = json.loads(path.read_text())
    return PromptTrial(
        trial_id=data["trial_id"],
        prompt_text=data["prompt_text"],
        justification=data.get("justification", ""),
        parent_trial_id=data.get("parent_trial_id"),
        generation=data.get("generation", -1),
        island_id=data.get("island_id", 0),
        current_rung=data["current_rung"],
        sample_results=[SampleResult(**sr) for sr in data["sample_results"]],
        rung_scores={int(k): tuple(v) for k, v in data["rung_scores"].items()},
        status=TrialStatus(data["status"]),
        created_at=data.get("created_at", 0),
    )


def _truncated_rung_scores(samples, rung_sizes):
    out = {}
    for i, n in enumerate(rung_sizes):
        sub = [s for s in samples[:n]
               if s.weak_score is not None and s.strong_score is not None]
        if not sub:
            continue
        w = sum(s.weak_score for s in sub) / len(sub)
        st = sum(s.strong_score for s in sub) / len(sub)
        out[i] = (w, st, len(sub))
    return out


async def _seed_evaluate(
    prompt_path: Path,
    label: str,
    output_dir: Path,
    monitor,
    eval_fn,
    dataset: list[dict],
    rung_samples: list[int],
    api_semaphore,
    log_dir: Path | None,
    island_id: int = 0,
) -> PromptTrial:
    """Evaluate a raw prompt file in-process and return an injected-ready PromptTrial.

    Mirrors the evaluator's per-sample pattern + reports per-sample progress to
    the monitor via set_seeding so the standalone monitor shows live progress.
    """
    import uuid
    import random as _random
    import time as _time

    prompt_text = prompt_path.read_text().strip()
    n_samples = rung_samples[-1]
    seed_dir = output_dir / "seeded"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_log_dir = seed_dir / f"{label}_eval_details"
    trial_id = uuid.uuid4().hex[:12]

    rng = _random.Random(42)
    sampled = rng.sample(dataset, min(n_samples, len(dataset)))

    monitor.set_seeding({
        "label": label,
        "trial_id": trial_id,
        "done": 0,
        "total": len(sampled),
        "failed": 0,
        "status": "running",
        "started_at": _time.time(),
    })
    logger.info(f"Seeding [{label}] from {prompt_path.name}: "
                f"evaluating {len(sampled)} samples (trial_id={trial_id[:8]})")

    results: list[SampleResult] = []
    failed = 0
    done = 0

    async def _one(sample, worker_key):
        nonlocal done, failed
        monitor.start_sample(worker_key, trial_id, rung=len(rung_samples) - 1)
        try:
            r = await eval_fn(
                trial_id=trial_id,
                prompt_text=prompt_text,
                sample=sample,
                semaphore=api_semaphore,
                log_dir=seed_log_dir if log_dir is not None else None,
            )
            return r
        except Exception as e:
            logger.warning(f"Seed sample {sample['paper_id']} failed: {e}")
            return SampleResult(paper_id=sample["paper_id"])
        finally:
            monitor.finish_sample(worker_key)

    tasks = [asyncio.create_task(_one(s, f"seed_w{i}"))
             for i, s in enumerate(sampled)]
    for fut in asyncio.as_completed(tasks):
        r = await fut
        results.append(r)
        done += 1
        if r.weak_score is None or r.strong_score is None:
            failed += 1
        # Running aggregate
        valid = [x for x in results
                 if x.weak_score is not None and x.strong_score is not None]
        running_w = (sum(x.weak_score for x in valid) / len(valid)) if valid else None
        running_s = (sum(x.strong_score for x in valid) / len(valid)) if valid else None
        monitor.set_seeding({
            "label": label,
            "trial_id": trial_id,
            "done": done,
            "total": len(sampled),
            "failed": failed,
            "status": "running",
            "running_mean_weak": running_w,
            "running_mean_strong": running_s,
        })

    rung_scores = _truncated_rung_scores(results, rung_samples)
    if not rung_scores:
        raise RuntimeError(f"Seeding [{label}]: no valid (weak, strong) sample pairs")

    target_rung = max(rung_scores.keys())
    w, st, n = rung_scores[target_rung]
    logger.info(f"Seeding [{label}] complete: rung {target_rung} "
                f"weak={w:.4f} strong={st:.4f} (n={n}, failed={failed})")

    trial = PromptTrial(
        trial_id=trial_id,
        prompt_text=prompt_text,
        justification=f"seeded from {prompt_path.name}",
        parent_trial_id=None,
        generation=-1,
        island_id=island_id,
        current_rung=target_rung,
        sample_results=results,
        rung_scores=rung_scores,
        status=TrialStatus.FINISHED if target_rung == len(rung_samples) - 1
               else TrialStatus.STOPPED,
    )

    # Persist the trial JSON so subsequent resumes can use it without re-evaluating.
    trial_json = {
        "trial_id": trial.trial_id,
        "prompt_text": trial.prompt_text,
        "justification": trial.justification,
        "parent_trial_id": trial.parent_trial_id,
        "generation": trial.generation,
        "island_id": trial.island_id,
        "current_rung": trial.current_rung,
        "sample_results": [
            {"paper_id": r.paper_id, "weak_score": r.weak_score,
             "strong_score": r.strong_score}
            for r in trial.sample_results
        ],
        "rung_scores": {str(k): list(v) for k, v in trial.rung_scores.items()},
        "status": trial.status.value,
        "_seed_label": label,
    }
    trial_path = seed_dir / f"{label}.json"
    trial_path.write_text(json.dumps(trial_json, indent=2, default=str))
    logger.info(f"Saved seed trial JSON to {trial_path}")

    monitor.set_seeding({
        "label": label, "trial_id": trial_id,
        "done": done, "total": len(sampled), "failed": failed,
        "status": "completed",
        "running_mean_weak": w, "running_mean_strong": st,
    })
    return trial


async def main_loop(config: dict, output_dir: Path, dry_run: bool = False,
                    resume_from: Path | None = None,
                    seed_prompt_paths: list[Path] | None = None):
    transcript = _init_api_logging() if not dry_run else None
    moasha_dict = config["moasha"]
    # Filter to only MOASHAConfig fields
    import dataclasses
    moasha_fields = {f.name for f in dataclasses.fields(MOASHAConfig)}
    moasha_cfg = MOASHAConfig(**{k: v for k, v in moasha_dict.items() if k in moasha_fields})
    island_cfg = IslandConfig(
        n_islands=moasha_cfg.n_islands,
        feature_bins=moasha_cfg.feature_bins,
        mode=moasha_cfg.mode,
        migration_interval=moasha_dict.get("migration_interval", 640),
        migration_rate=moasha_dict.get("migration_rate", 0.1),
        population_size=moasha_dict.get("population_size", 80),
        archive_size=moasha_dict.get("archive_size", 16),
        exploration_ratio=moasha_dict.get("exploration_ratio", 0.3),
        exploitation_ratio=moasha_dict.get("exploitation_ratio", 0.6),
    )
    islands = IslandManager(island_cfg, len(moasha_cfg.rung_samples))
    progress_dir = output_dir / "progress"
    monitor = MonitorState(progress_dir)
    log_dir = output_dir / "logs" / "eval_details"

    if resume_from:
        scheduler = load_state(resume_from, moasha_cfg)
    else:
        scheduler = MOASHAScheduler(moasha_cfg)

    split_file = str(output_dir / "split.json")
    if dry_run:
        from red_team.evaluator import eval_one_sample_mock as eval_fn
        from red_team.generator import MockPromptGenerator
        generator = MockPromptGenerator(config)
        dataset = [{"paper_id": f"mock_{i}", "input_text": "", "rq_text": "", "paper_text": ""}
                   for i in range(500)]
    else:
        from red_team.evaluator import (
            eval_one_sample as eval_fn, init_models, load_dataset, get_trial_connections,
        )
        init_models(config.get("eval_models", {}), moasha_cfg.max_connections)
        from red_team.generator import PromptGenerator
        generator = PromptGenerator(config)
        generator.set_log_dir(output_dir / "logs" / "optimizer_calls")
        dataset = load_dataset(split_file)

    scheduler.set_dataset(dataset)
    api_semaphore = asyncio.Semaphore(moasha_cfg.max_connections)
    monitor.set_semaphore(api_semaphore, moasha_cfg.max_connections)
    monitor.set_islands(islands)
    islands.set_completed_lists(scheduler.rungs)
    if not dry_run:
        monitor.set_trial_conn_fn(get_trial_connections)

    # Seed initial prompt
    if not scheduler.trials:
        initial_prompt = (RED_TEAM_DIR / "initial_prompt.txt").read_text().strip()
        n_proposals = config.get("eval_models", {}).get("n_proposals", 10)
        if n_proposals != 10:
            initial_prompt = initial_prompt.replace("exactly 10 experiments", f"exactly {n_proposals} experiments")
        trial = PromptTrial(prompt_text=initial_prompt, current_rung=0, island_id=0)
        scheduler.register_trial(trial)

    # Seed prompts: a path ending in .json is a pre-evaluated trial JSON;
    # anything else is treated as a raw prompt text file that we evaluate
    # in-process before injecting (with progress reported to the monitor).
    seeded_ids: list[str] = []
    for path in seed_prompt_paths or []:
        if path.suffix == ".json":
            seed_trial = _load_seed_trial(path)
            existing_json_label = path.stem
        else:
            label = path.stem
            # Skip re-evaluation if a previous seed JSON exists at the conventional path
            seed_dir = output_dir / "seeded"
            cached = seed_dir / f"{label}.json"
            if cached.exists():
                logger.info(f"Using cached seed trial: {cached}")
                seed_trial = _load_seed_trial(cached)
            else:
                if task != "quality":
                    logger.warning(f"Skipping seed eval of {path.name}: "
                                   f"only supported for the 'quality' task")
                    continue
                seed_trial = await _seed_evaluate(
                    prompt_path=path,
                    label=label,
                    output_dir=output_dir,
                    monitor=monitor,
                    eval_fn=eval_fn,
                    dataset=dataset,
                    rung_samples=moasha_cfg.rung_samples,
                    api_semaphore=api_semaphore,
                    log_dir=log_dir,
                    island_id=0,
                )
            existing_json_label = label

        if seed_trial.trial_id in scheduler.trials:
            logger.info(f"Seed trial {seed_trial.trial_id[:8]} already in state; skipping")
            seeded_ids.append(seed_trial.trial_id)
            continue
        scheduler.inject_completed_trial(seed_trial)
        rung = seed_trial.highest_rung
        obj = scheduler.rung_objectives(seed_trial, rung)
        if obj:
            islands.insert(seed_trial.island_id, rung, seed_trial.trial_id,
                           obj[0], obj[1], len(seed_trial.prompt_text))
        seeded_ids.append(seed_trial.trial_id)
        logger.info(
            f"Injected seed trial {seed_trial.trial_id[:8]} at rung {rung}: "
            f"weak={obj[0]:.4f} strong={obj[1]:.4f} ({len(seed_trial.prompt_text)} chars) "
            f"label={existing_json_label}"
        )

    # Clear seeding banner once all seeds are in place
    if seed_prompt_paths:
        monitor.set_seeding(None)

    # Make seeded trial IDs available to the generator for boundary-aware prompting
    if hasattr(generator, "set_seeded_ids"):
        generator.set_seeded_ids(seeded_ids)

    checkpoint_interval = config.get("checkpoint_interval", 10)
    last_checkpoint_n = 0
    monitor.update_scheduler(scheduler)

    active_tasks: dict[str, asyncio.Task] = {}
    worker_counter = 0

    async def _do_eval(worker_key: str, work: EvalWorkItem):
        """Evaluate one sample and report back."""
        trial = scheduler.trials.get(work.trial_id)
        if not trial:
            return
        monitor.start_sample(worker_key, work.trial_id, work.rung)
        try:
            SAMPLE_TIMEOUT = 86400
            result = await asyncio.wait_for(eval_fn(
                trial_id=work.trial_id,
                prompt_text=trial.prompt_text,
                sample=work.paper_sample,
                semaphore=api_semaphore,
                log_dir=log_dir if not dry_run else None,
            ), timeout=SAMPLE_TIMEOUT)
            scheduler.on_sample_complete(work.trial_id, result)
            # Insert into MAP-Elites when trial has rung scores
            t = scheduler.trials[work.trial_id]
            rung_obj = scheduler.rung_objectives(t, work.rung)
            if rung_obj:
                islands.insert(t.island_id, work.rung,
                               work.trial_id, rung_obj[0], rung_obj[1], len(t.prompt_text))
            # Check if rung just completed
            if work.trial_id in [r.completed[-1] for r in scheduler.rungs if r.completed]:
                t = scheduler.trials[work.trial_id]
                obj = scheduler.rung_objectives(t, work.rung)
                if obj:
                    monitor.log_event("rung_complete",
                        f"[{work.trial_id[:8]}] rung={work.rung} w={obj[0]:.3f} s={obj[1]:.3f}")
                if t.status.value == "promoted":
                    monitor.log_event("promoted",
                        f"[{work.trial_id[:8]}] rung {work.rung} -> {work.rung+1}")
                    # Add to new rung's MAP-Elites grid
                    if obj:
                        islands.promote(t.island_id, work.rung + 1,
                                        work.trial_id, obj[0], obj[1], len(t.prompt_text))
        except asyncio.TimeoutError:
            logger.warning(f"Sample timed out [{work.trial_id[:8]}] paper={work.paper_sample['paper_id']}")
            monitor.log_event("timeout", f"[{work.trial_id[:8]}] paper={work.paper_sample['paper_id']}")
            trial._pending_samples = max(0, trial._pending_samples - 1)
        except Exception as e:
            logger.warning(f"Sample eval failed [{work.trial_id[:8]}]: {e}")
            monitor.log_event("error", f"[{work.trial_id[:8]}] {e}")
            trial._pending_samples = max(0, trial._pending_samples - 1)
        finally:
            monitor.finish_sample(worker_key)
            monitor.update_scheduler(scheduler)

    async def _do_generate():
        """Generate a new prompt and register it."""
        scheduler._n_generating += 1
        monitor.update_scheduler(scheduler)
        try:
            island_id = islands.next_island()
            parent_id, sample_mode = islands.sample_parent(island_id)
            parent_trial = scheduler.trials.get(parent_id) if parent_id else None

            pareto = scheduler.get_high_rung_pareto()
            new_prompt = await generator.generate_prompt(
                pareto, parent_trial, api_semaphore,
                island_manager=islands, island_id=island_id, scheduler=scheduler)

            # Use the parent we actually passed, not generator.last_parent_id
            # (which can be overwritten by concurrent generates)
            actual_parent_id = parent_trial.trial_id if parent_trial else None
            actual_generation = (parent_trial.generation + 1) if parent_trial else 0
            new_trial = PromptTrial(
                prompt_text=new_prompt,
                justification=generator.last_justification,
                parent_trial_id=actual_parent_id,
                generation=actual_generation,
                island_id=island_id,
                current_rung=0,
            )
            scheduler.register_trial(new_trial)

            parent_str = f"parent={actual_parent_id[:8]}" if actual_parent_id else "scratch"
            logger.info(f"Generated trial {new_trial.trial_id[:8]} "
                        f"(gen={new_trial.generation}, island={island_id}, "
                        f"{parent_str}, sampled={sample_mode})")
            monitor.log_event("generated",
                f"[{new_trial.trial_id[:8]}] gen={new_trial.generation} "
                f"island={island_id} {parent_str} via {sample_mode}")

            # Check if migration should happen
            migrated = islands.maybe_migrate(scheduler.n_generated)
            for tid, from_isl, to_isl in migrated:
                monitor.log_event("migration", f"[{tid[:8]}] island {from_isl} -> {to_isl}")
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
        finally:
            scheduler._n_generating -= 1
            monitor.update_scheduler(scheduler)

    logger.info(f"Starting MO-ASHA v2: rungs={moasha_cfg.rung_samples}, eta={moasha_cfg.eta}, "
                f"target={moasha_cfg.target_highest_rung}, max_conn={moasha_cfg.max_connections}")

    while not scheduler.is_done():
        # Clean up completed tasks
        done_keys = [k for k, t in active_tasks.items() if t.done()]
        for k in done_keys:
            task = active_tasks.pop(k)
            if task.exception():
                logger.error(f"Task {k} raised: {task.exception()}")

        # Get work
        capacity = moasha_cfg.max_connections - len(active_tasks)
        if capacity <= 0:
            if active_tasks:
                done, _ = await asyncio.wait(active_tasks.values(),
                                              return_when=asyncio.FIRST_COMPLETED)
            continue

        work_items = scheduler.get_work_queue(capacity)

        gen_counter = 0
        for item in work_items:
            if isinstance(item, GenerateWorkItem):
                gen_counter += 1
                task = asyncio.create_task(_do_generate())
                active_tasks[f"gen_{worker_counter}_{gen_counter}"] = task
            elif isinstance(item, EvalWorkItem):
                worker_counter += 1
                wkey = f"w{worker_counter}"
                task = asyncio.create_task(_do_eval(wkey, item))
                active_tasks[wkey] = task

        # Always yield to let dispatched tasks run
        if active_tasks:
            done, _ = await asyncio.wait(active_tasks.values(),
                                          return_when=asyncio.FIRST_COMPLETED)
        else:
            await asyncio.sleep(0.01)

        # Periodic checkpoint + flush API events
        if scheduler.n_generated - last_checkpoint_n >= checkpoint_interval:
            save_state(output_dir / "checkpoints" / f"checkpoint_{scheduler.n_generated}",
                       scheduler)
            _flush_transcript(transcript, output_dir, label=str(scheduler.n_generated))
            last_checkpoint_n = scheduler.n_generated

    # Wait for remaining
    if active_tasks:
        logger.info(f"Waiting for {len(active_tasks)} remaining tasks...")
        await asyncio.gather(*active_tasks.values(), return_exceptions=True)

    _flush_transcript(transcript, output_dir, label="final")
    save_state(output_dir / "checkpoints" / "final", scheduler)
    monitor.update_scheduler(scheduler)

    pareto = scheduler.get_pareto_front()
    logger.info(f"MO-ASHA v2 complete. {len(scheduler.trials)} trials, "
                f"Pareto front: {len(pareto)}, "
                f"Highest rung: {scheduler.n_highest_rung_complete}")

    summary = {
        "n_trials": len(scheduler.trials),
        "n_highest_rung": scheduler.n_highest_rung_complete,
        "pareto": [
            {"trial_id": t.trial_id, "weak": t.objectives[0], "strong": t.objectives[1],
             "rung": t.highest_rung, "generation": t.generation}
            for t in pareto if t.objectives
        ],
    }
    (output_dir / "pareto_summary.json").write_text(json.dumps(summary, indent=2))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MO-ASHA v2 prompt optimization")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed-prompts", type=str, nargs="*", default=None,
                        help="Seed prompt paths. .json = pre-evaluated trial; "
                             "any other extension = raw prompt text to evaluate "
                             "in-process before the main loop starts.")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.resume:
        resume_path = Path(args.resume).resolve()
        output_dir = resume_path
        cp_dir = resume_path / "checkpoints"
        checkpoint_path = None
        if (cp_dir / "final").exists():
            checkpoint_path = cp_dir / "final"
        elif cp_dir.exists():
            # Numeric sort: string order would rank checkpoint_997 above checkpoint_1861.
            def _cp_num(p):
                tail = p.name.split("_")[-1]
                return int(tail) if tail.isdigit() else -1
            cps = sorted([p for p in cp_dir.iterdir() if p.is_dir()], key=_cp_num)
            checkpoint_path = cps[-1] if cps else None
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RED_TEAM_DIR / "runs" / timestamp
        checkpoint_path = None

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    config_out = output_dir / "config.yaml"
    config_out.write_text(yaml.dump(config, default_flow_style=False))

    config_name_path = output_dir / "config_name.txt"
    if not config_name_path.exists():
        config_name_path.write_text((args.config or "base.yaml").strip())

    split_path = output_dir / "split.json"
    if not split_path.exists():
        generate_split(config, split_path)

    print(f"Run directory: {output_dir}")
    print(f"Monitor: uv run python -m red_team.monitor {output_dir}")

    seed_paths: list[Path] = []
    cli_seed_paths = args.seed_prompts or []
    cfg_seed_paths = config.get("seed_prompts") or []
    for raw in list(cli_seed_paths) + list(cfg_seed_paths):
        p = Path(raw)
        if not p.is_absolute():
            p = (output_dir / p).resolve() if not p.exists() else p.resolve()
        if not p.exists():
            print(f"WARNING: seed prompt file not found: {p}")
            continue
        seed_paths.append(p)

    try:
        asyncio.run(main_loop(
            config=config,
            output_dir=output_dir,
            dry_run=args.dry_run,
            resume_from=checkpoint_path,
            seed_prompt_paths=seed_paths,
        ))
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception:
        logger.exception("Fatal error")


if __name__ == "__main__":
    main()
