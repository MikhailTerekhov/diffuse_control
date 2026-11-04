"""OpenEvolve run orchestrator for quality prompt optimization.

Usage:
    uv run python max_weak/run.py
    uv run python max_weak/run.py --config-name config_max_weak
    uv run python max_weak/run.py --config-name config_test
    uv run python max_weak/run.py max_iterations=10
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

load_dotenv()

MAX_WEAK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MAX_WEAK_DIR.parent

if str(MAX_WEAK_DIR) not in sys.path:
    sys.path.insert(0, str(MAX_WEAK_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Register OmegaConf resolver for dynamic config values
from build_system_message import build_system_message

OmegaConf.register_new_resolver(
    "optimizer_prompt",
    lambda mode="max_weak_nostrong", w_path="": build_system_message(mode, w_path=w_path),
)


def generate_split(cfg: DictConfig, out_path: Path):
    from eval_harness.generate_split import generate_split as _generate_split
    _generate_split(
        train_ratio=cfg.split.get("train_ratio", 0.7),
        seed=cfg.split.seed,
        output_path=out_path,
    )


def run_openevolve(cfg: DictConfig, output_dir: Path, config_path: Path, checkpoint_path: Path | None = None):
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(exist_ok=True)
    (progress_dir / "total_iterations").write_text(str(cfg.max_iterations + 1))

    print(f"Progress dir: {progress_dir}")

    env = os.environ.copy()
    env["EVOLVE_SPLIT_FILE"] = str(output_dir / "split.json")
    env["EVOLVE_PROGRESS_DIR"] = str(progress_dir)
    env["EVOLVE_CONFIG"] = str(config_path)
    env["EVOLVE_LLM_LOG_DIR"] = str(output_dir / "logs" / "optimizer_calls")

    cmd = [
        sys.executable, "-m", "max_weak.openevolve_main",
        str(MAX_WEAK_DIR / "initial_prompt.txt"),
        str(MAX_WEAK_DIR / "evaluator.py"),
        "--config", str(config_path),
        "--output", str(output_dir),
        "--iterations", str(cfg.max_iterations),
    ]

    if checkpoint_path:
        cmd += ["--checkpoint", str(checkpoint_path)]

    import signal
    stop_file = progress_dir / "STOP"
    proc = subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT),
                            start_new_session=True)
    try:
        while proc.poll() is None:
            # Check for STOP file from evaluator (stage2 limit reached)
            if stop_file.exists():
                print(f"STOP file detected: {stop_file.read_text().strip()}")
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait()
                break
            import time
            time.sleep(2)
    except (KeyboardInterrupt, EOFError):
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait()
    return proc.returncode


def copy_best_to_prompts(output_dir: Path, prompts_dir: Path):
    best = output_dir / "best" / "best_program.txt"
    if not best.exists():
        return
    prompts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, prompts_dir / "best.txt")
    for f in ("full_prompt.txt", "best_program_info.json"):
        src = output_dir / "best" / f
        if src.exists():
            shutil.copy(src, prompts_dir / f)
    split = output_dir / "split.json"
    if split.exists():
        shutil.copy(split, prompts_dir / "split.json")
    print(f"Best prompt saved to: {prompts_dir / 'best.txt'}")


@hydra.main(version_base=None, config_path="config", config_name="config_max_weak_nostrong")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    checkpoint = cfg.get("checkpoint")
    checkpoint_path = None
    if checkpoint:
        checkpoint_path = Path(checkpoint).resolve()
        output_dir = checkpoint_path.parent.parent
        timestamp = output_dir.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = MAX_WEAK_DIR / "runs" / timestamp

    prompts_dir = MAX_WEAK_DIR / "runs" / timestamp / "prompts"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.yaml"
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    config_path.write_text(resolved_yaml)

    config_name_path = output_dir / "config_name.txt"
    if not config_name_path.exists():
        try:
            from hydra.core.hydra_config import HydraConfig
            cfg_name = HydraConfig.get().job.config_name or "config_max_weak_nostrong"
        except Exception:
            cfg_name = "config_max_weak_nostrong"
        config_name_path.write_text(str(cfg_name))

    print(f"Run: {output_dir}")
    print(f"Prompts will be saved to: {prompts_dir}")
    display_cfg = OmegaConf.to_container(cfg, resolve=True)
    sm = display_cfg.get("prompt", {}).get("system_message", "")
    if len(sm) > 500:
        display_cfg["prompt"]["system_message"] = sm[:500] + f"... [{len(sm)} chars total]"
    print(f"Config:\n{OmegaConf.to_yaml(OmegaConf.create(display_cfg))}")

    # 1. Generate split (skip if resuming and split exists)
    split_path = output_dir / "split.json"
    if not split_path.exists():
        print("Generating split...")
        generate_split(cfg, split_path)
    else:
        print(f"Using existing split: {split_path}")

    # 2. Run OpenEvolve
    print("Starting evolution...")
    returncode = run_openevolve(cfg, output_dir, config_path,
                               checkpoint_path=checkpoint_path)

    # 3. Post-processing
    copy_best_to_prompts(output_dir, prompts_dir)

    return returncode


if __name__ == "__main__":
    main()
