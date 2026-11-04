"""Full saddle loop CLI: bootstrap + N cycles."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTER_DIR = Path(__file__).resolve().parent.parent

import json
import shutil

from blue_team import api_log
from blue_team.bootstrap import run_bootstrap
from blue_team.saddle_loop import run_saddle_loop


def copy_anchors_from(src_run: Path, dst_run: Path) -> None:
    """Seed dst_run with anchors + W_0 + F_0 from src_run, stripping later-cycle state.

    Copies: split.json, bootstrap/, proposals/, summaries/.
    Filters population.json: keeps only W_0 weak prompt, F_0 front, and prompts
    with role in {p_def, anchor, F_0}. Strips w_scores under W_{k>=1} from samples.
    """
    src_run = Path(src_run).resolve()
    dst_run = Path(dst_run).resolve()
    dst_run.mkdir(parents=True, exist_ok=True)

    if (src_run / "split.json").exists():
        shutil.copy(src_run / "split.json", dst_run / "split.json")
    for d in ("bootstrap", "proposals", "summaries"):
        s = src_run / d
        if s.exists() and not (dst_run / d).exists():
            shutil.copytree(s, dst_run / d)

    pop = json.loads((src_run / "population.json").read_text())

    # Keep only W_0
    pop["weak_prompts"] = {"W_0": pop["weak_prompts"]["W_0"]}
    # Keep only F_0
    pop["fronts"] = {"F_0": pop["fronts"].get("F_0", [])}
    # Keep prompts with role in {p_def, anchor, F_0}
    keep_roles = {"p_def", "anchor", "F_0"}
    new_prompts = {}
    for ph, p in pop["prompts"].items():
        if p["role"] not in keep_roles:
            continue
        # Strip per-paper w_scores under W_k>=1
        for s in p.get("samples", []):
            ws = s.get("w_scores", {})
            s["w_scores"] = {k: v for k, v in ws.items() if k == "W_0"}
        new_prompts[ph] = p
    pop["prompts"] = new_prompts

    (dst_run / "population.json").write_text(json.dumps(pop, indent=2))


def setup_logging(output_dir: Path):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "outer.log", mode="a"),
            logging.StreamHandler(),
        ],
    )
    # Route HTTP-client loggers to a dedicated file so the monitor can scrape
    # status codes (mirrors red_team/run.py's setup).
    http_handler = logging.FileHandler(log_dir / "http_requests.log")
    http_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    for noisy in ("httpx", "httpcore", "anthropic", "LiteLLM"):
        lg = logging.getLogger(noisy)
        lg.setLevel(logging.INFO)
        lg.addHandler(http_handler)
        lg.propagate = False


async def main_async(config: dict, output_dir: Path, n_cycles: int, skip_bootstrap: bool):
    api_log.set_log_dir(output_dir)
    try:
        if not skip_bootstrap:
            await run_bootstrap(config, output_dir)
        await run_saddle_loop(config, output_dir, n_cycles)
    finally:
        await api_log.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="base.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--cycles", type=int, default=None,
                        help="Override n_cycles from config")
    parser.add_argument("--resume", default=None,
                        help="Existing run-dir to resume in (skip bootstrap)")
    parser.add_argument("--anchors-from", default=None,
                        help="Reuse anchors + W_0 + F_0 from a prior run dir; "
                             "creates a fresh run dir and skips bootstrap.")
    parser.add_argument("--source-run", default=None,
                        help="Finished red-team run whose Pareto front seeds F_0. "
                             "Overrides source_run in the config.")
    args = parser.parse_args()

    config = yaml.safe_load((OUTER_DIR / "config" / args.config).read_text())
    n_cycles = args.cycles if args.cycles is not None else config.get("n_cycles", 10)
    if args.source_run:
        config["source_run"] = args.source_run

    # Bootstrap needs a real red-team run to defend against. The shipped configs carry a
    # sentinel because that path is machine-specific.
    if not (args.resume or args.anchors_from or config.get("anchors_from")):
        src = config.get("source_run")
        if not src or src == "REPLACE_ME":
            parser.error(
                "config source_run is unset. The blue team defends against a finished "
                "red-team run, so pass --source-run red_team/runs/<timestamp> (that run "
                "must still have its logs/eval_details/ proposal cache), or set "
                "source_run in the config."
            )

    # Config-level fallback for --anchors-from
    anchors_from = args.anchors_from or config.get("anchors_from")

    if args.resume:
        output_dir = Path(args.resume).resolve()
        skip_bootstrap = True
    elif anchors_from:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTER_DIR / "runs" / ts
        anchors_path = Path(anchors_from)
        if not anchors_path.is_absolute():
            anchors_path = (PROJECT_ROOT / anchors_path).resolve()
        copy_anchors_from(anchors_path, output_dir)
        skip_bootstrap = True
    elif args.out:
        output_dir = Path(args.out).resolve()
        skip_bootstrap = (output_dir / "population.json").exists()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTER_DIR / "runs" / ts
        skip_bootstrap = False

    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "config.yaml").exists():
        (output_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))
    setup_logging(output_dir)
    print(f"Run directory: {output_dir}")

    asyncio.run(main_async(config, output_dir, n_cycles, skip_bootstrap))


if __name__ == "__main__":
    main()
