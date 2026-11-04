"""Live TUI monitor for rl_red_team training runs.

Usage: uv run python -m rl_red_team.monitor [run_dir]

Reads debug.log and metrics.jsonl directly - no progress.json needed.
"""

import json
import os
import re
import sys
import time
from pathlib import Path


REFRESH = 1.0

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
WHITE = "\033[97m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"


def fmt_time(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def bar(filled: int, total: int, width: int) -> str:
    if total <= 0:
        return ""
    n = min(int(width * filled / total), width)
    return "█" * n + "░" * (width - n)


def parse_debug_log(path: Path) -> dict:
    """Parse debug.log to extract live state."""
    state = {
        "sampling_active": 0,
        "sampling_peak": 0,
        "sampling_starts": 0,
        "sampling_dones": 0,
        "judge_active": 0,
        "judge_peak": 0,
        "judge_starts": 0,
        "judge_dones": 0,
        "is_eval": False,
        "last_ts": None,
        "events": [],  # last N events
    }

    if not path.exists():
        return state

    try:
        with open(path) as f:
            for line in f:
                if "Sampling start" in line:
                    state["sampling_starts"] += 1
                    m = re.search(r"active=(\d+).*peak=(\d+)", line)
                    if m:
                        state["sampling_active"] = int(m.group(1))
                        state["sampling_peak"] = max(state["sampling_peak"], int(m.group(2)))
                    if "eval=True" in line:
                        state["is_eval"] = True
                    elif "eval=False" in line:
                        state["is_eval"] = False
                    state["last_ts"] = line[:23]
                elif "Sampling done" in line:
                    state["sampling_dones"] += 1
                    m = re.search(r"active=(\d+)", line)
                    if m:
                        state["sampling_active"] = int(m.group(1))
                    state["last_ts"] = line[:23]
                elif "judge start" in line:
                    state["judge_starts"] += 1
                    m = re.search(r"active=(\d+).*peak=(\d+)", line)
                    if m:
                        state["judge_active"] = int(m.group(1))
                        state["judge_peak"] = max(state["judge_peak"], int(m.group(2)))
                    state["last_ts"] = line[:23]
                elif "judge done" in line:
                    state["judge_dones"] += 1
                    m = re.search(r"active=(\d+)", line)
                    if m:
                        state["judge_active"] = int(m.group(1))
                    state["last_ts"] = line[:23]
                elif "judge" in line.lower() and ("failed" in line or "error" in line):
                    state["events"].append(line.strip()[-100:])
    except Exception:
        pass

    state["events"] = state["events"][-5:]  # keep last 5
    return state


def render(debug_state: dict, metrics_lines: list[dict], gantt_files: list[str],
           run_start: float, term_width: int) -> str:
    lines = []
    w = min(term_width, 100)
    now = time.time()

    # Header
    lines.append(f"{BOLD}{CYAN}{'═' * w}{RESET}")
    lines.append(f"{BOLD}{CYAN}  Tinker RL Training Monitor{RESET}")
    lines.append(f"{BOLD}{CYAN}{'═' * w}{RESET}")

    n_steps = len(metrics_lines)
    is_eval = debug_state["is_eval"]
    uptime = now - run_start if run_start else 0

    # Phase detection
    sa = debug_state["sampling_active"]
    # How old is the last debug entry
    last_activity_age = None
    if debug_state["last_ts"]:
        try:
            from datetime import datetime
            last_dt = datetime.strptime(debug_state["last_ts"], "%Y-%m-%d %H:%M:%S")
            last_activity_age = (datetime.now() - last_dt).total_seconds()
        except Exception:
            pass

    all_sampled = debug_state["sampling_starts"] > 0 and sa == 0
    log_gone_quiet = last_activity_age is not None and last_activity_age > 30

    if sa > 0:
        phase = f"{'📊 Eval' if is_eval else '🔄 Training'} - Sampling"
        phase_color = MAGENTA if is_eval else CYAN
    elif not all_sampled and debug_state["sampling_starts"] == 0 and n_steps == 0:
        phase = "⏳ Initializing"
        phase_color = DIM
    elif all_sampled and not log_gone_quiet:
        phase = f"{'📊 Eval' if is_eval else '⚖️  Training'} - Judging"
        phase_color = MAGENTA if is_eval else YELLOW
    elif all_sampled and log_gone_quiet:
        phase = f"🏋️  Optimizer step ({fmt_time(last_activity_age)} since last judge)"
        phase_color = GREEN
    else:
        phase = "⏸  Between steps"
        phase_color = DIM

    # Step timing
    if metrics_lines:
        step_times = [m.get("time/total", 0) for m in metrics_lines]
        avg_step = sum(step_times) / len(step_times) if step_times else 0
    else:
        avg_step = 0

    lines.append("")
    lines.append(f"  {BOLD}Steps completed: {n_steps}{RESET}   "
                 f"Uptime: {fmt_time(uptime)}   "
                 f"Avg step: {fmt_time(avg_step)}")
    lines.append(f"  Phase: {phase_color}{BOLD}{phase}{RESET}")

    # Last activity
    if debug_state["last_ts"]:
        lines.append(f"  Last activity: {DIM}{debug_state['last_ts']}{RESET}")

    # Sampling
    lines.append("")
    lines.append(f"  {CYAN}━━━ Sampling ━━━{RESET}")
    total_starts = debug_state["sampling_starts"]
    total_dones = debug_state["sampling_dones"]
    in_flight = debug_state["sampling_active"]
    peak = debug_state["sampling_peak"]

    if in_flight > 0:
        lines.append(f"    In-flight: {BOLD}{in_flight}{RESET}   "
                     f"Peak: {peak}   "
                     f"Total: {total_starts} started, {total_dones} done")
    else:
        lines.append(f"    {DIM}Idle{RESET}   "
                     f"Peak: {peak}   "
                     f"Total: {total_starts} started, {total_dones} done")

    # Judging
    lines.append("")
    lines.append(f"  {YELLOW}━━━ Judging ━━━{RESET}")
    ja = debug_state["judge_active"]
    j_starts = debug_state["judge_starts"]
    j_dones = debug_state["judge_dones"]
    j_peak = debug_state["judge_peak"]

    if ja > 0:
        judge_bar = bar(j_dones, j_starts, w - 40) if j_starts > 0 else ""
        lines.append(f"    Active: {BOLD}{ja}{RESET}   "
                     f"Done: {j_dones}/{j_starts}   "
                     f"Peak: {j_peak}")
        if judge_bar:
            lines.append(f"    [{judge_bar}]")
    elif j_starts > 0:
        lines.append(f"    {DIM}Done{RESET}   "
                     f"Total: {j_dones}/{j_starts}   "
                     f"Peak: {j_peak}")
    else:
        lines.append(f"    {DIM}Idle{RESET}")

    # Errors/warnings
    if debug_state["events"]:
        lines.append("")
        lines.append(f"  {RED}━━━ Recent Issues ━━━{RESET}")
        for ev in debug_state["events"]:
            lines.append(f"    {DIM}{ev}{RESET}")

    # Metrics table
    lines.append("")
    lines.append(f"  {GREEN}━━━ Metrics ━━━{RESET}")
    if metrics_lines:
        lines.append(f"    {'Step':>5} {'Reward':>8} {'Weak':>8} {'Strong':>8} {'Fail%':>7} {'Time':>7}")
        lines.append(f"    {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")
        for m in metrics_lines[-10:]:
            s = m.get("progress/batch", "?")
            r = m.get("env/all/reward", 0)
            wk = m.get("env/all/weak_score", 0)
            st = m.get("env/all/strong_score", 0)
            pf = m.get("env/all/judge_parse_fail", 0)
            tt = m.get("time/total", 0)
            lines.append(f"    {s:>5} {r:>8.3f} {wk:>8.3f} {st:>8.3f} {pf*100:>6.1f}% {fmt_time(tt):>7}")
    else:
        lines.append(f"    {DIM}No metrics yet{RESET}")

    # Gantt charts
    if gantt_files:
        lines.append("")
        lines.append(f"  {BLUE}━━━ Gantt Charts ━━━{RESET}")
        for gf in gantt_files[-3:]:
            lines.append(f"    {DIM}{gf}{RESET}")

    lines.append("")
    lines.append(f"{DIM}  Ctrl+C to exit  |  Refresh: {REFRESH}s{RESET}")
    lines.append(f"{CYAN}{'─' * w}{RESET}")

    return "\n".join(lines)


def run_monitor(run_dir: str):
    run_path = Path(run_dir)
    debug_path = run_path / "debug.log"
    metrics_path = run_path / "metrics.jsonl"
    run_start = run_path.stat().st_ctime if run_path.exists() else time.time()

    sys.stdout.write("\033[?25l")
    try:
        while True:
            debug_state = parse_debug_log(debug_path)

            metrics_lines = []
            if metrics_path.exists():
                try:
                    metrics_lines = [
                        json.loads(line)
                        for line in metrics_path.read_text().strip().splitlines()
                        if line.strip()
                    ]
                except (json.JSONDecodeError, OSError):
                    pass

            gantt_files = [
                str(d.relative_to(run_path))
                for d in sorted(run_path.glob("iteration_*/timing_gantt.html"))
            ]

            term_width = os.get_terminal_size().columns
            output = render(debug_state, metrics_lines, gantt_files, run_start, term_width)

            sys.stdout.write("\033[H\033[2J")
            sys.stdout.write(output)
            sys.stdout.flush()

            time.sleep(REFRESH)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")
        sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        runs_dir = Path("rl_red_team/runs")
        if not runs_dir.exists():
            print("No runs found in rl_red_team/runs/")
            sys.exit(1)
        runs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        if not runs:
            print("No runs found in rl_red_team/runs/")
            sys.exit(1)
        run_dir = str(runs[-1])
    run_monitor(run_dir)
