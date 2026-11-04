"""Quick status check for a running rl_red_team training run."""

import json
import sys
from pathlib import Path


def status(run_dir: str | None = None):
    if run_dir is None:
        runs = sorted(Path("rl_red_team/runs").iterdir(), key=lambda p: p.stat().st_mtime)
        if not runs:
            print("No runs found")
            return
        run_dir = str(runs[-1])

    run_path = Path(run_dir)
    print(f"Run: {run_path.name}")

    # Metrics
    metrics_path = run_path / "metrics.jsonl"
    if metrics_path.exists():
        lines = [json.loads(l) for l in metrics_path.read_text().strip().splitlines() if l.strip()]
        if lines:
            last = lines[-1]
            step = last.get("progress/batch", "?")
            done = last.get("progress/done_frac", 0)
            reward = last.get("env/all/reward", "?")
            weak = last.get("env/all/weak_score", "?")
            strong = last.get("env/all/strong_score", "?")
            parse_fail = last.get("env/all/judge_parse_fail", "?")
            t_sample = last.get("time/sampling", "?")
            t_train = last.get("time/train_step", "?")
            t_total = last.get("time/total", "?")

            print(f"  Steps: {step}/{len(lines)} ({done:.0%})")
            print(f"  Reward: {reward:.3f}  weak={weak:.3f}  strong={strong:.3f}  parse_fail={parse_fail}")
            print(f"  Time: total={t_total:.0f}s  sample={t_sample:.0f}s  train={t_train:.0f}s")

            # Reward trend (last 5)
            if len(lines) >= 2:
                recent = lines[-min(5, len(lines)):]
                rewards = [l.get("env/all/reward", 0) for l in recent]
                print(f"  Reward trend (last {len(recent)}): {' → '.join(f'{r:.3f}' for r in rewards)}")
        else:
            print("  No metrics yet")
    else:
        print("  No metrics file yet")

    # Debug log - check sampling/judge peaks
    debug_path = run_path / "debug.log"
    if debug_path.exists():
        text = debug_path.read_text()
        import re
        sampling_peaks = re.findall(r"peak=(\d+)", text)
        if sampling_peaks:
            print(f"  Sampling peak concurrency: {max(int(p) for p in sampling_peaks)}")
        judge_peaks = [int(p) for p in re.findall(r"judge start.*peak=(\d+)", text)]
        if judge_peaks:
            print(f"  Judge peak concurrency: {max(judge_peaks)}")

    # Check if process is still running
    import subprocess
    result = subprocess.run(
        ["pgrep", "-f", "rl_red_team.train"],
        capture_output=True, text=True,
    )
    pids = result.stdout.strip().splitlines()
    print(f"  Process: {'running (PID ' + ', '.join(pids) + ')' if pids else 'NOT RUNNING'}")


if __name__ == "__main__":
    status(sys.argv[1] if len(sys.argv) > 1 else None)
