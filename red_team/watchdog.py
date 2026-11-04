"""Connection watchdog — kills the run if connections exceed a limit.

Usage:
    uv run python red_team/watchdog.py <run_dir> --limit 64
"""

import json
import os
import signal
import sys
import time
from pathlib import Path


def main(run_dir: str, limit: int = 64, check_interval: float = 2.0):
    d = Path(run_dir)
    progress = d / "progress" if not (d / "progress" / "scheduler.json").exists() else d
    if (d / "progress").exists():
        progress = d / "progress"

    print(f"Watchdog: monitoring {progress}, limit={limit} connections")
    print(f"Will kill the run if connections exceed {limit}")

    # Find the main process PID
    pid_file = progress / "pid"

    while True:
        # Check active workers
        workers_path = progress / "active_workers.json"
        if workers_path.exists():
            try:
                workers = json.loads(workers_path.read_text())
                total_conn = sum(w.get("connections", 0) for w in workers.values())
                n_workers = len(workers)

                if total_conn > limit:
                    print(f"\n*** LIMIT EXCEEDED: {total_conn} connections > {limit} ***")
                    print(f"Workers: {n_workers}")
                    # Find and kill the python process
                    import subprocess
                    result = subprocess.run(
                        ["pgrep", "-f", "red_team/run.py"],
                        capture_output=True, text=True
                    )
                    pids = result.stdout.strip().split("\n")
                    for pid in pids:
                        if pid.strip():
                            print(f"Killing PID {pid}")
                            os.kill(int(pid), signal.SIGTERM)
                    print("Run terminated.")
                    sys.exit(1)

                # Also check via lsof for actual TCP connections to Anthropic
                try:
                    result = subprocess.run(
                        ["ss", "-tnp", "state", "established"],
                        capture_output=True, text=True, timeout=5
                    )
                    anthropic_conns = sum(1 for line in result.stdout.split("\n")
                                         if "api.anthropic.com" in line or "anthropic" in line.lower())
                    status = f"workers={n_workers} tracked_conn={total_conn} tcp_anthropic={anthropic_conns}"
                except Exception:
                    status = f"workers={n_workers} tracked_conn={total_conn}"

                # Print status line (overwrite)
                sys.stdout.write(f"\r  {status}    ")
                sys.stdout.flush()

            except (json.JSONDecodeError, FileNotFoundError):
                pass

        time.sleep(check_interval)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()
    try:
        main(args.run_dir, args.limit)
    except KeyboardInterrupt:
        print("\nWatchdog stopped.")
