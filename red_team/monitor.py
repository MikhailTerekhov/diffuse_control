"""Live progress monitor for MO-ASHA v2.

Rich-based TUI with collapsible sections. Keys:
  1-6: toggle sections  |  q: quit  |  r: refresh

Run in a separate terminal:
    uv run python -m red_team.monitor <run_dir>
"""

import json
import sys
import time
import threading
from pathlib import Path

REFRESH = 1.0


class MonitorState:
    """Writes progress to disk for the standalone monitor to read."""

    def __init__(self, progress_dir: Path):
        self.progress_dir = progress_dir
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self._active_workers: dict[str, dict] = {}
        self._run_start = time.time()
        self._events: list[dict] = []

    def start_sample(self, worker_key: str, trial_id: str, rung: int):
        self._active_workers[worker_key] = {
            "trial_id": trial_id, "rung": rung,
            "connections": 0, "ts": time.time(),
        }
        self._write_workers()

    def update_connections(self, worker_key: str, delta: int):
        if worker_key in self._active_workers:
            self._active_workers[worker_key]["connections"] += delta
            self._active_workers[worker_key]["ts"] = time.time()
            self._write_workers()

    def finish_sample(self, worker_key: str):
        self._active_workers.pop(worker_key, None)
        self._write_workers()

    def log_event(self, event_type: str, detail: str):
        self._events.append({
            "ts": time.time(), "type": event_type, "detail": detail,
        })
        self._atomic_write(self.progress_dir / "events.json", self._events)

    def set_seeding(self, payload: dict | None):
        """Write current seeding status. Pass None to clear."""
        path = self.progress_dir / "seeding.json"
        if payload is None:
            if path.exists():
                path.unlink()
            return
        payload = {**payload, "ts": time.time()}
        self._atomic_write(path, payload)

    def set_semaphore(self, semaphore, max_connections: int):
        self._semaphore = semaphore
        self._max_connections = max_connections

    def set_trial_conn_fn(self, fn):
        self._trial_conn_fn = fn

    def set_islands(self, islands):
        self._islands = islands

    def update_scheduler(self, scheduler):
        summary = scheduler.summary()
        summary["ts"] = time.time()
        summary["run_start"] = self._run_start
        if hasattr(self, "_semaphore"):
            try:
                used = self._max_connections - self._semaphore._value
                summary["connections_used"] = max(0, used)
                summary["connections_max"] = self._max_connections
            except Exception:
                pass
        if hasattr(self, "_trial_conn_fn"):
            try:
                summary["trial_connections"] = self._trial_conn_fn()
            except Exception:
                pass
        if hasattr(self, "_islands"):
            summary["islands"] = self._islands.summary()
        self._atomic_write(self.progress_dir / "scheduler.json", summary)

    def _write_workers(self):
        self._atomic_write(self.progress_dir / "active_workers.json", self._active_workers)

    def _atomic_write(self, path: Path, data):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(path)


def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        return "-"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


class HttpStatsTracker:
    """Incremental HTTP log parser. Tracks file position to never miss entries."""

    def __init__(self):
        self.total: dict[str, int] = {}
        self._recent_entries: list[tuple[float, str]] = []  # (timestamp, status)
        self._file_pos: int = 0
        import re
        self._pattern = re.compile(r'HTTP/[\d.]+ (\d+)')
        self._ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

    def update(self, log_path: Path) -> tuple[dict[str, int], dict[str, int]]:
        """Read new lines since last call. Returns (total, last_30min)."""
        if not log_path.exists():
            return self.total, {}

        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size < self._file_pos:
                    self._file_pos = 0  # file truncated, reset
                f.seek(self._file_pos)
                new_data = f.read().decode("utf-8", errors="replace")
                self._file_pos = f.tell()

            now = time.time()
            for line in new_data.split("\n"):
                m = self._pattern.search(line)
                if not m:
                    continue
                status = m.group(1)
                self.total[status] = self.total.get(status, 0) + 1

                ts_match = self._ts_pattern.match(line)
                ts = now
                if ts_match:
                    try:
                        from datetime import datetime
                        ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
                    except Exception:
                        pass
                self._recent_entries.append((ts, status))

            # Prune entries older than 30 min
            cutoff = now - 1800
            self._recent_entries = [(t, s) for t, s in self._recent_entries if t >= cutoff]

            recent: dict[str, int] = {}
            for _, status in self._recent_entries:
                recent[status] = recent.get(status, 0) + 1

            return self.total, recent
        except Exception:
            return self.total, {}


def _load_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


# --- Rich TUI ---

def main_monitor(progress_dir: str):
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.columns import Columns
    from rich import box

    d = Path(progress_dir)
    if (d / "progress").exists():
        d = d / "progress"
    if not d.exists():
        print(f"Waiting for progress dir: {d}")
        while not d.exists():
            time.sleep(1)

    console = Console()
    sections_visible = {
        "header": True,
        "rungs": True,
        "active": True,
        "islands": True,
        "grids": False,
        "events": True,
        "logs": True,
    }
    section_keys = list(sections_visible.keys())

    # Key listener thread with queue for reliable multi-key input
    import queue
    key_queue: queue.Queue[str] = queue.Queue()

    def _key_listener():
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        sys.stdin.read(1)  # consume arrow key code
                    continue
                key_queue.put(ch)
                if ch == "q":
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    listener = threading.Thread(target=_key_listener, daemon=True)
    listener.start()

    max_conn = 0
    http_tracker = HttpStatsTracker()

    def build_display():
        nonlocal max_conn
        sched = _load_json(d / "scheduler.json")
        workers = _load_json(d / "active_workers.json")
        events = _load_json(d / "events.json") or _load_json(d / "recent_events.json")
        if not isinstance(events, list):
            events = []

        n_gen = sched.get("n_generated", 0)
        n_highest = sched.get("n_highest_rung", 0)
        target = sched.get("target_highest_rung", "?")
        n_generating = sched.get("n_generating", 0)
        conn_used = sched.get("connections_used", 0)
        conn_max = sched.get("connections_max", "?")
        if conn_used > max_conn:
            max_conn = conn_used

        elapsed = ""
        eta = ""
        run_start = sched.get("run_start")
        if run_start:
            el = time.time() - run_start
            elapsed = _fmt_duration(el)
            if n_highest > 0:
                rate = n_highest / el
                rem = (target - n_highest) / rate if rate > 0 and isinstance(target, int) else 0
                eta = _fmt_duration(rem)

        seeding = _load_json(d / "seeding.json")
        if not isinstance(seeding, dict):
            seeding = {}

        parts = []

        # Toggle hints
        toggles = " ".join(
            f"[bold {'green' if sections_visible[k] else 'dim'}]{i+1}[/]:{k}"
            for i, k in enumerate(section_keys)
        )
        parts.append(Text.from_markup(f"  {toggles}  [dim]q:quit[/]"))

        # --- Seeding banner (when a seed prompt is being evaluated) ---
        if seeding:
            label = seeding.get("label", "?")
            done = seeding.get("done", 0)
            total = seeding.get("total", 0)
            failed = seeding.get("failed", 0)
            status = seeding.get("status", "running")
            mean_w = seeding.get("running_mean_weak")
            mean_s = seeding.get("running_mean_strong")
            bar_len = 30
            filled = int(bar_len * done / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_len - filled)
            score_str = ""
            if mean_w is not None and mean_s is not None:
                score_str = f"  running: w={mean_w:.3f} s={mean_s:.3f}"
            color = "magenta" if status == "running" else "green"
            text = (f"[bold {color}]SEEDING[/] [{label}] "
                    f"[{bar}] {done}/{total} done"
                    + (f"  fail={failed}" if failed else "")
                    + score_str
                    + f"  [dim]({status})[/]")
            parts.append(Panel(Text.from_markup(text), title="Seed Evaluation",
                               box=box.ROUNDED, border_style=color))

        # --- Header ---
        if sections_visible["header"]:
            island_data = sched.get("islands") or {}
            total_records = island_data.get("total_records", 0) if isinstance(island_data, dict) else 0
            archive_size = island_data.get("archive_size", 0) if isinstance(island_data, dict) else 0

            # HTTP stats
            http_total, http_recent = http_tracker.update(d.parent / "logs" / "http_requests.log")
            def _fmt_http(counts):
                if not counts:
                    return "-"
                parts_h = []
                for code in sorted(counts):
                    color = "green" if code.startswith("2") else "yellow" if code.startswith("4") else "red"
                    parts_h.append(f"[{color}]{code}[/]:{counts[code]}")
                return " ".join(parts_h)

            isl_list = island_data.get("islands", []) if isinstance(island_data, dict) else []
            pool_total = sum(i.get("pool_size", 0) for i in isl_list)
            cfg_name_path = d.parent / "config_name.txt"
            cfg_name = cfg_name_path.read_text().strip() if cfg_name_path.exists() else "?"
            run_name = d.parent.name
            header = (
                f"[bold]MO-ASHA v2[/] — Run: [cyan]{run_name}[/]  |  Config: [cyan]{cfg_name}[/]\n"
                f"Trials: {n_gen}  |  "
                f"Pool: {pool_total}  Total: {total_records}  Archive: {archive_size}  |  "
                f"Highest rung: [bold]{n_highest}[/]/{target}\n"
                f"Conn: {conn_used}/{conn_max} (peak {max_conn})  |  "
                f"Generating: {n_generating}  |  "
                f"Elapsed: {elapsed}  |  ETA: {eta}\n"
                f"HTTP total: {_fmt_http(http_total)}  |  last 30m: {_fmt_http(http_recent)}"
            )
            parts.append(Panel(Text.from_markup(header), title="Status", box=box.ROUNDED))

        # --- Rungs ---
        if sections_visible["rungs"]:
            rung_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
            rung_table.add_column("Rung", style="bold")
            rung_table.add_column("Samples")
            rung_table.add_column("Completed")
            rung_table.add_column("Promoted")
            rung_table.add_column("Rate")
            rung_table.add_column("PF")
            rung_table.add_column("Pareto Front Scores")

            for r in sched.get("rungs", []):
                comp = r["completed"]
                prom = r["promoted"]
                rate = f"{prom/comp*100:.0f}%" if comp > 0 else "-"
                pf_pts = r.get("pareto_points", [])
                pf_str = "  ".join(
                    f"w={p['weak']:.3f}/s={p['strong']:.3f}"
                    for p in sorted(pf_pts, key=lambda x: -x["weak"])[:6]
                )
                if len(pf_pts) > 6:
                    pf_str += f"  +{len(pf_pts)-6}"
                rung_table.add_row(
                    str(r["rung"]), str(r["n_samples"]),
                    str(comp), str(prom), rate,
                    str(r.get("pareto_size", 0)), pf_str
                )
            parts.append(Panel(rung_table, title="Rungs", box=box.ROUNDED))

        # --- Active trials ---
        if sections_visible["active"]:
            active_trials = sched.get("active_trials", [])
            if active_trials:
                at_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
                at_table.add_column("Trial")
                at_table.add_column("Rung")
                at_table.add_column("Progress")
                at_table.add_column("Bar")
                at_table.add_column("Inflight")

                for t in sorted(active_trials, key=lambda x: (-x["rung"], x["trial_id"])):
                    done = t["done"]
                    target_n = t["target"]
                    pct = done / target_n * 100 if target_n > 0 else 0
                    bar_len = 20
                    filled = int(bar_len * done / target_n) if target_n > 0 else 0
                    bar = "█" * filled + "░" * (bar_len - filled)
                    rung_color = ["yellow", "cyan", "red"][min(t["rung"], 2)]
                    at_table.add_row(
                        t["trial_id"][:8],
                        f"[{rung_color}]{t['rung']}[/]",
                        f"{done}/{target_n} ({pct:.0f}%)",
                        bar,
                        str(t.get("pending", 0)),
                    )
                parts.append(Panel(at_table, title=f"Active Trials ({len(active_trials)})", box=box.ROUNDED))
            else:
                generating_str = "generating..." if n_generating > 0 else "idle"
                parts.append(Panel(f"  ({generating_str})", title="Active Trials", box=box.ROUNDED))

        # --- Islands ---
        if sections_visible["islands"]:
            island_data = sched.get("islands")
            if island_data and isinstance(island_data, dict):
                bounds = island_data.get("bounds")
                mode = island_data.get("mode", "?")
                feat0_name = "score_sum" if mode == "sandbagging" else "score_diff"

                isl_table = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1))
                isl_table.add_column("Island", style="bold")
                isl_table.add_column("Pool", justify="right")
                isl_islands = island_data.get("islands", [])
                n_rungs = len(isl_islands[0].get("rung_sizes", [])) if isl_islands else 0
                for r in range(n_rungs):
                    isl_table.add_column(f"Grid R{r}", justify="right")

                for isl in isl_islands:
                    isl_table.add_row(
                        str(isl["island"]),
                        str(isl.get("pool_size", "?")),
                        *[str(s) for s in isl["rung_sizes"]],
                    )

                bounds_line = ""
                if bounds:
                    bounds_line = (f"{feat0_name}=[{bounds['feat0'][0]:.3f}, {bounds['feat0'][1]:.3f}]  "
                                   f"len=[{bounds['feat1'][0]:.0f}, {bounds['feat1'][1]:.0f}]")

                parts.append(Panel(
                    isl_table,
                    title=f"Islands  {bounds_line}",
                    box=box.ROUNDED,
                ))

        # --- Grids (collapsed by default) ---
        if sections_visible["grids"]:
            island_data = sched.get("islands")
            if island_data and isinstance(island_data, dict):
                n_bins = island_data.get("feature_bins", 4)
                mode = island_data.get("mode", "sandbagging")
                feat0_name = "score_sum" if mode == "sandbagging" else "score_diff"
                bounds = island_data.get("bounds")

                grid_panels = []
                for isl in island_data.get("islands", []):
                    for g in isl.get("grids", []):
                        cells = g.get("cells", [])
                        if not cells:
                            continue

                        # Build cell lookup
                        cell_map = {}
                        for c in cells:
                            key = (c["cell"][0], c["cell"][1])
                            cell_map[key] = c

                        # Build 4x4 table
                        gt = Table(
                            box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 0),
                            title=f"Island {isl['island']} Rung {g['rung']} ({len(cells)} cells)",
                        )
                        # Compute bin edge labels from bounds
                        def _bin_edges(dim, n):
                            if not bounds:
                                return [f"b{i}" for i in range(n)]
                            lo, hi = bounds[f"feat{dim}"]
                            step = (hi - lo) / n if hi > lo else 0
                            if dim == 1:  # prompt length
                                return [f"{lo + i*step:.0f}" for i in range(n)]
                            else:
                                return [f"{lo + i*step:.2f}" for i in range(n)]

                        f0_edges = _bin_edges(0, n_bins)
                        f1_edges = _bin_edges(1, n_bins)

                        gt.add_column(f"{feat0_name}\\len", style="bold", width=8)
                        for j in range(n_bins):
                            gt.add_column(f"len≥{f1_edges[j]}", width=30, no_wrap=True)

                        for i in range(n_bins - 1, -1, -1):
                            row = [f"≥{f0_edges[i]}"]
                            for j in range(n_bins):
                                c = cell_map.get((i, j))
                                if c:
                                    row.append(
                                        f"{c['trial_id'][:6]} "
                                        f"w={c['weak']:.3f} s={c['strong']:.3f} "
                                        f"L={c['prompt_length']}"
                                    )
                                else:
                                    row.append("[dim]·[/]")
                            gt.add_row(*row)

                        grid_panels.append(gt)

                if grid_panels:
                    from rich.console import Group as RGroup
                    parts.append(Panel(RGroup(*grid_panels), title="MAP-Elites Grids", box=box.ROUNDED))

        # --- Events ---
        if sections_visible["events"]:
            if events:
                now = time.time()
                ev_lines = []
                for ev in events[-15:]:
                    age = now - ev.get("ts", 0)
                    age_str = f"{age:.0f}s" if age < 120 else f"{age/60:.0f}m"
                    ev_type = ev["type"]
                    color = {"generated": "green", "rung_complete": "cyan",
                             "promoted": "yellow", "error": "red",
                             "timeout": "red", "migration": "magenta"}.get(ev_type, "white")
                    ev_lines.append(f"  [{color}]{ev_type:15s}[/] {ev['detail']}  [dim]({age_str} ago)[/]")
                parts.append(Panel("\n".join(ev_lines), title="Recent Events", box=box.ROUNDED))

        # --- Logs ---
        if sections_visible["logs"]:
            log_path = d.parent / "logs" / "moasha_v2.log"
            if log_path.exists():
                try:
                    with open(log_path, "rb") as f:
                        # Read last ~4KB
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 4096))
                        tail = f.read().decode("utf-8", errors="replace")
                    log_lines = tail.strip().split("\n")[-15:]
                    # Colorize by level
                    colored = []
                    for line in log_lines:
                        if " ERROR" in line or " WARNING" in line:
                            colored.append(f"[red]{line}[/]")
                        elif " INFO" in line and "Generated" in line:
                            colored.append(f"[green]{line}[/]")
                        elif " INFO" in line and ("rung" in line.lower() or "checkpoint" in line.lower()):
                            colored.append(f"[cyan]{line}[/]")
                        else:
                            colored.append(f"[dim]{line}[/]")
                    parts.append(Panel("\n".join(colored), title="Log (last 15 lines)", box=box.ROUNDED))
                except Exception:
                    pass

        from rich.console import Group
        return Group(*parts)

    with Live(build_display(), console=console, refresh_per_second=1/REFRESH, screen=True) as live:
        while True:
            # Drain all queued keys
            quit_requested = False
            while not key_queue.empty():
                try:
                    k = key_queue.get_nowait()
                except queue.Empty:
                    break
                if k == "q":
                    quit_requested = True
                    break
                if k.isdigit():
                    idx = int(k) - 1
                    if 0 <= idx < len(section_keys):
                        sections_visible[section_keys[idx]] = not sections_visible[section_keys[idx]]
            if quit_requested:
                break

            live.update(build_display())
            time.sleep(REFRESH)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m red_team.monitor <run_dir>")
        print("Keys: 1-7 toggle sections, q quit")
        sys.exit(1)
    try:
        main_monitor(sys.argv[1])
    except KeyboardInterrupt:
        pass
