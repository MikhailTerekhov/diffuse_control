"""Live monitor for the outer saddle loop.

Sections (toggle with 1-8, q to quit):
  1. header     — cycle k/N, current phase, elapsed, ETA
  2. cycles     — completed cycle history with J / Pearson / saddle metrics
  3. timings    — per-phase durations across the run
  4. w_phase    — current W candidate progress (when active)
  5. pf_rungs   — current PF phase v2 rungs (when active)
  6. pf_active  — current PF phase active trials (when active)
  7. events     — outer-loop event log
  8. logs       — tail of outer.log

Run in a separate terminal:
    uv run python -m blue_team.monitor <run_dir>
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

REFRESH = 1.0


def _load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _fmt_duration(s: float) -> str:
    if s < 0:
        return "-"
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{sec:02d}s"


class HttpStatsTracker:
    """Incremental HTTP log parser. Tracks file position to never miss entries."""

    def __init__(self):
        self.total: dict[str, int] = {}
        self._recent_entries: list[tuple[float, str]] = []
        self._file_pos: int = 0
        import re
        self._pattern = re.compile(r"HTTP/[\d.]+ (\d+)")
        self._ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

    def update(self, log_path: Path) -> tuple[dict[str, int], dict[str, int]]:
        if not log_path.exists():
            return self.total, {}
        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size < self._file_pos:
                    self._file_pos = 0
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
                        ts = datetime.strptime(ts_match.group(1),
                                               "%Y-%m-%d %H:%M:%S").timestamp()
                    except Exception:
                        pass
                self._recent_entries.append((ts, status))
            cutoff = now - 1800
            self._recent_entries = [(t, s) for t, s in self._recent_entries if t >= cutoff]
            recent: dict[str, int] = {}
            for _, status in self._recent_entries:
                recent[status] = recent.get(status, 0) + 1
            return self.total, recent
        except Exception:
            return self.total, {}


def _count_anthropic_conns(pid: int) -> tuple[int, int] | None:
    """Return (established, established_with_data) TCP conns to Anthropic for `pid`,
    or None if psutil can't read them."""
    try:
        import psutil
        p = psutil.Process(pid)
        conns = p.connections(kind="tcp")
    except Exception:
        return None
    n_estab = 0
    n_busy = 0
    for c in conns:
        if c.status == psutil.CONN_ESTABLISHED and c.raddr and c.raddr.port == 443:
            n_estab += 1
            # rudimentary "busy" heuristic: socket has data in recv/send queue.
            # psutil doesn't expose Recv-Q/Send-Q directly, so we just report estab.
    return (n_estab, n_busy)


def main_monitor(run_dir: str):
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    run = Path(run_dir).resolve()
    progress = run / "progress"

    console = Console()
    sections_visible = {
        "header": True,
        "cycles": True,
        "timings": True,
        "w_phase": True,
        "pf_rungs": True,
        "pf_active": True,
        "events": True,
        "logs": True,
    }
    section_keys = list(sections_visible.keys())
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
                        sys.stdin.read(1)
                    continue
                key_queue.put(ch)
                if ch == "q":
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    listener = threading.Thread(target=_key_listener, daemon=True)
    listener.start()

    # Read the configured max_connections so we can show "limit" in the header.
    cfg_path = run / "config.yaml"
    try:
        import yaml as _yaml
        cfg = _yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    except Exception:
        cfg = {}
    conn_limit = cfg.get("max_connections")

    # Persistent across refresh ticks: peak observed connection count.
    conn_peak = [0]
    http_tracker = HttpStatsTracker()

    def build_display():
        state = _load_json(progress / "state.json")
        cycles = _load_json(progress / "cycles.json")
        if not isinstance(cycles, list):
            cycles = []
        w_state = _load_json(progress / "w_phase.json")
        events = _load_json(progress / "events.json")
        if not isinstance(events, list):
            events = []
        phase_history = _load_json(progress / "phase_history.json")
        if not isinstance(phase_history, list):
            phase_history = []

        cur_cycle = state.get("current_cycle", "?")
        n_total = state.get("n_cycles_total", "?")
        cur_phase = state.get("current_phase", "?")
        run_started = state.get("run_started_at")
        phase_started = state.get("phase_started_at")
        cycle_pf_dir = Path(state["cycle_run_dir"]) if state.get("cycle_run_dir") else None

        parts = []

        # Toggle hints
        toggles = " ".join(
            f"[bold {'green' if sections_visible[k] else 'dim'}]{i+1}[/]:{k}"
            for i, k in enumerate(section_keys)
        )
        parts.append(Text.from_markup(f"  {toggles}  [dim]q:quit[/]"))

        # --- Header ---
        if sections_visible["header"]:
            elapsed_total = _fmt_duration(time.time() - run_started) if run_started else "-"
            elapsed_phase = _fmt_duration(time.time() - phase_started) if phase_started else "-"
            # ETA for current phase: average duration of completed phases of same type
            eta_str = ""
            done_same = [h["duration"] for h in phase_history
                         if h.get("phase") == cur_phase and h.get("duration") is not None]
            if done_same and phase_started:
                avg = sum(done_same) / len(done_same)
                remaining = max(0.0, avg - (time.time() - phase_started))
                eta_str = (f"  |  ETA: ~{_fmt_duration(remaining)} "
                           f"(avg {_fmt_duration(avg)} over {len(done_same)})")
            phase_color = {
                "bootstrap": "blue", "w_phase": "yellow",
                "pf_phase": "magenta", "done": "green",
            }.get(cur_phase, "white")
            pid_info = _load_json(progress / "pid.json")
            pid = pid_info.get("pid")
            conn_str = ""
            if pid:
                counts = _count_anthropic_conns(pid)
                if counts is None:
                    conn_str = "  |  conns: ?"
                else:
                    n_estab, _ = counts
                    if n_estab > conn_peak[0]:
                        conn_peak[0] = n_estab
                    limit_str = f"/{conn_limit}" if conn_limit is not None else ""
                    conn_str = (
                        f"  |  conns: [bold cyan]{n_estab}{limit_str}[/]  "
                        f"(peak [magenta]{conn_peak[0]}[/])"
                    )

            http_total, http_recent = http_tracker.update(run / "logs" / "http_requests.log")
            def _fmt_http(counts):
                if not counts:
                    return "-"
                parts_h = []
                for code in sorted(counts):
                    color = ("green" if code.startswith("2")
                            else "yellow" if code.startswith("4")
                            else "red")
                    parts_h.append(f"[{color}]{code}[/]:{counts[code]}")
                return " ".join(parts_h)

            header = (
                f"[bold]Outer saddle loop[/]  —  Cycle [bold]{cur_cycle}[/]/{n_total}  "
                f"|  Phase: [{phase_color}]{cur_phase}[/]  "
                f"|  Phase elapsed: {elapsed_phase}{eta_str}  |  Total: {elapsed_total}"
                f"{conn_str}\n"
                f"HTTP total: {_fmt_http(http_total)}  |  last 30m: {_fmt_http(http_recent)}\n"
                f"Run dir: [dim]{run}[/]  pid={pid or '?'}"
            )
            parts.append(Panel(Text.from_markup(header), title="Status", box=box.ROUNDED))

        # --- Cycles history ---
        if sections_visible["cycles"]:
            ct = Table(box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1))
            ct.add_column("Cyc", style="bold")
            ct.add_column("J(W_k)", justify="right")
            ct.add_column("Cand_J", justify="right")
            ct.add_column("Pearson", justify="right")
            ct.add_column("sq_eps_0.05", justify="right")
            ct.add_column("sq_eps_0.20", justify="right")
            ct.add_column("F_k size", justify="right")
            ct.add_column("New", justify="right")
            for c in cycles:
                cyc = c["cycle"]
                w_label = f"W_{cyc}"
                metrics = c.get("metrics") or {}
                m = metrics.get(w_label) or {}
                pf = c.get("pf_summary") or {}
                cand_j = c.get("candidates_J") or []
                cand_str = "/".join(
                    (f"{x:+.2f}" if isinstance(x, (int, float)) and x != float("inf") else "X")
                    for x in cand_j
                )
                ct.add_row(
                    str(cyc),
                    f"{c.get('best_J', float('nan')):+.3f}" if c.get('best_J') is not None else "-",
                    cand_str,
                    f"{m.get('pearson', float('nan')):.3f}" if m.get('pearson') is not None else "-",
                    f"{m.get('sq_eps_0.05'):.3f}" if m.get('sq_eps_0.05') is not None else "-",
                    f"{m.get('sq_eps_0.20'):.3f}" if m.get('sq_eps_0.20') is not None else "-",
                    str(pf.get("n_pf", "-")),
                    str(pf.get("n_new_in_pf", "-")),
                )
            parts.append(Panel(ct, title=f"Cycle history ({len(cycles)} done)", box=box.ROUNDED))

        # --- Phase timings ---
        if sections_visible["timings"] and phase_history:
            tt = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
            tt.add_column("Cyc", style="bold")
            tt.add_column("Phase")
            tt.add_column("Duration", justify="right")
            tt.add_column("Bar")
            durs = [h["duration"] for h in phase_history if h.get("duration") is not None]
            cur_d = (time.time() - phase_started) if phase_started else 0
            max_d = max(durs + [cur_d]) if (durs or cur_d) else 1.0
            phase_color = {"bootstrap": "blue", "w_phase": "yellow",
                           "pf_phase": "magenta", "done": "green"}
            bar_len = 30
            for h in phase_history:
                cyc = h.get("cycle", "?")
                ph = h.get("phase", "?")
                d = h.get("duration")
                col = phase_color.get(ph, "white")
                if d is None:
                    elapsed = time.time() - h.get("started_at", time.time())
                    fill = int(bar_len * elapsed / max_d) if max_d > 0 else 0
                    bar = f"[{col}]" + "█" * fill + "░" * (bar_len - fill) + "[/]"
                    tt.add_row(str(cyc), f"[{col}]{ph}[/] [yellow](in progress)[/]",
                               _fmt_duration(elapsed), bar)
                else:
                    fill = int(bar_len * d / max_d) if max_d > 0 else 0
                    bar = f"[{col}]" + "█" * fill + "░" * (bar_len - fill) + "[/]"
                    tt.add_row(str(cyc), f"[{col}]{ph}[/]", _fmt_duration(d), bar)
            # Summary footer: avg per phase type
            agg = {}
            for h in phase_history:
                if h.get("duration") is None:
                    continue
                agg.setdefault(h["phase"], []).append(h["duration"])
            summary_bits = []
            for ph_name, lst in agg.items():
                avg = sum(lst) / len(lst)
                summary_bits.append(f"{ph_name}: avg {_fmt_duration(avg)} (n={len(lst)})")
            title = "Phase timings"
            if summary_bits:
                title += "  —  " + "  |  ".join(summary_bits)
            parts.append(Panel(tt, title=title, box=box.ROUNDED))

        # --- W phase active ---
        if sections_visible["w_phase"] and cur_phase == "w_phase" and w_state:
            wt = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
            wt.add_column("Cand", style="bold")
            wt.add_column("Status")
            wt.add_column("J", justify="right")
            wt.add_column("fail%", justify="right")
            wt.add_column("ideate", justify="right")
            wt.add_column("score", justify="right")
            wt.add_column("Progress", justify="right")
            wt.add_column("Bar")
            wt.add_column("Summary")

            done_cands = w_state.get("completed_candidates") or []
            cur_idx = w_state.get("current_candidate", 0)
            n_cands = w_state.get("n_candidates", 0)
            haiku_done = w_state.get("haiku_calls_done", 0)
            haiku_total = w_state.get("haiku_calls_total", 0)
            cur_status = w_state.get("status", "?")

            for r in done_cands:
                idx = r["index"]
                jv = r.get("J")
                jstr = (f"{jv:+.3f}" if isinstance(jv, (int, float)) and jv != float('inf')
                        else "FAIL")
                fr = r.get("fail_rate")
                it = r.get("ideation_time")
                st = r.get("scoring_time")
                wt.add_row(
                    str(idx), "[green]done[/]",
                    jstr,
                    f"{fr*100:.1f}%" if fr is not None else "-",
                    _fmt_duration(it) if it is not None else "-",
                    _fmt_duration(st) if st is not None else "-",
                    "-", "-",
                    (r.get("summary") or "")[:80],
                )
            # current candidate
            if cur_idx and cur_idx not in {r["index"] for r in done_cands}:
                bar_len = 24
                pct = haiku_done / haiku_total * 100 if haiku_total else 0
                fill = int(bar_len * pct / 100)
                bar = "█" * fill + "░" * (bar_len - fill)
                wt.add_row(
                    str(cur_idx),
                    f"[yellow]{cur_status}[/]",
                    "-", "-",
                    "-", "-",
                    f"{haiku_done}/{haiku_total} ({pct:.0f}%)",
                    bar, "-",
                )
            parts.append(Panel(
                wt,
                title=f"W phase — cycle {cur_cycle}, "
                      f"P*={w_state.get('p_star_size','?')}, "
                      f"papers/prompt={w_state.get('papers_per_prompt','?')}",
                box=box.ROUNDED,
                border_style="yellow",
            ))

        # --- PF phase rungs ---
        if cur_phase == "pf_phase" and cycle_pf_dir is not None:
            pf_progress = cycle_pf_dir / "progress"
            pf_sched = _load_json(pf_progress / "scheduler.json")
            pf_workers = _load_json(pf_progress / "active_workers.json")

            if sections_visible["pf_rungs"] and pf_sched:
                rt = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
                rt.add_column("Rung", style="bold")
                rt.add_column("N", justify="right")
                rt.add_column("Done", justify="right")
                rt.add_column("Promoted", justify="right")
                rt.add_column("Rate")
                rt.add_column("PF size", justify="right")
                for r in pf_sched.get("rungs", []):
                    comp = r["completed"] if not isinstance(r["completed"], list) else len(r["completed"])
                    prom = r["promoted"] if not isinstance(r["promoted"], list) else len(r["promoted"])
                    rate = f"{prom/comp*100:.0f}%" if comp > 0 else "-"
                    rt.add_row(
                        str(r["rung"]), str(r["n_samples"]),
                        str(comp), str(prom), rate,
                        str(r.get("pareto_size", 0)),
                    )
                conn_used = pf_sched.get("connections_used", 0)
                conn_max = pf_sched.get("connections_max", "?")
                tgt = pf_sched.get("target_highest_rung", "?")
                n_high = pf_sched.get("n_highest_rung", 0)
                title = (f"PF phase — cycle {cur_cycle}  |  "
                         f"highest_rung: {n_high}/{tgt}  |  "
                         f"conn: {conn_used}/{conn_max}  |  "
                         f"trials: {pf_sched.get('n_trials','?')} total, "
                         f"{pf_sched.get('n_generated','?')} generated")
                parts.append(Panel(rt, title=title, box=box.ROUNDED, border_style="magenta"))

            if sections_visible["pf_active"] and pf_sched:
                active = pf_sched.get("active_trials", [])
                if active:
                    at = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
                    at.add_column("Trial")
                    at.add_column("Rung")
                    at.add_column("Progress")
                    at.add_column("Bar")
                    for t in sorted(active, key=lambda x: (-x["rung"], x["trial_id"]))[:20]:
                        done = t["done"]
                        target_n = t["target"]
                        pct = done / target_n * 100 if target_n > 0 else 0
                        bar_len = 14
                        fill = int(bar_len * done / target_n) if target_n > 0 else 0
                        bar = "█" * fill + "░" * (bar_len - fill)
                        rcolor = ["yellow", "cyan", "red"][min(t["rung"], 2)]
                        at.add_row(
                            t["trial_id"][:8],
                            f"[{rcolor}]{t['rung']}[/]",
                            f"{done}/{target_n} ({pct:.0f}%)",
                            bar,
                        )
                    parts.append(Panel(
                        at,
                        title=f"PF phase active trials ({len(active)})",
                        box=box.ROUNDED,
                        border_style="magenta",
                    ))

        # --- Events ---
        if sections_visible["events"] and events:
            now = time.time()
            ev_lines = []
            for ev in events[-15:]:
                age = now - ev.get("ts", 0)
                age_str = f"{age:.0f}s" if age < 120 else f"{age/60:.0f}m" if age < 7200 else f"{age/3600:.1f}h"
                t_color = {
                    "w_phase_start": "yellow", "w_phase_done": "green",
                    "pf_phase_start": "magenta", "pf_phase_done": "green",
                    "loop_done": "bold green",
                }.get(ev.get("type", ""), "white")
                ev_lines.append(f"  [{t_color}]{ev['type']:18s}[/] {ev['detail']}  [dim]({age_str} ago)[/]")
            parts.append(Panel("\n".join(ev_lines), title="Recent events", box=box.ROUNDED))

        # --- Logs ---
        if sections_visible["logs"]:
            log_path = run / "logs" / "outer.log"
            if log_path.exists():
                try:
                    with open(log_path, "rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 4096))
                        tail = f.read().decode("utf-8", errors="replace")
                    log_lines = tail.strip().split("\n")[-12:]
                    colored = []
                    for line in log_lines:
                        if " ERROR" in line or " WARNING" in line:
                            colored.append(f"[red]{line}[/]")
                        elif " Cycle " in line or "complete" in line or "harvested" in line:
                            colored.append(f"[cyan]{line}[/]")
                        else:
                            colored.append(f"[dim]{line}[/]")
                    parts.append(Panel("\n".join(colored), title="outer.log (last 12 lines)", box=box.ROUNDED))
                except Exception:
                    pass

        return Group(*parts)

    with Live(build_display(), console=console, refresh_per_second=1 / REFRESH, screen=True) as live:
        while True:
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
        print("Usage: uv run python -m blue_team.monitor <run_dir>")
        print("Keys: 1-7 toggle sections, q quit")
        sys.exit(1)
    try:
        main_monitor(sys.argv[1])
    except KeyboardInterrupt:
        pass
