"""Metrics + scatter plot generation for outer-loop diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blue_team.population import Population, PromptEntry


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def saddle_quantity(p_star: list[PromptEntry], w_label: str, p_def: PromptEntry,
                    epsilons: list[float] | None = None) -> dict:
    """For each ε in epsilons: min s_mean over prompts whose w_mean >= w_def - ε."""
    if epsilons is None:
        epsilons = [0.0, 0.05, 0.10, 0.20]
    w_def = p_def.w_mean(w_label)
    if w_def is None:
        return {f"sq_eps_{e:.2f}": None for e in epsilons}
    out = {}
    for eps in epsilons:
        feasible_s = []
        for p in p_star:
            wm = p.w_mean(w_label)
            sm = p.s_mean
            if wm is None or sm is None:
                continue
            if wm >= w_def - eps:
                feasible_s.append(sm)
        out[f"sq_eps_{eps:.2f}"] = min(feasible_s) if feasible_s else None
    return out


def metrics_under_w(p_star: list[PromptEntry], w_label: str, p_def: PromptEntry) -> dict:
    pts = [(p.s_mean, p.w_mean(w_label)) for p in p_star]
    pts = [(s, w) for s, w in pts if s is not None and w is not None]
    if not pts:
        return {"pearson": float("nan"), "n": 0}
    xs, ys = zip(*pts)
    out = {
        "pearson": pearson(xs, ys),
        "n": len(pts),
        "w_def": p_def.w_mean(w_label),
        "s_def": p_def.s_mean,
    }
    out.update(saddle_quantity(p_star, w_label, p_def))
    return out


def scatter_population(
    population: Population,
    w_label: str,
    out_path: Path,
    title: str = "",
    extra_label_for_w: str | None = None,
) -> None:
    """Scatter (s_mean, w_mean) for all prompts in population, color-coded by role."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    role_colors = {"p_def": "black", "anchor": "tab:green", "F_0": "tab:red",
                   "F_1": "tab:orange", "F_2": "tab:purple", "F_3": "tab:blue"}
    fig, ax = plt.subplots(figsize=(7, 6))

    by_role: dict[str, list[tuple[float, float]]] = {}
    for p in population.prompts.values():
        s = p.s_mean
        w = p.w_mean(w_label)
        if s is None or w is None:
            continue
        role = p.role
        by_role.setdefault(role, []).append((s, w))

    for role, pts in by_role.items():
        xs, ys = zip(*pts)
        c = role_colors.get(role, "gray")
        marker = "*" if role == "p_def" else ("D" if role == "anchor" else "o")
        size = 200 if role == "p_def" else (60 if role == "anchor" else 30)
        ax.scatter(xs, ys, label=role, c=c, marker=marker, s=size, alpha=0.7,
                   edgecolors="black", linewidth=0.5)

    ax.set_xlabel("strong (s_mean)")
    ax.set_ylabel(f"weak under {w_label} (w_mean)")
    if extra_label_for_w:
        ax.set_ylabel(f"weak under {w_label} — {extra_label_for_w}")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.4, label="diagonal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def scatter_w_phase_change(
    population: Population,
    prev_w_label: str,
    new_w_label: str,
    out_path: Path,
    title: str = "",
) -> None:
    """For each prompt with both old and new w scores, draw an arrow from
    (s, w_old) → (s, w_new). s should be unchanged. Color-coded by role."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    role_colors = {"p_def": "black", "anchor": "tab:green", "F_0": "tab:red",
                   "F_1": "tab:orange", "F_2": "tab:purple", "F_3": "tab:blue",
                   "F_4": "tab:pink", "F_5": "tab:brown"}
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.4, label="diagonal")

    by_role: dict[str, list[tuple[float, float, float]]] = {}
    for p in population.prompts.values():
        s = p.s_mean
        w_old = p.w_mean(prev_w_label)
        w_new = p.w_mean(new_w_label)
        if s is None or w_old is None or w_new is None:
            continue
        by_role.setdefault(p.role, []).append((s, w_old, w_new))

    legend_handles = []
    for role, pts in by_role.items():
        c = role_colors.get(role, "gray")
        marker = "*" if role == "p_def" else ("D" if role == "anchor" else "o")
        size = 200 if role == "p_def" else (60 if role == "anchor" else 30)
        for s, w_old, w_new in pts:
            ax.scatter([s], [w_old], c=c, marker=marker, s=size, alpha=0.25,
                       edgecolors="black", linewidth=0.4)
            ax.scatter([s], [w_new], c=c, marker=marker, s=size, alpha=0.95,
                       edgecolors="black", linewidth=0.6)
            dy = w_new - w_old
            if abs(dy) > 0.005:
                ax.annotate("", xy=(s, w_new), xytext=(s, w_old),
                            arrowprops=dict(arrowstyle="-|>", color=c, alpha=0.6,
                                            lw=1.2, mutation_scale=10))
        # Legend entry: filled marker for the role
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D([0], [0], marker=marker, color="w",
                                     markerfacecolor=c, markersize=10,
                                     markeredgecolor="black", label=role))

    ax.set_xlabel("strong (s_mean) — unchanged across W phase")
    ax.set_ylabel(f"weak: {prev_w_label} (faded) → {new_w_label} (solid)")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    legend_handles.append(Line2D([0], [0], linestyle="--", color="gray",
                                 alpha=0.4, label="diagonal"))
    ax.legend(handles=legend_handles, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _sandbagging_pareto(points: list[tuple[float, float]]) -> list[bool]:
    """Pareto frontier under: minimize s, maximize w. Returns is_dominant flag."""
    n = len(points)
    is_dom = [True] * n
    for i, (si, wi) in enumerate(points):
        for j, (sj, wj) in enumerate(points):
            if i == j:
                continue
            # j dominates i iff sj <= si AND wj >= wi AND at least one strict
            if sj <= si and wj >= wi and (sj < si or wj > wi):
                is_dom[i] = False
                break
    return is_dom


def _load_dominated_new_rung2(cycle_pf_dir: Path, exclude_hashes: set) -> list[tuple[float, float]]:
    """Return (s, w) for every NEW rung-2 trial in the checkpoint whose prompt
    hash is not already in `exclude_hashes` (typically population prompts).
    These are the new prompts that got fully evaluated but were dominated by
    seeds, so they didn't make it into the PF and aren't in population.json."""
    cp_root = cycle_pf_dir / "checkpoints"
    final = cp_root / "final"
    if not final.exists():
        cps = sorted([p for p in cp_root.iterdir() if p.is_dir() and p.name != "final"],
                     key=lambda p: int(p.name.rsplit("_", 1)[-1]))
        if not cps:
            return []
        final = cps[-1]
    state = json.loads((final / "state.json").read_text())
    out = []
    from blue_team.population import prompt_hash
    for tid in state["rungs"][2]["completed"]:
        t = state["trials"][tid]
        if t.get("generation", 0) < 0:
            continue   # seed, already in population
        ph = prompt_hash(t["prompt_text"])
        if ph in exclude_hashes:
            continue
        samples = t["sample_results"]
        valid = [s for s in samples
                 if s.get("weak_score") is not None and s.get("strong_score") is not None]
        if not valid:
            continue
        avg_w = sum(s["weak_score"] for s in valid) / len(valid)
        avg_s = sum(s["strong_score"] for s in valid) / len(valid)
        out.append((avg_s, avg_w))
    return out


def scatter_after_pf(
    population: Population,
    w_label: str,
    new_front_label: str,     # e.g. "F_3" for cycle 3
    out_path: Path,
    title: str = "",
    cycle_pf_dir: Path | None = None,
) -> None:
    """Two-color scatter (old vs new) with Pareto-dominant points highlighted.

    "New" = prompts with role == new_front_label (just added by this cycle's
    PF phase). All other roles (anchors + earlier F_k) are "old".
    Pareto = sandbagging-style: minimize s, maximize w. Dominant points are
    drawn larger with a star marker.

    If `cycle_pf_dir` is provided, also reads the v2 checkpoint and adds any
    new rung-2 trials that were dominated and didn't make it into population.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.4)

    pts = []   # (s, w, ph, is_new, role)
    for p in population.prompts.values():
        s = p.s_mean
        w = p.w_mean(w_label)
        if s is None or w is None:
            continue
        pts.append((s, w, p.prompt_hash, p.role == new_front_label, p.role))

    # Pull dominated new rung-2 trials from the v2 checkpoint (not in population).
    if cycle_pf_dir is not None:
        excl = {ph for _, _, ph, *_ in pts}
        for s, w in _load_dominated_new_rung2(cycle_pf_dir, excl):
            pts.append((s, w, "(dominated)", True, f"{new_front_label}_dominated"))
    if not pts:
        ax.set_title(title)
        fig.savefig(out_path, dpi=120); plt.close(fig)
        return

    is_dom = _sandbagging_pareto([(s, w) for s, w, *_ in pts])

    # Classify each point: p_def / anchor / old PF / new PF
    pdef_pts, anchor_pts = [], []
    old_dom, old_nondom, new_dom, new_nondom = [], [], [], []
    for (s, w, ph, is_new, role), d in zip(pts, is_dom):
        if role == "p_def":
            pdef_pts.append((s, w))
        elif role == "anchor":
            anchor_pts.append((s, w))
        elif is_new:
            (new_dom if d else new_nondom).append((s, w))
        else:
            (old_dom if d else old_nondom).append((s, w))

    # Layer 1: non-dominant PF points (small)
    if old_nondom:
        xs, ys = zip(*old_nondom)
        ax.scatter(xs, ys, c="tab:gray", marker="o", s=22, alpha=0.55,
                   edgecolors="black", linewidth=0.3,
                   label=f"old PF non-dom ({len(old_nondom)})")
    if new_nondom:
        xs, ys = zip(*new_nondom)
        ax.scatter(xs, ys, c="tab:orange", marker="o", s=22, alpha=0.55,
                   edgecolors="black", linewidth=0.3,
                   label=f"{new_front_label} non-dom ({len(new_nondom)})")
    # Layer 2: dominant PF points (a touch larger, star)
    if old_dom:
        xs, ys = zip(*old_dom)
        ax.scatter(xs, ys, c="tab:gray", marker="*", s=110, alpha=0.95,
                   edgecolors="black", linewidth=0.6,
                   label=f"old PF dom ({len(old_dom)})")
    if new_dom:
        xs, ys = zip(*new_dom)
        ax.scatter(xs, ys, c="tab:orange", marker="*", s=110, alpha=0.95,
                   edgecolors="black", linewidth=0.6,
                   label=f"{new_front_label} dom ({len(new_dom)})")
    # Layer 3: anchors + p_def (highlighted distinctly)
    if anchor_pts:
        xs, ys = zip(*anchor_pts)
        ax.scatter(xs, ys, c="tab:green", marker="D", s=70, alpha=0.95,
                   edgecolors="black", linewidth=0.6,
                   label=f"anchors ({len(anchor_pts)})")
    if pdef_pts:
        xs, ys = zip(*pdef_pts)
        ax.scatter(xs, ys, c="black", marker="*", s=220, alpha=1.0,
                   edgecolors="white", linewidth=0.6,
                   label="p_def")

    from matplotlib.lines import Line2D
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], linestyle="--", color="gray",
                          alpha=0.4, label="diagonal"))
    ax.legend(handles=handles, loc="best", fontsize=8)
    ax.set_xlabel("strong (s_mean)")
    ax.set_ylabel(f"weak under {w_label} (w_mean)")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def scatter_bootstrap_overview(
    population: Population,
    candidate_entries: list[PromptEntry],
    s_def: float,
    threshold: float,
    out_path: Path,
) -> None:
    """Combined view: F_0 PF + anchor candidates (kept/dropped) + p_def, all under W_0."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    # F_0
    f0_pts = []
    for h in population.fronts.get("F_0", []):
        p = population.prompts.get(h)
        if p is None or p.s_mean is None or p.w_mean("W_0") is None:
            continue
        f0_pts.append((p.s_mean, p.w_mean("W_0")))
    if f0_pts:
        xs, ys = zip(*f0_pts)
        ax.scatter(xs, ys, c="tab:red", marker="o", s=40, alpha=0.6, label=f"F_0 ({len(f0_pts)})",
                   edgecolors="black", linewidth=0.4)

    # Anchor candidates (kept and dropped) — using role assigned by bootstrap
    pts_kept, pts_dropped, pts_pdef = [], [], []
    for e in candidate_entries:
        s = e.s_mean
        w = e.w_mean("W_0")
        if s is None or w is None:
            continue
        if e.role in ("p_def", "p_def_candidate"):
            pts_pdef.append((s, w))
        elif e.role == "anchor":
            pts_kept.append((s, w))
        else:
            pts_dropped.append((s, w))

    if pts_dropped:
        xs, ys = zip(*pts_dropped)
        ax.scatter(xs, ys, c="tab:gray", marker="x", s=60, label=f"anchor_dropped ({len(pts_dropped)})")
    if pts_kept:
        xs, ys = zip(*pts_kept)
        ax.scatter(xs, ys, c="tab:green", marker="D", s=80, label=f"anchor_kept ({len(pts_kept)})",
                   edgecolors="black", linewidth=0.5)
    if pts_pdef:
        xs, ys = zip(*pts_pdef)
        ax.scatter(xs, ys, c="black", marker="*", s=300, label="p_def",
                   edgecolors="white", linewidth=1.5)

    ax.axvspan(s_def - threshold, s_def + threshold, alpha=0.08, color="tab:green",
               label=f"|s - s_def| ≤ {threshold}")
    ax.axvline(s_def, color="black", linestyle=":", alpha=0.5)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.4, label="diagonal")

    ax.set_xlabel("strong score (Sonnet w/ paper)")
    ax.set_ylabel("weak score under W_0 (rubric)")
    ax.set_title("Phase 0 overview: F_0 PF + anchor candidates under W_0")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def scatter_anchor_candidates(
    candidate_entries: list[PromptEntry],
    s_def: float,
    threshold: float,
    out_path: Path,
    title: str = "Anchor candidates (Phase 0)",
) -> None:
    """Scatter (s_mean, w_mean under W_0) for the 21 anchor candidates,
    coloring kept vs filtered."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    pts_kept, pts_dropped, pts_pdef = [], [], []
    labels_kept, labels_dropped = [], []
    for e in candidate_entries:
        s = e.s_mean
        w = e.w_mean("W_0")
        if s is None or w is None:
            continue
        if e.role == "p_def" or e.role == "p_def_candidate":
            pts_pdef.append((s, w))
            continue
        if abs(s - s_def) <= threshold:
            pts_kept.append((s, w))
            labels_kept.append(e.summary)
        else:
            pts_dropped.append((s, w))
            labels_dropped.append(e.summary)

    if pts_kept:
        xs, ys = zip(*pts_kept)
        ax.scatter(xs, ys, c="tab:green", marker="D", s=80, label=f"kept ({len(pts_kept)})",
                   edgecolors="black", linewidth=0.5)
    if pts_dropped:
        xs, ys = zip(*pts_dropped)
        ax.scatter(xs, ys, c="tab:gray", marker="x", s=60, label=f"dropped ({len(pts_dropped)})")
    if pts_pdef:
        xs, ys = zip(*pts_pdef)
        ax.scatter(xs, ys, c="black", marker="*", s=240, label="p_def",
                   edgecolors="white", linewidth=1)

    # Mark the s_def threshold band
    ax.axvspan(s_def - threshold, s_def + threshold, alpha=0.08, color="tab:green",
               label=f"|s - s_def| ≤ {threshold}")
    ax.axvline(s_def, color="black", linestyle=":", alpha=0.5)

    ax.set_xlabel("strong (s_mean)")
    ax.set_ylabel("weak under W_0 (rubric)")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
