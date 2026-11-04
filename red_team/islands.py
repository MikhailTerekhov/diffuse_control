"""Island-based program database with MAP-Elites + archive for prompt diversity.

Three-way sampling (like OpenEvolve):
  - Exploration (30%): random from the island's program pool
  - Exploitation (60%): from the global archive (highest-rung Pareto front)
  - Random (10%): any program globally

MAP-Elites grids handle cell replacement (eviction policy).
Population pruning uses NSGA-II Pareto ranking + epsilon-net.
"""

import logging
import random
from dataclasses import dataclass

import numpy as np

from red_team.pareto import non_dominated_sort, epsilon_net_order, moasha_rank

logger = logging.getLogger(__name__)


@dataclass
class IslandConfig:
    n_islands: int = 3
    feature_bins: int = 4
    mode: str = "sandbagging"
    migration_interval: int = 640
    migration_rate: float = 0.1
    population_size: int = 80  # per island
    archive_size: int = 16  # global
    exploration_ratio: float = 0.3
    exploitation_ratio: float = 0.6
    # remaining 0.1 = random


@dataclass
class TrialRecord:
    """Lightweight record of a trial's scores for the island database."""
    trial_id: str
    weak: float
    strong: float
    prompt_length: int
    island_id: int
    rung: int  # highest rung at which this was recorded


def _is_sandbagging_mode(mode: str) -> bool:
    return mode in ("sandbagging", "sandbagging_detector")


def _raw_features(weak: float, strong: float, prompt_length: int, mode: str) -> tuple[float, float]:
    if _is_sandbagging_mode(mode):
        feat0 = weak + strong
    else:
        feat0 = strong - weak
    return (feat0, float(prompt_length))


def _objectives(weak: float, strong: float, mode: str) -> tuple[float, float]:
    """Mode-specific objectives (both maximized)."""
    if _is_sandbagging_mode(mode):
        return (-strong, weak)
    else:
        return (strong, weak)


def _combined_score(weak: float, strong: float, mode: str) -> float:
    if _is_sandbagging_mode(mode):
        return ((1 - strong) + weak) / 2
    else:
        return (strong + weak) / 2


class FeatureBounds:
    def __init__(self, n_dims: int = 2):
        self.mins = [float("inf")] * n_dims
        self.maxs = [float("-inf")] * n_dims
        self.n_observed = 0

    def update(self, features: tuple[float, ...]):
        for i, v in enumerate(features):
            self.mins[i] = min(self.mins[i], v)
            self.maxs[i] = max(self.maxs[i], v)
        self.n_observed += 1

    def bin_features(self, features: tuple[float, ...], n_bins: int) -> tuple[int, ...]:
        return tuple(self._bin(i, v, n_bins) for i, v in enumerate(features))

    def _bin(self, dim: int, value: float, n_bins: int) -> int:
        lo, hi = self.mins[dim], self.maxs[dim]
        if hi <= lo:
            return 0
        frac = (value - lo) / (hi - lo)
        return min(n_bins - 1, max(0, int(frac * n_bins)))


class MAPElitesGrid:
    """MAP-Elites grid — one program per cell, best by combined score wins."""

    def __init__(self, feature_bins: int, mode: str):
        self.feature_bins = feature_bins
        self.mode = mode
        self.cells: dict[tuple[int, ...], str] = {}
        self.scores: dict[str, float] = {}
        self._trial_data: dict[str, tuple[float, float, int]] = {}

    def try_insert(self, trial_id: str, weak: float, strong: float,
                   prompt_length: int, bounds: FeatureBounds) -> str | None:
        """Insert trial. Returns evicted trial_id if a replacement happened, else None."""
        raw = _raw_features(weak, strong, prompt_length, self.mode)
        coords = bounds.bin_features(raw, self.feature_bins)
        score = _combined_score(weak, strong, self.mode)

        current = self.cells.get(coords)
        if current is None or score > self.scores.get(current, -1):
            evicted = None
            if current is not None and current != trial_id:
                evicted = current
                self.scores.pop(current, None)
                self._trial_data.pop(current, None)
            self.cells[coords] = trial_id
            self.scores[trial_id] = score
            self._trial_data[trial_id] = (weak, strong, prompt_length)
            return evicted
        return None  # not inserted, no eviction

    def rebin(self, bounds: FeatureBounds):
        old_data = dict(self._trial_data)
        self.cells.clear()
        self.scores.clear()
        self._trial_data.clear()
        for trial_id, (weak, strong, prompt_length) in old_data.items():
            self.try_insert(trial_id, weak, strong, prompt_length, bounds)

    def remove(self, trial_id: str):
        to_remove = [c for c, tid in self.cells.items() if tid == trial_id]
        for c in to_remove:
            del self.cells[c]
        self.scores.pop(trial_id, None)
        self._trial_data.pop(trial_id, None)

    def mark_all_stale(self):
        """Set all elite scores to -inf so the next arrival to each cell wins."""
        for tid in list(self.scores.keys()):
            self.scores[tid] = float("-inf")

    def cell_details(self) -> list[dict]:
        details = []
        for coords, tid in self.cells.items():
            w, s, plen = self._trial_data.get(tid, (0, 0, 0))
            details.append({
                "cell": list(coords), "trial_id": tid,
                "weak": w, "strong": s,
                "combined": self.scores.get(tid, 0),
                "prompt_length": plen,
            })
        return sorted(details, key=lambda d: d["cell"])

    @property
    def size(self) -> int:
        return len(self.cells)


class IslandManager:
    """Manages islands with program databases, MAP-Elites grids, and global archive."""

    def __init__(self, config: IslandConfig, n_rungs: int):
        self.config = config
        self.n_rungs = n_rungs
        self._next_island = 0
        self.bounds = FeatureBounds(n_dims=2)

        # Per-island: program pool (like OpenEvolve's self.islands[i])
        self.island_pools: list[dict[str, TrialRecord]] = [{} for _ in range(config.n_islands)]

        # Per-island per-rung: MAP-Elites grids (for cell replacement)
        self.grids: list[list[MAPElitesGrid]] = [
            [MAPElitesGrid(config.feature_bins, config.mode) for _ in range(n_rungs)]
            for _ in range(config.n_islands)
        ]

        # Global archive: top programs from highest rung
        self.archive: list[str] = []  # trial_ids, ordered by Pareto rank
        self._all_records: dict[str, TrialRecord] = {}  # global trial_id -> record

    def insert(self, island_id: int, rung: int, trial_id: str,
               weak: float, strong: float, prompt_length: int) -> None:
        """Add/update a trial in the database after rung completion."""
        raw = _raw_features(weak, strong, prompt_length, self.config.mode)
        self.bounds.update(raw)
        self._rebin_all()

        record = TrialRecord(trial_id, weak, strong, prompt_length, island_id, rung)
        self._all_records[trial_id] = record

        # Add to island pool
        self.island_pools[island_id][trial_id] = record

        # Update MAP-Elites grid (may evict another trial from the grid, but NOT from the pool)
        self.grids[island_id][rung].try_insert(
            trial_id, weak, strong, prompt_length, self.bounds)

        # Prune island if over population limit
        self._prune_island(island_id)

        # Update global archive
        self._update_archive()

    def _prune_island(self, island_id: int):
        """Prune island pool using NSGA-II Pareto ranking if over population limit."""
        pool = self.island_pools[island_id]
        if len(pool) <= self.config.population_size:
            return

        records = list(pool.values())
        objectives = np.array([_objectives(r.weak, r.strong, self.config.mode) for r in records])

        # Keep top population_size by Pareto rank + epsilon-net
        keep_ids = set()
        keep_n = self.config.population_size
        ranked = moasha_rank(objectives, keep_n)
        for idx in ranked:
            keep_ids.add(records[idx].trial_id)

        # Remove pruned trials from pool (but keep in _all_records for archive/random)
        to_remove = [tid for tid in pool if tid not in keep_ids]
        for tid in to_remove:
            del pool[tid]

        logger.debug(f"Pruned island {island_id}: {len(to_remove)} removed, {len(pool)} remaining")

    def _update_archive(self):
        """Rebuild global archive from highest-rung completed programs.

        Uses _completed_by_rung (set by set_completed_lists) to ensure
        only fully-evaluated trials enter the archive.
        """
        if not hasattr(self, "_completed_by_rung"):
            return

        # Find highest rung with completed programs
        for rung in range(self.n_rungs - 1, -1, -1):
            completed_ids = self._completed_by_rung.get(rung, [])
            candidates = [self._all_records[tid] for tid in completed_ids
                          if tid in self._all_records]
            if not candidates:
                continue

            objectives = np.array([_objectives(r.weak, r.strong, self.config.mode) for r in candidates])
            n_keep = min(self.config.archive_size, len(candidates))
            ranked = moasha_rank(objectives, n_keep)
            self.archive = [candidates[i].trial_id for i in ranked]
            return

    def set_completed_lists(self, rungs):
        """Pass scheduler's rung completion lists for archive building."""
        self._completed_by_rung = {r.rung_index: r.completed for r in rungs}

    def _rebin_all(self):
        for island_grids in self.grids:
            for grid in island_grids:
                if grid.size > 0:
                    grid.rebin(self.bounds)

    def promote(self, island_id: int, new_rung: int,
                trial_id: str, weak: float, strong: float, prompt_length: int):
        self.grids[island_id][new_rung].try_insert(
            trial_id, weak, strong, prompt_length, self.bounds)
        # Update the record's rung
        if trial_id in self._all_records:
            self._all_records[trial_id].rung = new_rung
        if trial_id in self.island_pools[island_id]:
            self.island_pools[island_id][trial_id].rung = new_rung

    def sample_parent(self, island_id: int) -> tuple[str | None, str]:
        """Three-way sampling: exploration / exploitation / random.

        Returns (trial_id, sampling_mode) where mode is 'exploration', 'exploitation', or 'random'.
        """
        rand_val = random.random()

        if rand_val < self.config.exploration_ratio:
            return self._sample_exploration(island_id), "exploration"
        elif rand_val < self.config.exploration_ratio + self.config.exploitation_ratio:
            return self._sample_exploitation(island_id), "exploitation"
        else:
            return self._sample_random(), "random"

    def _sample_exploration(self, island_id: int) -> str | None:
        """Random from the island's program pool."""
        pool = self.island_pools[island_id]
        if not pool:
            # Fall back to any island
            for p in self.island_pools:
                if p:
                    return random.choice(list(p.keys()))
            return None
        return random.choice(list(pool.keys()))

    def _sample_exploitation(self, island_id: int) -> str | None:
        """From the global archive, preferring current island."""
        if not self.archive:
            return self._sample_exploration(island_id)

        # Prefer archive members on the same island
        same_island = [tid for tid in self.archive if tid in self.island_pools[island_id]]
        if same_island:
            return random.choice(same_island)
        return random.choice(self.archive)

    def _sample_random(self) -> str | None:
        """Any program from the global pool."""
        if not self._all_records:
            return None
        return random.choice(list(self._all_records.keys()))

    def maybe_migrate(self, n_generated: int) -> list[tuple[str, int, int]]:
        if self.config.n_islands < 2:
            return []
        if n_generated % self.config.migration_interval != 0 or n_generated == 0:
            return []
        return self._migrate()

    def _migrate(self) -> list[tuple[str, int, int]]:
        """Migrate Pareto-front trials between islands (ring topology)."""
        migrated = []
        for i in range(self.config.n_islands):
            pool = self.island_pools[i]
            if len(pool) < 2:
                continue

            records = list(pool.values())
            objectives = np.array([_objectives(r.weak, r.strong, self.config.mode) for r in records])

            n_migrate = max(1, int(len(records) * self.config.migration_rate))
            ranked = moasha_rank(objectives, n_migrate)

            targets = [(i + 1) % self.config.n_islands, (i - 1) % self.config.n_islands]

            for idx in ranked:
                record = records[idx]
                for target in targets:
                    if record.trial_id in self.island_pools[target]:
                        continue
                    # Add to target island pool and grid
                    self.island_pools[target][record.trial_id] = record
                    self.grids[target][record.rung].try_insert(
                        record.trial_id, record.weak, record.strong,
                        record.prompt_length, self.bounds)
                    migrated.append((record.trial_id, i, target))

        if migrated:
            logger.info(f"Migration: {len(migrated)} trials migrated between islands")
        return migrated

    def next_island(self) -> int:
        island = self._next_island
        self._next_island = (self._next_island + 1) % self.config.n_islands
        return island

    def mark_all_stale(self):
        """Mark all MAP-Elites cell scores as stale across all islands & rungs.

        The cells keep their current trial_ids (so MAP-Elites can still be sampled
        from), but the next prompt arriving at each cell will overwrite
        unconditionally. After that first overwrite, normal Pareto-dominance applies.
        """
        for island_grids in self.grids:
            for grid in island_grids:
                grid.mark_all_stale()

    def summary(self) -> dict:
        islands = []
        for i in range(self.config.n_islands):
            rung_grids = []
            for r in range(self.n_rungs):
                grid = self.grids[i][r]
                rung_grids.append({
                    "rung": r, "size": grid.size, "cells": grid.cell_details(),
                })
            rung_sizes = [g["size"] for g in rung_grids]
            islands.append({
                "island": i, "rung_sizes": rung_sizes, "grids": rung_grids,
                "pool_size": len(self.island_pools[i]),
            })
        return {
            "islands": islands,
            "archive_size": len(self.archive),
            "total_records": len(self._all_records),
            "bounds": {
                "feat0": [self.bounds.mins[0], self.bounds.maxs[0]],
                "feat1": [self.bounds.mins[1], self.bounds.maxs[1]],
            } if self.bounds.n_observed > 0 else None,
            "mode": self.config.mode,
            "feature_bins": self.config.feature_bins,
        }
