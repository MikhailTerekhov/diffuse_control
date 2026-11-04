"""Tests for island-based program database with MAP-Elites + archive."""

from red_team.islands import (
    IslandConfig,
    IslandManager,
    MAPElitesGrid,
    FeatureBounds,
    _raw_features,
    _combined_score,
    _objectives,
)


class TestRawFeatures:
    def test_sandbagging(self):
        f0, f1 = _raw_features(0.8, 0.2, 1000, "sandbagging")
        assert abs(f0 - 1.0) < 1e-6
        assert f1 == 1000.0

    def test_strong_weak(self):
        f0, f1 = _raw_features(0.8, 0.2, 1000, "strong_weak")
        assert abs(f0 - (-0.6)) < 1e-6


class TestObjectives:
    def test_sandbagging(self):
        o = _objectives(0.8, 0.2, "sandbagging")
        assert abs(o[0] - (-0.2)) < 1e-6  # -strong
        assert abs(o[1] - 0.8) < 1e-6     # weak

    def test_strong_weak(self):
        o = _objectives(0.8, 0.2, "strong_weak")
        assert abs(o[0] - 0.2) < 1e-6
        assert abs(o[1] - 0.8) < 1e-6


class TestCombinedScore:
    def test_sandbagging(self):
        assert abs(_combined_score(0.8, 0.2, "sandbagging") - 0.8) < 1e-6

    def test_strong_weak(self):
        assert abs(_combined_score(0.8, 0.2, "strong_weak") - 0.5) < 1e-6


class TestFeatureBounds:
    def test_initial(self):
        b = FeatureBounds(2)
        assert b.n_observed == 0

    def test_update(self):
        b = FeatureBounds(2)
        b.update((0.5, 100.0))
        b.update((1.5, 500.0))
        assert b.mins == [0.5, 100.0]
        assert b.maxs == [1.5, 500.0]

    def test_bin_features(self):
        b = FeatureBounds(2)
        b.update((0.0, 0.0))
        b.update((1.0, 1000.0))
        # Midpoint
        bins = b.bin_features((0.5, 500.0), 4)
        assert bins == (2, 2)
        # Min
        bins = b.bin_features((0.0, 0.0), 4)
        assert bins == (0, 0)

    def test_update_expands_range(self):
        b = FeatureBounds(2)
        b.update((0.0, 0.0))
        b.update((1.0, 1000.0))
        b.update((2.0, 1000.0))
        assert b.maxs[0] == 2.0


class TestMAPElitesGrid:
    def _make_bounds(self, points):
        b = FeatureBounds(2)
        for w, s, l in points:
            b.update(_raw_features(w, s, l, "sandbagging"))
        return b

    def test_insert_empty(self):
        grid = MAPElitesGrid(4, "sandbagging")
        b = self._make_bounds([(0.8, 0.2, 1000)])
        evicted = grid.try_insert("t1", 0.8, 0.2, 1000, b)
        assert grid.size == 1
        assert evicted is None

    def test_different_features_different_cells(self):
        grid = MAPElitesGrid(4, "sandbagging")
        b = self._make_bounds([(0.3, 0.1, 100), (0.9, 0.9, 10000)])
        grid.try_insert("t1", 0.3, 0.1, 100, b)
        grid.try_insert("t2", 0.9, 0.9, 10000, b)
        assert grid.size == 2

    def test_better_replaces_and_returns_evicted(self):
        grid = MAPElitesGrid(4, "sandbagging")
        b = self._make_bounds([(0.5, 0.5, 1000), (0.8, 0.2, 1000)])
        grid.try_insert("t1", 0.5, 0.5, 1000, b)
        evicted = grid.try_insert("t2", 0.8, 0.2, 1000, b)
        assert "t2" in grid.cells.values()
        assert evicted == "t1"

    def test_worse_not_inserted(self):
        grid = MAPElitesGrid(4, "sandbagging")
        b = self._make_bounds([(0.8, 0.2, 1000), (0.6, 0.4, 1000)])
        grid.try_insert("t1", 0.8, 0.2, 1000, b)
        evicted = grid.try_insert("t2", 0.6, 0.4, 1000, b)
        assert "t1" in grid.cells.values()
        assert evicted is None

    def test_remove(self):
        grid = MAPElitesGrid(4, "sandbagging")
        b = self._make_bounds([(0.8, 0.2, 1000)])
        grid.try_insert("t1", 0.8, 0.2, 1000, b)
        grid.remove("t1")
        assert grid.size == 0


class TestIslandManager:
    def test_init(self):
        mgr = IslandManager(IslandConfig(n_islands=2, feature_bins=4), n_rungs=3)
        assert len(mgr.grids) == 2
        assert len(mgr.island_pools) == 2

    def test_insert_and_sample(self):
        mgr = IslandManager(IslandConfig(n_islands=2, feature_bins=4), n_rungs=3)
        mgr.insert(0, 0, "t1", 0.8, 0.2, 1000)
        # Should be able to sample from island 0
        found = False
        for _ in range(20):
            tid, mode = mgr.sample_parent(0)
            if tid == "t1":
                found = True
                assert mode in ("exploration", "exploitation", "random")
                break
        assert found

    def test_pool_keeps_evicted(self):
        """When MAP-Elites evicts a trial, it stays in the island pool."""
        mgr = IslandManager(IslandConfig(n_islands=1, feature_bins=4), n_rungs=1)
        mgr.insert(0, 0, "t1", 0.5, 0.5, 1000)
        mgr.insert(0, 0, "t2", 0.8, 0.2, 1000)  # same cell, better
        # t1 evicted from grid but still in pool
        assert "t1" in mgr.island_pools[0]
        assert "t2" in mgr.island_pools[0]

    def test_pruning_uses_pareto(self):
        """Population pruning removes worst by Pareto ranking."""
        cfg = IslandConfig(n_islands=1, feature_bins=4, population_size=3)
        mgr = IslandManager(cfg, n_rungs=1)
        # Add 4 trials, should prune to 3
        mgr.insert(0, 0, "good1", 0.9, 0.1, 1000)  # PF
        mgr.insert(0, 0, "good2", 0.5, 0.05, 1000)  # PF
        mgr.insert(0, 0, "ok", 0.7, 0.2, 2000)
        mgr.insert(0, 0, "bad", 0.3, 0.8, 1500)      # dominated
        assert len(mgr.island_pools[0]) == 3
        assert "bad" not in mgr.island_pools[0]

    def test_archive_from_highest_rung(self):
        from red_team.scheduler import RungState
        mgr = IslandManager(IslandConfig(n_islands=1, archive_size=2), n_rungs=3)
        mgr.insert(0, 0, "r0_trial", 0.9, 0.1, 1000)
        mgr.insert(0, 2, "r2_trial", 0.7, 0.3, 1000)
        # Simulate completed lists from scheduler
        rungs = [RungState(0, 8), RungState(1, 32), RungState(2, 128)]
        rungs[0].completed = ["r0_trial"]
        rungs[2].completed = ["r2_trial"]
        mgr.set_completed_lists(rungs)
        mgr._update_archive()
        # Archive should prefer rung 2
        assert "r2_trial" in mgr.archive

    def test_three_way_sampling(self):
        """All three sampling modes should work."""
        cfg = IslandConfig(n_islands=2, feature_bins=4, archive_size=5)
        mgr = IslandManager(cfg, n_rungs=2)
        mgr.insert(0, 0, "t1", 0.8, 0.2, 1000)
        mgr.insert(0, 1, "t2", 0.7, 0.3, 2000)
        mgr.insert(1, 0, "t3", 0.6, 0.4, 1500)
        modes_seen = set()
        for _ in range(100):
            tid, mode = mgr.sample_parent(0)
            assert tid is not None
            modes_seen.add(mode)
        # Should hit at least 2 of the 3 modes in 100 tries
        assert len(modes_seen) >= 2

    def test_promote_updates_rung(self):
        mgr = IslandManager(IslandConfig(n_islands=1, feature_bins=4), n_rungs=3)
        mgr.insert(0, 0, "t1", 0.8, 0.2, 1000)
        mgr.promote(0, 1, "t1", 0.8, 0.2, 1000)
        assert mgr._all_records["t1"].rung == 1

    def test_round_robin(self):
        mgr = IslandManager(IslandConfig(n_islands=3, feature_bins=4), n_rungs=2)
        islands = [mgr.next_island() for _ in range(6)]
        assert islands == [0, 1, 2, 0, 1, 2]

    def test_summary(self):
        mgr = IslandManager(IslandConfig(n_islands=2, feature_bins=4), n_rungs=3)
        mgr.insert(0, 0, "t1", 0.8, 0.2, 1000)
        mgr.insert(1, 1, "t2", 0.7, 0.3, 2000)
        s = mgr.summary()
        assert s["islands"][0]["pool_size"] == 1
        assert s["islands"][1]["pool_size"] == 1
        assert s["total_records"] == 2
        assert s["bounds"] is not None
