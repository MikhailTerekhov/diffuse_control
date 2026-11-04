"""Tests for non-dominated sorting and epsilon-net ranking."""

import numpy as np
import pytest

from red_team.pareto import (
    pareto_efficient,
    epsilon_net_order,
    non_dominated_sort,
    moasha_rank,
)


class TestParetoEfficient:
    def test_simple_2d(self):
        # (0,3) dominates nothing uniquely, (3,0) dominates nothing uniquely
        # (1,2) and (2,1) are non-dominated, (0,0) is dominated by all
        obj = np.array([
            [3, 0],  # 0: non-dominated
            [0, 3],  # 1: non-dominated
            [2, 1],  # 2: non-dominated
            [1, 2],  # 3: non-dominated
            [0, 0],  # 4: dominated by all above
        ])
        mask = pareto_efficient(obj)
        assert mask[0] and mask[1] and mask[2] and mask[3]
        assert not mask[4]

    def test_all_non_dominated(self):
        obj = np.array([[1, 3], [2, 2], [3, 1]])
        mask = pareto_efficient(obj)
        assert all(mask)

    def test_all_dominated_by_one(self):
        obj = np.array([[5, 5], [1, 1], [2, 2], [3, 3]])
        mask = pareto_efficient(obj)
        assert mask[0]
        assert not mask[1] and not mask[2] and not mask[3]

    def test_single_point(self):
        obj = np.array([[1, 2]])
        mask = pareto_efficient(obj)
        assert mask[0]

    def test_identical_points(self):
        obj = np.array([[1, 1], [1, 1], [1, 1]])
        mask = pareto_efficient(obj)
        # No point strictly dominates another if they're equal
        assert all(mask)


class TestNonDominatedSort:
    def test_three_fronts(self):
        obj = np.array([
            [3, 3],  # 0: front 0
            [1, 1],  # 1: front 2
            [2, 2],  # 2: front 1
            [4, 0],  # 3: front 0
            [0, 4],  # 4: front 0
        ])
        fronts = non_dominated_sort(obj)
        assert len(fronts) == 3
        assert set(fronts[0]) == {0, 3, 4}
        assert set(fronts[1]) == {2}
        assert set(fronts[2]) == {1}

    def test_single_front(self):
        obj = np.array([[1, 3], [2, 2], [3, 1]])
        fronts = non_dominated_sort(obj)
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1, 2}

    def test_empty(self):
        obj = np.array([]).reshape(0, 2)
        fronts = non_dominated_sort(obj)
        assert fronts == []


class TestEpsilonNetOrder:
    def test_spreads_points(self):
        # 5 points on a line in 2D
        obj = np.array([
            [0.0, 1.0],
            [0.25, 0.75],
            [0.5, 0.5],
            [0.75, 0.25],
            [1.0, 0.0],
        ])
        indices = list(range(5))
        ordered = epsilon_net_order(obj, indices)
        # First point should have highest dim-0 value (index 4)
        assert ordered[0] == 4
        # Second should be farthest from first → index 0
        assert ordered[1] == 0
        assert len(ordered) == 5
        assert set(ordered) == set(indices)

    def test_single_point(self):
        obj = np.array([[1.0, 2.0]])
        assert epsilon_net_order(obj, [0]) == [0]

    def test_two_points(self):
        obj = np.array([[0.0, 1.0], [1.0, 0.0]])
        ordered = epsilon_net_order(obj, [0, 1])
        assert len(ordered) == 2
        # Point with highest dim-0 is index 1
        assert ordered[0] == 1


class TestMoashaRank:
    def test_selects_top_k_from_front(self):
        obj = np.array([
            [3, 0],  # front 0
            [0, 3],  # front 0
            [2, 1],  # front 0
            [1, 1],  # front 1 (dominated by [2,1] and [3,0] etc.)
            [0, 0],  # front 2
        ])
        # k=2: should pick 2 from front 0
        top = moasha_rank(obj, 2)
        assert len(top) == 2
        assert all(i in {0, 1, 2} for i in top)

    def test_k_larger_than_n(self):
        obj = np.array([[1, 2], [2, 1]])
        top = moasha_rank(obj, 5)
        assert set(top) == {0, 1}

    def test_k_zero(self):
        obj = np.array([[1, 2], [2, 1]])
        assert moasha_rank(obj, 0) == []

    def test_spills_to_second_front(self):
        obj = np.array([
            [3, 0],  # front 0
            [0, 3],  # front 0
            [1, 1],  # front 1
            [0, 0],  # front 2
        ])
        # k=3: front 0 has 2, need 1 more from front 1
        top = moasha_rank(obj, 3)
        assert len(top) == 3
        assert 0 in top and 1 in top  # front 0
        assert 2 in top  # front 1

    def test_3d_objectives(self):
        obj = np.array([
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [1, 1, 1],
        ])
        top = moasha_rank(obj, 2)
        assert len(top) == 2
        # All of first 3 are non-dominated; index 3 is dominated
        assert all(i in {0, 1, 2} for i in top)
