"""Pick k well-spread points out of a 2D set.

Used to subselect attacks from a Pareto front for the blue team's loss (paper Appendix G:
"2 points with the highest w and 3 more sparsely sampled from the rest"). Greedy
farthest-point seeding on range-normalized coordinates, then a local swap pass that
maximizes the minimum pairwise distance. The two extremes in x are always kept.
"""

import itertools

import numpy as np


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def select_k_max_min_distance(points_2d, k: int) -> list[int]:
    """Indices of k points spread across points_2d. Returns all of them if n <= k."""
    points_2d = np.asarray(points_2d, dtype=float)
    n = len(points_2d)
    if n <= k:
        return list(range(n))

    mins = points_2d.min(axis=0)
    ranges = points_2d.max(axis=0) - mins
    ranges[ranges < 1e-12] = 1.0
    norm = (points_2d - mins) / ranges
    dist = _pairwise_distances(norm)

    # Seed with the x-extremes, then greedily add the point farthest from the selection.
    sorted_by_x = np.argsort(norm[:, 0])
    selected = [int(sorted_by_x[0]), int(sorted_by_x[-1])]
    for _ in range(k - 2):
        min_dists = dist[selected].min(axis=0)
        for s in selected:
            min_dists[s] = -1
        selected.append(int(np.argmax(min_dists)))

    def min_pairwise(sel):
        if len(sel) < 2:
            return 0.0
        return min(dist[i, j] for i, j in itertools.combinations(sel, 2))

    protected = {0, 1}   # positions of the two extremes
    improved = True
    while improved:
        improved = False
        current_min = min_pairwise(selected)
        for i_idx in range(len(selected)):
            if i_idx in protected:
                continue
            for candidate in range(n):
                if candidate in selected:
                    continue
                trial = selected.copy()
                trial[i_idx] = candidate
                trial_min = min_pairwise(trial)
                if trial_min > current_min + 1e-10:
                    selected, current_min, improved = trial, trial_min, True
                    break
            if improved:
                break

    return selected
