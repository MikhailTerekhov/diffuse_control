"""Non-dominated sorting and epsilon-net ranking for MO-ASHA.

Implements the multi-objective priority ranking from:
  "Multi-Objective Asynchronous Successive Halving" (Schmucker et al., 2021)

All objectives are MAXIMIZED. A point is dominated if another point is
>= on all objectives and > on at least one.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def pareto_efficient(objectives: np.ndarray) -> np.ndarray:
    """Find indices of Pareto-efficient (non-dominated) points.

    Args:
        objectives: (N, D) array, all objectives maximized.

    Returns:
        Boolean mask of length N, True for non-dominated points.
    """
    n = len(objectives)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # Check if any other point dominates i
        for j in range(n):
            if i == j or not is_efficient[j]:
                continue
            # j dominates i if j >= i on all and j > i on at least one
            if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                is_efficient[i] = False
                break
    return is_efficient


def epsilon_net_order(objectives: np.ndarray, indices: list[int]) -> list[int]:
    """Greedy farthest-first traversal for diversity within a Pareto front.

    Orders points so that earlier points provide better coverage of the
    objective space. This is the epsilon-net procedure from the MO-ASHA paper.

    Args:
        objectives: (N, D) array of ALL points' objectives.
        indices: Subset of row indices to order (e.g., a single Pareto front).

    Returns:
        Reordered list of indices (same elements, different order).
    """
    if len(indices) <= 1:
        return list(indices)

    # Extract and normalize objectives for the subset
    sub_obj = objectives[indices]
    mins = sub_obj.min(axis=0)
    maxs = sub_obj.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-12] = 1.0
    norm = (sub_obj - mins) / ranges

    # Start with the point that has the highest value in dimension 0
    # (following SyneTune's convention)
    selected = [int(np.argmax(norm[:, 0]))]
    min_dists = np.full(len(indices), np.inf)

    for _ in range(len(indices) - 1):
        last = norm[selected[-1]]
        dists_to_last = np.linalg.norm(norm - last, axis=1)
        min_dists = np.minimum(min_dists, dists_to_last)
        # Exclude already selected
        min_dists_masked = min_dists.copy()
        for s in selected:
            min_dists_masked[s] = -1.0
        next_idx = int(np.argmax(min_dists_masked))
        selected.append(next_idx)

    return [indices[s] for s in selected]


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Non-dominated sorting into successive Pareto fronts.

    Args:
        objectives: (N, D) array, all objectives maximized.

    Returns:
        List of fronts. Front 0 = Pareto-optimal, Front 1 = next layer, etc.
        Each front is a list of row indices.
    """
    n = len(objectives)
    if n == 0:
        return []

    remaining = set(range(n))
    fronts = []

    while remaining:
        remaining_list = sorted(remaining)
        sub_obj = objectives[remaining_list]
        mask = pareto_efficient(sub_obj)
        front = [remaining_list[i] for i in range(len(remaining_list)) if mask[i]]
        if not front:
            # Safety: shouldn't happen, but avoid infinite loop
            logger.warning(f"non_dominated_sort: pareto_efficient returned empty front "
                           f"for {len(remaining)} remaining points — dumping all as one front")
            front = list(remaining)
        fronts.append(front)
        remaining -= set(front)

    return fronts


def moasha_rank(
    objectives: np.ndarray,
    k: int,
) -> list[int]:
    """MO-ASHA ranking: non-dominated sort + epsilon-net diversity.

    Returns the top-k indices, ordered by (front rank, then diversity within front).
    Used for promotion decisions: promote the top k = floor(n / eta).

    Args:
        objectives: (N, D) array, all objectives maximized.
        k: Number of top indices to return.

    Returns:
        List of up to k indices, best first.
    """
    if k <= 0:
        return []
    if len(objectives) <= k:
        # All qualify — still order them by front + diversity
        fronts = non_dominated_sort(objectives)
        result = []
        for front in fronts:
            ordered = epsilon_net_order(objectives, front)
            result.extend(ordered)
        return result

    fronts = non_dominated_sort(objectives)
    selected: list[int] = []
    remaining_k = k

    for front in fronts:
        if remaining_k <= 0:
            break
        ordered = epsilon_net_order(objectives, front)
        take = min(len(ordered), remaining_k)
        selected.extend(ordered[:take])
        remaining_k -= take

    return selected
