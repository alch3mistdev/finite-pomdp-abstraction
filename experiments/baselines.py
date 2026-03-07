"""Baseline partition methods for comparison with epsilon-quotient."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .clustering import cluster_complete_linkage
from .pomdp_core import (
    DeterministicFSC,
    FinitePOMDP,
    History,
    all_histories,
    belief_after_history,
)
from .quotient import PartitionResult


def truncation_partition(
    num_observations: int,
    horizon: int,
    depth: int,
) -> PartitionResult:
    """Naive history truncation: merge histories sharing last `depth` observations.

    Histories are grouped by their suffix of length min(len(h), depth).
    """
    histories_by_d = all_histories(num_observations, horizon)

    classes_by_depth: Dict[int, Tuple[int, ...]] = {}
    class_histories: Dict[int, Tuple[History, ...]] = {}
    class_depth: Dict[int, int] = {}
    history_to_class: Dict[History, int] = {}

    next_class_id = 0

    for d in range(horizon + 1):
        hs = histories_by_d[d]
        # Group by suffix of length min(d, depth)
        suffix_len = min(d, depth)
        groups: Dict[Tuple[int, ...], List[History]] = {}
        for h in hs:
            suffix = h[-suffix_len:] if suffix_len > 0 else ()
            groups.setdefault(suffix, []).append(h)

        depth_ids: List[int] = []
        for suffix in sorted(groups.keys()):
            cid = next_class_id
            next_class_id += 1
            members = tuple(groups[suffix])
            class_histories[cid] = members
            class_depth[cid] = d
            for h in members:
                history_to_class[h] = cid
            depth_ids.append(cid)
        classes_by_depth[d] = tuple(depth_ids)

    return PartitionResult(
        horizon=horizon,
        epsilon=float("nan"),
        classes_by_depth=classes_by_depth,
        class_histories=class_histories,
        class_depth=class_depth,
        history_to_class=history_to_class,
    )


def random_partition(
    num_observations: int,
    horizon: int,
    k_per_depth: Dict[int, int],
    seed: int = 42,
) -> PartitionResult:
    """Randomly assign histories at each depth to k clusters.

    Parameters
    ----------
    k_per_depth : dict mapping depth -> number of clusters at that depth.
        If a depth is missing, histories at that depth each get their own class.
    """
    rng = np.random.default_rng(seed)
    histories_by_d = all_histories(num_observations, horizon)

    classes_by_depth: Dict[int, Tuple[int, ...]] = {}
    class_histories: Dict[int, Tuple[History, ...]] = {}
    class_depth: Dict[int, int] = {}
    history_to_class: Dict[History, int] = {}

    next_class_id = 0

    for d in range(horizon + 1):
        hs = histories_by_d[d]
        k = k_per_depth.get(d, len(hs))
        k = min(k, len(hs))
        k = max(k, 1)

        assignments = rng.integers(0, k, size=len(hs))
        groups: Dict[int, List[History]] = {}
        for idx, h in enumerate(hs):
            groups.setdefault(int(assignments[idx]), []).append(h)

        depth_ids: List[int] = []
        for g in sorted(groups.keys()):
            cid = next_class_id
            next_class_id += 1
            members = tuple(groups[g])
            class_histories[cid] = members
            class_depth[cid] = d
            for h in members:
                history_to_class[h] = cid
            depth_ids.append(cid)
        classes_by_depth[d] = tuple(depth_ids)

    return PartitionResult(
        horizon=horizon,
        epsilon=float("nan"),
        classes_by_depth=classes_by_depth,
        class_histories=class_histories,
        class_depth=class_depth,
        history_to_class=history_to_class,
    )


def belief_distance_partition(
    pomdp: FinitePOMDP,
    policy: DeterministicFSC,
    horizon: int,
    epsilon: float,
) -> PartitionResult:
    """Cluster histories by TV distance between belief states (Ferns-style).

    Uses a single representative policy to compute beliefs, then applies
    complete-linkage clustering on the belief TV distance matrix.
    """
    histories_by_d = all_histories(num_observations=pomdp.num_observations, horizon=horizon)

    classes_by_depth: Dict[int, Tuple[int, ...]] = {}
    class_histories: Dict[int, Tuple[History, ...]] = {}
    class_depth: Dict[int, int] = {}
    history_to_class: Dict[History, int] = {}

    next_class_id = 0

    for d in range(horizon + 1):
        hs = histories_by_d[d]
        n = len(hs)

        # Compute beliefs
        beliefs = []
        for h in hs:
            beliefs.append(belief_after_history(pomdp=pomdp, policy=policy, history=h))

        # Compute pairwise TV distance
        dist_mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                tv = 0.5 * float(np.sum(np.abs(beliefs[i] - beliefs[j])))
                dist_mat[i, j] = tv
                dist_mat[j, i] = tv

        # Complete-linkage clustering
        clusters = cluster_complete_linkage(dist_mat, epsilon)

        depth_ids: List[int] = []
        for cluster in clusters:
            cid = next_class_id
            next_class_id += 1
            members = tuple(hs[i] for i in cluster)
            class_histories[cid] = members
            class_depth[cid] = d
            for h in members:
                history_to_class[h] = cid
            depth_ids.append(cid)
        classes_by_depth[d] = tuple(depth_ids)

    return PartitionResult(
        horizon=horizon,
        epsilon=epsilon,
        classes_by_depth=classes_by_depth,
        class_histories=class_histories,
        class_depth=class_depth,
        history_to_class=history_to_class,
    )


def _wasserstein_state(
    p: np.ndarray,
    q: np.ndarray,
    d_state: np.ndarray,
) -> float:
    """W1 between two distributions over a finite state set with ground metric d_state."""
    from scipy.optimize import linprog

    n = len(p)
    if n == 0:
        return 0.0
    c = d_state.ravel()
    # Row marginal constraints (sum over columns = p)
    a_eq = np.zeros((2 * n, n * n), dtype=float)
    for i in range(n):
        a_eq[i, i * n : (i + 1) * n] = 1.0
    # Column marginal constraints (sum over rows = q)
    for j in range(n):
        a_eq[n + j, j::n] = 1.0
    b_eq = np.concatenate([p, q])
    bounds = [(0.0, None)] * (n * n)
    res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Wasserstein LP failed in bisimulation baseline: {res.message}")
    return float(res.fun)


def bisimulation_metric_partition(
    pomdp: FinitePOMDP,
    horizon: int,
    epsilon: float,
    num_iterations: int = 10,
    discount: float = 0.9,
) -> PartitionResult:
    """Approximate bisimulation metric baseline (Ferns et al. 2004/2012).

    Computes an approximate bisimulation metric on states via fixed-point
    iteration, derives history-level distance as the Wasserstein distance
    between belief distributions under the state metric, and clusters with
    complete-linkage at threshold epsilon.
    """
    num_s = pomdp.num_states
    num_a = pomdp.num_actions

    # Iterative fixed-point for state-level bisimulation metric
    d_state = np.zeros((num_s, num_s), dtype=float)
    for _iteration in range(num_iterations):
        d_new = np.zeros((num_s, num_s), dtype=float)
        for s1 in range(num_s):
            for s2 in range(s1 + 1, num_s):
                max_over_actions = 0.0
                for a in range(num_a):
                    reward_diff = abs(float(pomdp.rewards[s1, a]) - float(pomdp.rewards[s2, a]))
                    w1_trans = _wasserstein_state(
                        pomdp.transition[s1, a, :],
                        pomdp.transition[s2, a, :],
                        d_state,
                    )
                    val = reward_diff + discount * w1_trans
                    max_over_actions = max(max_over_actions, val)
                d_new[s1, s2] = max_over_actions
                d_new[s2, s1] = max_over_actions
        d_state = d_new

    # Normalise to [0, 1]
    if d_state.max() > 0:
        d_state = d_state / d_state.max()

    # Derive history-level distances via belief-Wasserstein under the state metric
    from .fsc_enum import enumerate_deterministic_fscs

    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1,
        include_smaller=True,
    )
    policy = policies[0]

    histories_by_d = all_histories(num_observations=pomdp.num_observations, horizon=horizon)

    classes_by_depth: Dict[int, Tuple[int, ...]] = {}
    class_histories: Dict[int, Tuple[History, ...]] = {}
    class_depth: Dict[int, int] = {}
    history_to_class: Dict[History, int] = {}
    next_class_id = 0

    for d in range(horizon + 1):
        hs = histories_by_d[d]
        n = len(hs)

        beliefs = []
        for h in hs:
            beliefs.append(belief_after_history(pomdp=pomdp, policy=policy, history=h))

        # Pairwise Wasserstein distance between beliefs under state metric
        dist_mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                w1 = _wasserstein_state(beliefs[i], beliefs[j], d_state)
                dist_mat[i, j] = w1
                dist_mat[j, i] = w1

        clusters = cluster_complete_linkage(dist_mat, epsilon)

        depth_ids: List[int] = []
        for cluster in clusters:
            cid = next_class_id
            next_class_id += 1
            members = tuple(hs[i] for i in cluster)
            class_histories[cid] = members
            class_depth[cid] = d
            for h in members:
                history_to_class[h] = cid
            depth_ids.append(cid)
        classes_by_depth[d] = tuple(depth_ids)

    return PartitionResult(
        horizon=horizon,
        epsilon=epsilon,
        classes_by_depth=classes_by_depth,
        class_histories=class_histories,
        class_depth=class_depth,
        history_to_class=history_to_class,
    )
