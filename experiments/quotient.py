"""History partitioning and canonical quotient utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .clustering import cluster_complete_linkage
from .metrics import distribution_distance
from .pomdp_core import (
    DeterministicFSC,
    FinitePOMDP,
    History,
    ObsSeq,
    all_histories,
    belief_after_history,
    conditional_future_observation_distribution,
    trajectory_observation_distribution,
)


@dataclass(frozen=True)
class DistanceCache:
    horizon: int
    histories_by_depth: Dict[int, List[History]]
    max_distance_matrices: Dict[int, np.ndarray]


@dataclass(frozen=True)
class PartitionResult:
    horizon: int
    epsilon: float
    classes_by_depth: Dict[int, Tuple[int, ...]]
    class_histories: Dict[int, Tuple[History, ...]]
    class_depth: Dict[int, int]
    history_to_class: Dict[History, int]

    @property
    def num_classes_total(self) -> int:
        return len(self.class_histories)


def precompute_distance_cache(
    pomdp: FinitePOMDP,
    policies: Sequence[DeterministicFSC],
    horizon: int,
    distance_mode: str,
    d_obs: np.ndarray,
) -> DistanceCache:
    histories_by_depth = all_histories(num_observations=pomdp.num_observations, horizon=horizon)

    cond: List[Dict[History, Mapping[ObsSeq, float]]] = []
    for policy in policies:
        mapping: Dict[History, Mapping[ObsSeq, float]] = {}
        for depth, histories in histories_by_depth.items():
            for h in histories:
                mapping[h] = conditional_future_observation_distribution(
                    pomdp=pomdp,
                    policy=policy,
                    history=h,
                    horizon=horizon,
                )
        cond.append(mapping)

    max_distance_matrices: Dict[int, np.ndarray] = {}
    for depth, histories in histories_by_depth.items():
        n = len(histories)
        matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                dmax = 0.0
                hi = histories[i]
                hj = histories[j]
                for p_idx in range(len(policies)):
                    d = distribution_distance(
                        cond[p_idx][hi],
                        cond[p_idx][hj],
                        mode=distance_mode,
                        d_obs=d_obs,
                    )
                    if d > dmax:
                        dmax = d
                matrix[i, j] = dmax
                matrix[j, i] = dmax
        max_distance_matrices[depth] = matrix

    return DistanceCache(
        horizon=horizon,
        histories_by_depth=histories_by_depth,
        max_distance_matrices=max_distance_matrices,
    )


# Backward-compatible alias for internal imports (e.g. spectral.py)
_cluster_with_complete_linkage = cluster_complete_linkage


def compute_partition_from_cache(cache: DistanceCache, epsilon: float) -> PartitionResult:
    classes_by_depth: Dict[int, Tuple[int, ...]] = {}
    class_histories: Dict[int, Tuple[History, ...]] = {}
    class_depth: Dict[int, int] = {}
    history_to_class: Dict[History, int] = {}

    next_class_id = 0
    for depth in range(cache.horizon + 1):
        histories = cache.histories_by_depth[depth]
        mat = cache.max_distance_matrices[depth]

        clusters = _cluster_with_complete_linkage(mat, epsilon=epsilon)
        depth_ids: List[int] = []
        for cluster in clusters:
            cid = next_class_id
            next_class_id += 1
            members = tuple(histories[i] for i in cluster)
            class_histories[cid] = members
            class_depth[cid] = depth
            for h in members:
                history_to_class[h] = cid
            depth_ids.append(cid)
        classes_by_depth[depth] = tuple(depth_ids)

    return PartitionResult(
        horizon=cache.horizon,
        epsilon=epsilon,
        classes_by_depth=classes_by_depth,
        class_histories=class_histories,
        class_depth=class_depth,
        history_to_class=history_to_class,
    )


def compute_class_count_curve(
    cache: DistanceCache,
    eps_grid: Sequence[float],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    total_histories = sum(len(cache.histories_by_depth[d]) for d in range(cache.horizon + 1))
    for eps in eps_grid:
        part = compute_partition_from_cache(cache, epsilon=float(eps))
        count = part.num_classes_total
        rows.append(
            {
                "epsilon": float(eps),
                "class_count": float(count),
                "compression_ratio": float(count) / float(total_histories),
            }
        )
    return rows


def _class_canonical_beliefs(
    pomdp: FinitePOMDP,
    partition: PartitionResult,
    policy: DeterministicFSC,
) -> Dict[int, np.ndarray]:
    beliefs: Dict[int, np.ndarray] = {}
    for cid, members in partition.class_histories.items():
        bs = [belief_after_history(pomdp=pomdp, policy=policy, history=h) for h in members]
        b = np.mean(np.stack(bs, axis=0), axis=0)
        b = b / b.sum()
        beliefs[cid] = b
    return beliefs


def _representative_history(partition: PartitionResult, class_id: int) -> History:
    members = partition.class_histories[class_id]
    return sorted(members)[0]


def quotient_observation_sequence_distribution(
    pomdp: FinitePOMDP,
    partition: PartitionResult,
    policy: DeterministicFSC,
) -> Dict[ObsSeq, float]:
    """Generate P_Q(O^T) under canonical quotient and a deterministic FSC.

    This uses representative-history transitions for approximate merged classes.
    Supports both single-node and multi-node FSCs by tracking joint (class, node)
    state during the recursive rollout.
    """
    horizon = partition.horizon
    beliefs = _class_canonical_beliefs(pomdp=pomdp, partition=partition, policy=policy)

    @lru_cache(maxsize=None)
    def recurse(class_id: int, node: int) -> Dict[ObsSeq, float]:
        depth = partition.class_depth[class_id]
        if depth == horizon:
            return {(): 1.0}

        b = beliefs[class_id]
        action = policy.action_index(node)

        obs_probs = np.zeros(pomdp.num_observations, dtype=float)
        for obs in range(pomdp.num_observations):
            p_obs = 0.0
            for s in range(pomdp.num_states):
                for s_next in range(pomdp.num_states):
                    p_obs += b[s] * pomdp.transition[s, action, s_next] * pomdp.observation[s_next, action, obs]
            obs_probs[obs] = p_obs

        z = float(obs_probs.sum())
        if z > 0:
            obs_probs = obs_probs / z

        rep = _representative_history(partition, class_id)
        out: Dict[ObsSeq, float] = {}
        for obs in range(pomdp.num_observations):
            p_obs = obs_probs[obs]
            if p_obs <= 0.0:
                continue
            next_h = rep + (obs,)
            next_cid = partition.history_to_class[next_h]
            next_node = policy.next_node_index(node, obs)
            suffix = recurse(next_cid, next_node)
            for suf, p_suf in suffix.items():
                key = (obs,) + suf
                out[key] = out.get(key, 0.0) + p_obs * p_suf
        return out

    start_class = partition.history_to_class[()]
    return recurse(start_class, policy.initial_node)


def value_state_action_original(
    pomdp: FinitePOMDP,
    policy: DeterministicFSC,
    horizon: int,
) -> float:
    """Value of a deterministic FSC on the original POMDP. Supports multi-node FSCs."""

    @lru_cache(maxsize=None)
    def recurse(history: History, node: int) -> float:
        depth = len(history)
        if depth == horizon:
            return 0.0

        b = belief_after_history(pomdp=pomdp, policy=policy, history=history)
        action = policy.action_index(node)
        immediate = float(np.dot(b, pomdp.rewards[:, action]))

        obs_probs = np.zeros(pomdp.num_observations, dtype=float)
        for obs in range(pomdp.num_observations):
            p_obs = 0.0
            for s in range(pomdp.num_states):
                for s_next in range(pomdp.num_states):
                    p_obs += b[s] * pomdp.transition[s, action, s_next] * pomdp.observation[s_next, action, obs]
            obs_probs[obs] = p_obs

        z = float(obs_probs.sum())
        if z > 0.0:
            obs_probs = obs_probs / z

        future = 0.0
        for obs in range(pomdp.num_observations):
            if obs_probs[obs] <= 0.0:
                continue
            next_node = policy.next_node_index(node, obs)
            future += obs_probs[obs] * recurse(history + (obs,), next_node)

        return immediate + future

    return recurse((), policy.initial_node)


def value_state_action_quotient(
    pomdp: FinitePOMDP,
    partition: PartitionResult,
    policy: DeterministicFSC,
) -> float:
    """Value of a deterministic FSC on the quotient POMDP. Supports multi-node FSCs."""
    beliefs = _class_canonical_beliefs(pomdp=pomdp, partition=partition, policy=policy)

    @lru_cache(maxsize=None)
    def recurse(class_id: int, node: int) -> float:
        depth = partition.class_depth[class_id]
        if depth == partition.horizon:
            return 0.0

        b = beliefs[class_id]
        action = policy.action_index(node)
        immediate = float(np.dot(b, pomdp.rewards[:, action]))

        rep = _representative_history(partition, class_id)

        obs_probs = np.zeros(pomdp.num_observations, dtype=float)
        for obs in range(pomdp.num_observations):
            p_obs = 0.0
            for s in range(pomdp.num_states):
                for s_next in range(pomdp.num_states):
                    p_obs += b[s] * pomdp.transition[s, action, s_next] * pomdp.observation[s_next, action, obs]
            obs_probs[obs] = p_obs

        z = float(obs_probs.sum())
        if z > 0.0:
            obs_probs = obs_probs / z

        future = 0.0
        for obs in range(pomdp.num_observations):
            if obs_probs[obs] <= 0.0:
                continue
            next_h = rep + (obs,)
            next_cid = partition.history_to_class[next_h]
            next_node = policy.next_node_index(node, obs)
            future += obs_probs[obs] * recurse(next_cid, next_node)

        return immediate + future

    start = partition.history_to_class[()]
    return recurse(start, policy.initial_node)


def d_m_t_between_original_and_quotient(
    pomdp: FinitePOMDP,
    partition: PartitionResult,
    policies: Sequence[DeterministicFSC],
    distance_mode: str,
    d_obs: np.ndarray,
) -> float:
    dmax = 0.0
    for policy in policies:
        p_m = trajectory_observation_distribution(pomdp=pomdp, policy=policy, horizon=partition.horizon)
        p_q = quotient_observation_sequence_distribution(
            pomdp=pomdp,
            partition=partition,
            policy=policy,
        )
        d = distribution_distance(p_m, p_q, mode=distance_mode, d_obs=d_obs)
        dmax = max(dmax, d)
    return dmax


def d_m_t_between_two_pomdps(
    pomdp1: FinitePOMDP,
    pomdp2: FinitePOMDP,
    policies: Sequence[DeterministicFSC],
    horizon: int,
    distance_mode: str,
    d_obs: np.ndarray,
) -> float:
    """Behavioural distance between two POMDPs sharing the same (A, O).

    For each policy, computes the full trajectory observation distribution
    on both POMDPs and takes the max distance over all policies.
    """
    dmax = 0.0
    for policy in policies:
        p1 = trajectory_observation_distribution(pomdp=pomdp1, policy=policy, horizon=horizon)
        p2 = trajectory_observation_distribution(pomdp=pomdp2, policy=policy, horizon=horizon)
        d = distribution_distance(p1, p2, mode=distance_mode, d_obs=d_obs)
        dmax = max(dmax, d)
    return dmax
