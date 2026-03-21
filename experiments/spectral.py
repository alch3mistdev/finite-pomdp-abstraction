"""Spectral analysis of the FSC distance tensor for approximate quotient computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .benchmarks import tiger_discrete_observation_metric, tiger_full_actions_pomdp
from .fsc_enum import enumerate_deterministic_fscs
from .metrics import distribution_distance, wasserstein_distance
from .pomdp_core import (
    DeterministicFSC,
    FinitePOMDP,
    History,
    ObsSeq,
    all_histories,
    conditional_future_observation_distribution,
)
from .quotient import (
    DistanceCache,
    PartitionResult,
    _cluster_with_complete_linkage,
    compute_partition_from_cache,
)
from .sampling import empirical_distribution, sample_future_observations


@dataclass(frozen=True)
class SpectralCache:
    """Stores the full per-FSC distance tensor alongside the max-distance matrices."""

    per_fsc_distances: Dict[int, np.ndarray]  # depth -> [n_histories, n_histories, n_fscs]
    histories_by_depth: Dict[int, List[History]]
    policies: List[DeterministicFSC]
    horizon: int
    max_distance_matrices: Dict[int, np.ndarray]  # depth -> [n_histories, n_histories]


def build_fsc_distance_tensor(
    pomdp: FinitePOMDP,
    policies: Sequence[DeterministicFSC],
    horizon: int,
    distance_mode: str,
    d_obs: np.ndarray,
) -> SpectralCache:
    """Build per-FSC distance tensor: tensor[i, j, p] = W_1(P^{pi_p}(.|h_i), P^{pi_p}(.|h_j))."""
    histories_by_depth = all_histories(num_observations=pomdp.num_observations, horizon=horizon)

    # Precompute conditional distributions for all (policy, history) pairs
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

    per_fsc_distances: Dict[int, np.ndarray] = {}
    max_distance_matrices: Dict[int, np.ndarray] = {}

    for depth, histories in histories_by_depth.items():
        n = len(histories)
        n_fscs = len(policies)
        tensor = np.zeros((n, n, n_fscs), dtype=float)

        for p_idx in range(n_fscs):
            for i in range(n):
                for j in range(i + 1, n):
                    d = distribution_distance(
                        cond[p_idx][histories[i]],
                        cond[p_idx][histories[j]],
                        mode=distance_mode,
                        d_obs=d_obs,
                    )
                    tensor[i, j, p_idx] = d
                    tensor[j, i, p_idx] = d

        per_fsc_distances[depth] = tensor
        max_distance_matrices[depth] = tensor.max(axis=2)

    return SpectralCache(
        per_fsc_distances=per_fsc_distances,
        histories_by_depth=histories_by_depth,
        policies=list(policies),
        horizon=horizon,
        max_distance_matrices=max_distance_matrices,
    )


def build_sampling_based_fsc_distance_tensor(
    pomdp: FinitePOMDP,
    policies: Sequence[DeterministicFSC],
    horizon: int,
    d_obs: np.ndarray,
    num_samples: int = 500,
    seed: int = 42,
) -> SpectralCache:
    """Build per-FSC distance tensor using sampling-based W1 estimation.

    Same interface as ``build_fsc_distance_tensor`` but uses forward simulation
    instead of exact conditional distributions, enabling scalability to large
    state spaces (|S| >= 64) where exact computation is prohibitive.
    """
    rng = np.random.default_rng(seed)
    histories_by_depth = all_histories(
        num_observations=pomdp.num_observations, horizon=horizon
    )

    # Pre-sample and convert to empirical distributions
    empiricals: List[Dict[History, Mapping[ObsSeq, float]]] = []
    for policy in policies:
        mapping: Dict[History, Mapping[ObsSeq, float]] = {}
        for depth, histories in histories_by_depth.items():
            for h in histories:
                samples = sample_future_observations(
                    pomdp, policy, h, horizon, num_samples, rng
                )
                mapping[h] = empirical_distribution(samples)
        empiricals.append(mapping)

    per_fsc_distances: Dict[int, np.ndarray] = {}
    max_distance_matrices: Dict[int, np.ndarray] = {}

    for depth, histories in histories_by_depth.items():
        n = len(histories)
        n_fscs = len(policies)
        tensor = np.zeros((n, n, n_fscs), dtype=float)

        for p_idx in range(n_fscs):
            for i in range(n):
                for j in range(i + 1, n):
                    d = wasserstein_distance(
                        empiricals[p_idx][histories[i]],
                        empiricals[p_idx][histories[j]],
                        d_obs=d_obs,
                    )
                    tensor[i, j, p_idx] = d
                    tensor[j, i, p_idx] = d

        per_fsc_distances[depth] = tensor
        max_distance_matrices[depth] = tensor.max(axis=2)

    return SpectralCache(
        per_fsc_distances=per_fsc_distances,
        histories_by_depth=histories_by_depth,
        policies=list(policies),
        horizon=horizon,
        max_distance_matrices=max_distance_matrices,
    )


def spectral_analysis(cache: SpectralCache) -> Dict:
    """SVD analysis of the flattened pair-by-FSC distinguishing matrix per depth.

    Returns per-depth singular values, cumulative explained variance ratios,
    and effective rank at 90%, 95%, 99% thresholds.
    """
    results: Dict[int, Dict] = {}

    for depth, tensor in cache.per_fsc_distances.items():
        n = tensor.shape[0]
        n_fscs = tensor.shape[2]

        # Flatten upper-triangle pairs into rows, FSCs into columns
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append(tensor[i, j, :])  # shape [n_fscs]

        if not pairs:
            results[depth] = {
                "singular_values": np.array([]),
                "explained_variance_ratio": np.array([]),
                "cumulative_variance_ratio": np.array([]),
                "effective_rank_90": 0,
                "effective_rank_95": 0,
                "effective_rank_99": 0,
            }
            continue

        D = np.array(pairs, dtype=float)  # [n_pairs, n_fscs]
        _, s, _ = np.linalg.svd(D, full_matrices=False)

        total_var = float((s**2).sum())
        if total_var > 0:
            explained = s**2 / total_var
            cumulative = np.cumsum(explained)
        else:
            explained = np.zeros_like(s)
            cumulative = np.zeros_like(s)

        def effective_rank(threshold: float) -> int:
            if total_var == 0:
                return 0
            idx = np.searchsorted(cumulative, threshold - 1e-12)
            return int(min(idx + 1, len(s)))

        results[depth] = {
            "singular_values": s,
            "explained_variance_ratio": explained,
            "cumulative_variance_ratio": cumulative,
            "effective_rank_90": effective_rank(0.90),
            "effective_rank_95": effective_rank(0.95),
            "effective_rank_99": effective_rank(0.99),
        }

    return results


def greedy_select_fscs(cache: SpectralCache, k: int) -> List[int]:
    """Greedy submodular selection of k principal distinguishing FSCs.

    At each step, selects the FSC maximizing the total increase in pairwise
    max-distances across all depths.
    """
    n_fscs = len(cache.policies)
    k = min(k, n_fscs)

    # Current best distance per (depth, i, j) pair
    current_max: Dict[int, np.ndarray] = {}
    for depth, tensor in cache.per_fsc_distances.items():
        n = tensor.shape[0]
        current_max[depth] = np.zeros((n, n), dtype=float)

    selected: List[int] = []
    remaining = set(range(n_fscs))

    for _ in range(k):
        best_gain = -1.0
        best_fsc = -1

        for p_idx in remaining:
            gain = 0.0
            for depth, tensor in cache.per_fsc_distances.items():
                n = tensor.shape[0]
                for i in range(n):
                    for j in range(i + 1, n):
                        improvement = max(0.0, tensor[i, j, p_idx] - current_max[depth][i, j])
                        gain += improvement
            if gain > best_gain:
                best_gain = gain
                best_fsc = p_idx

        if best_fsc < 0:
            break

        selected.append(best_fsc)
        remaining.discard(best_fsc)

        # Update current_max
        for depth, tensor in cache.per_fsc_distances.items():
            n = tensor.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    val = tensor[i, j, best_fsc]
                    if val > current_max[depth][i, j]:
                        current_max[depth][i, j] = val
                        current_max[depth][j, i] = val

    return selected


def subset_probe_gap_sup(cache: SpectralCache, fsc_indices: Sequence[int]) -> float:
    """Maximum probe-envelope gap between the full FSC set and a selected subset."""
    probe_gap = 0.0
    for depth, tensor in cache.per_fsc_distances.items():
        full_envelope = tensor.max(axis=2)
        if fsc_indices:
            subset_envelope = tensor[:, :, list(fsc_indices)].max(axis=2)
        else:
            subset_envelope = np.zeros_like(full_envelope)
        probe_gap = max(probe_gap, float(np.max(full_envelope - subset_envelope)))
    return probe_gap


def approximate_partition_from_subset(
    cache: SpectralCache,
    fsc_indices: List[int],
    epsilon: float,
) -> PartitionResult:
    """Compute partition using only the selected FSC subset."""
    approx_max_matrices: Dict[int, np.ndarray] = {}
    for depth, tensor in cache.per_fsc_distances.items():
        if fsc_indices:
            approx_max_matrices[depth] = tensor[:, :, fsc_indices].max(axis=2)
        else:
            n = tensor.shape[0]
            approx_max_matrices[depth] = np.zeros((n, n), dtype=float)

    approx_cache = DistanceCache(
        horizon=cache.horizon,
        histories_by_depth=cache.histories_by_depth,
        max_distance_matrices=approx_max_matrices,
    )
    return compute_partition_from_cache(approx_cache, epsilon=epsilon)


def partition_agreement(exact: PartitionResult, approx: PartitionResult) -> Dict:
    """Compare two partitions via Adjusted Rand Index and merge fidelity."""
    # Build label vectors from both partitions over the same history set
    histories = sorted(exact.history_to_class.keys())
    labels_exact = [exact.history_to_class[h] for h in histories]
    labels_approx = [approx.history_to_class[h] for h in histories]

    # Adjusted Rand Index
    ari = _adjusted_rand_index(labels_exact, labels_approx)

    # Merge fidelity: fraction of pairs correctly classified as same/different
    n = len(histories)
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_exact = labels_exact[i] == labels_exact[j]
            same_approx = labels_approx[i] == labels_approx[j]
            if same_exact == same_approx:
                correct += 1
            total += 1

    merge_fidelity = float(correct) / float(total) if total > 0 else 1.0

    return {
        "adjusted_rand_index": ari,
        "exact_classes": exact.num_classes_total,
        "approx_classes": approx.num_classes_total,
        "merge_fidelity": merge_fidelity,
    }


def _adjusted_rand_index(labels_a: List[int], labels_b: List[int]) -> float:
    """Compute the Adjusted Rand Index between two label assignments."""
    n = len(labels_a)
    if n < 2:
        return 1.0

    # Build contingency table
    classes_a = sorted(set(labels_a))
    classes_b = sorted(set(labels_b))
    map_a = {c: i for i, c in enumerate(classes_a)}
    map_b = {c: i for i, c in enumerate(classes_b)}

    nij = np.zeros((len(classes_a), len(classes_b)), dtype=int)
    for la, lb in zip(labels_a, labels_b):
        nij[map_a[la], map_b[lb]] += 1

    # Row and column sums
    a_sums = nij.sum(axis=1)
    b_sums = nij.sum(axis=0)

    def comb2(x):
        return x * (x - 1) // 2

    sum_comb_nij = sum(comb2(int(nij[i, j])) for i in range(len(classes_a)) for j in range(len(classes_b)))
    sum_comb_a = sum(comb2(int(x)) for x in a_sums)
    sum_comb_b = sum(comb2(int(x)) for x in b_sums)
    comb_n = comb2(n)

    expected = float(sum_comb_a * sum_comb_b) / float(comb_n) if comb_n > 0 else 0.0
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denominator = max_index - expected

    if abs(denominator) < 1e-15:
        return 1.0 if abs(float(sum_comb_nij) - expected) < 1e-15 else 0.0

    return (float(sum_comb_nij) - expected) / denominator
