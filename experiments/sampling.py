"""Sampling-based W1 estimation for scalable distance computation."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .metrics import wasserstein_distance
from .pomdp_core import (
    DeterministicFSC,
    FinitePOMDP,
    History,
    ObsSeq,
    all_histories,
    joint_posterior_after_history,
)
from .quotient import DistanceCache


def _sample_one_trajectory(
    pomdp: FinitePOMDP,
    policy: DeterministicFSC,
    joint: np.ndarray,
    steps: int,
    rng: np.random.Generator,
) -> ObsSeq:
    """Sample a single future observation sequence from a joint (state, node) distribution."""
    # Sample initial (state, node) from the joint
    flat = joint.ravel()
    flat = np.maximum(flat, 0.0)
    z = flat.sum()
    if z <= 0.0:
        flat = np.ones_like(flat) / flat.size
    else:
        flat = flat / z

    idx = rng.choice(flat.size, p=flat)
    s, n = divmod(idx, policy.num_nodes)

    obs_seq: List[int] = []
    for _ in range(steps):
        a = policy.action_index(n)

        # Sample next state
        t_probs = pomdp.transition[s, a, :]
        s_next = rng.choice(pomdp.num_states, p=t_probs)

        # Sample observation
        o_probs = pomdp.observation[s_next, a, :]
        obs = rng.choice(pomdp.num_observations, p=o_probs)

        obs_seq.append(obs)

        # Update FSC node
        n = policy.next_node_index(n, obs)
        s = s_next

    return tuple(obs_seq)


def sample_future_observations(
    pomdp: FinitePOMDP,
    policy: DeterministicFSC,
    history: History,
    horizon: int,
    num_samples: int,
    rng: np.random.Generator,
) -> List[ObsSeq]:
    """Sample future observation sequences from P(O_{t+1:T} | history, policy)."""
    steps = horizon - len(history)
    if steps <= 0:
        return [()] * num_samples

    joint = joint_posterior_after_history(pomdp, policy, history)
    return [
        _sample_one_trajectory(pomdp, policy, joint, steps, rng)
        for _ in range(num_samples)
    ]


def empirical_distribution(samples: List[ObsSeq]) -> Dict[ObsSeq, float]:
    """Convert a list of observation sequences into an empirical distribution."""
    counts: Dict[ObsSeq, int] = {}
    for seq in samples:
        counts[seq] = counts.get(seq, 0) + 1
    n = len(samples)
    return {k: v / n for k, v in counts.items()}


def empirical_wasserstein_distance(
    samples_p: List[ObsSeq],
    samples_q: List[ObsSeq],
    d_obs: np.ndarray,
) -> float:
    """Compute W1 between two empirical distributions of observation sequences."""
    dist_p = empirical_distribution(samples_p)
    dist_q = empirical_distribution(samples_q)
    return wasserstein_distance(dist_p, dist_q, d_obs=d_obs)


def sampling_based_distance_cache(
    pomdp: FinitePOMDP,
    policies: Sequence[DeterministicFSC],
    horizon: int,
    d_obs: np.ndarray,
    num_samples: int = 500,
    seed: int = 42,
) -> DistanceCache:
    """Build a DistanceCache using sampling-based W1 estimation."""
    rng = np.random.default_rng(seed)
    histories_by_depth = all_histories(
        num_observations=pomdp.num_observations, horizon=horizon
    )

    # Pre-sample future observations for each (policy, history) pair
    samples: List[Dict[History, List[ObsSeq]]] = []
    for policy in policies:
        policy_samples: Dict[History, List[ObsSeq]] = {}
        for depth, histories in histories_by_depth.items():
            for h in histories:
                policy_samples[h] = sample_future_observations(
                    pomdp, policy, h, horizon, num_samples, rng
                )
        samples.append(policy_samples)

    # Compute max distance matrices
    max_distance_matrices: Dict[int, np.ndarray] = {}
    for depth, histories in histories_by_depth.items():
        n = len(histories)
        matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                hi, hj = histories[i], histories[j]
                dmax = 0.0
                for p_idx in range(len(policies)):
                    dist_i = empirical_distribution(samples[p_idx][hi])
                    dist_j = empirical_distribution(samples[p_idx][hj])
                    d = wasserstein_distance(dist_i, dist_j, d_obs=d_obs)
                    dmax = max(dmax, d)
                matrix[i, j] = dmax
                matrix[j, i] = dmax
        max_distance_matrices[depth] = matrix

    return DistanceCache(
        horizon=horizon,
        histories_by_depth=histories_by_depth,
        max_distance_matrices=max_distance_matrices,
    )


def run_convergence_sweep(
    pomdp: FinitePOMDP,
    policies: Sequence[DeterministicFSC],
    horizon: int,
    d_obs: np.ndarray,
    num_samples_grid: Sequence[int] = (50, 100, 250, 500, 1000),
    eps_grid: Sequence[float] = (0.0, 0.1, 0.3, 0.5),
    exact_cache: "DistanceCache | None" = None,
    seed: int = 42,
) -> List[Dict]:
    """Convergence analysis: vary sample count, compare partitions to exact.

    Returns list of dicts with columns: num_samples, epsilon, class_count_exact,
    class_count_sampled, ari_vs_exact.
    """
    from .quotient import compute_partition_from_cache, precompute_distance_cache
    from .spectral import partition_agreement

    if exact_cache is None:
        exact_cache = precompute_distance_cache(
            pomdp=pomdp, policies=policies, horizon=horizon,
            distance_mode="w1", d_obs=d_obs,
        )

    rows: List[Dict] = []
    for n_samples in num_samples_grid:
        sampled_cache = sampling_based_distance_cache(
            pomdp=pomdp, policies=policies, horizon=horizon,
            d_obs=d_obs, num_samples=n_samples, seed=seed,
        )
        for eps in eps_grid:
            exact_part = compute_partition_from_cache(exact_cache, epsilon=float(eps))
            sampled_part = compute_partition_from_cache(sampled_cache, epsilon=float(eps))
            agreement = partition_agreement(exact_part, sampled_part)
            rows.append({
                "num_samples": n_samples,
                "epsilon": float(eps),
                "class_count_exact": exact_part.num_classes_total,
                "class_count_sampled": sampled_part.num_classes_total,
                "ari_vs_exact": float(agreement["adjusted_rand_index"]),
            })

    return rows
