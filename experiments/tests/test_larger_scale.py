"""Tests for larger-scale experiments, random POMDP generator, and sampling module."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.benchmarks import (
    coarsen_observations,
    coarsened_observation_metric,
    gridworld_geometric_observation_metric,
    gridworld_pomdp,
    random_observation_metric,
    random_structured_pomdp,
    tiger_discrete_observation_metric,
    tiger_full_actions_pomdp,
    tiger_listen_only_pomdp,
)
from experiments.fsc_enum import enumerate_deterministic_fscs
from experiments.quotient import (
    compute_class_count_curve,
    compute_partition_from_cache,
    precompute_distance_cache,
)
from experiments.sampling import (
    empirical_distribution,
    sample_future_observations,
    sampling_based_distance_cache,
)
from experiments.spectral import (
    build_fsc_distance_tensor,
    build_sampling_based_fsc_distance_tensor,
    spectral_analysis,
)


# ---------------------------------------------------------------------------
# GridWorld generalization
# ---------------------------------------------------------------------------


def test_gridworld_5x5_construction() -> None:
    pomdp = gridworld_pomdp(size=5)
    assert pomdp.num_states == 25
    assert pomdp.num_actions == 5
    assert pomdp.num_observations == 4
    # Transition rows must sum to 1
    for s in range(pomdp.num_states):
        for a in range(pomdp.num_actions):
            assert abs(pomdp.transition[s, a, :].sum() - 1.0) < 1e-10
    # Observation rows must sum to 1
    for s in range(pomdp.num_states):
        for a in range(pomdp.num_actions):
            assert abs(pomdp.observation[s, a, :].sum() - 1.0) < 1e-10
    # Initial belief sums to 1
    assert abs(pomdp.initial_belief.sum() - 1.0) < 1e-10


def test_gridworld_3x3_unchanged() -> None:
    """Ensure 3x3 still works after removing the assertion."""
    pomdp = gridworld_pomdp(size=3)
    assert pomdp.num_states == 9


def test_gridworld_6x6_construction() -> None:
    pomdp = gridworld_pomdp(size=6)
    assert pomdp.num_states == 36


# ---------------------------------------------------------------------------
# Random structured POMDP
# ---------------------------------------------------------------------------


def test_random_pomdp_construction() -> None:
    pomdp = random_structured_pomdp(num_states=20, seed=42)
    assert pomdp.num_states == 20
    assert pomdp.num_actions == 3
    assert pomdp.num_observations == 4
    # Transition rows must be valid distributions
    for s in range(pomdp.num_states):
        for a in range(pomdp.num_actions):
            row = pomdp.transition[s, a, :]
            assert abs(row.sum() - 1.0) < 1e-10
            assert (row >= 0).all()
    # Observation rows must be valid distributions
    for s in range(pomdp.num_states):
        for a in range(pomdp.num_actions):
            row = pomdp.observation[s, a, :]
            assert abs(row.sum() - 1.0) < 1e-10
            assert (row >= 0).all()
    # Initial belief sums to 1
    assert abs(pomdp.initial_belief.sum() - 1.0) < 1e-10


def test_random_pomdp_different_seeds() -> None:
    """Different seeds produce different POMDPs."""
    p1 = random_structured_pomdp(num_states=10, seed=1)
    p2 = random_structured_pomdp(num_states=10, seed=2)
    assert not np.allclose(p1.transition, p2.transition)


def test_random_observation_metric() -> None:
    d = random_observation_metric(4, seed=42)
    assert d.shape == (4, 4)
    # Metric properties
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T)  # Symmetric
    assert (d >= 0).all()
    assert d.max() <= 1.0 + 1e-10  # Normalized


# ---------------------------------------------------------------------------
# Larger-scale class count monotonicity
# ---------------------------------------------------------------------------


def test_larger_scale_class_count_monotonicity() -> None:
    """Class count should be non-increasing as epsilon grows (5x5 GridWorld)."""
    pomdp = gridworld_pomdp(size=5)
    d_obs = gridworld_geometric_observation_metric()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1,
        include_smaller=True,
    )
    cache = precompute_distance_cache(
        pomdp=pomdp, policies=policies, horizon=2,
        distance_mode="w1", d_obs=d_obs,
    )
    eps_grid = [0.0, 0.1, 0.2, 0.3, 0.5]
    rows = compute_class_count_curve(cache, eps_grid)
    counts = [r["class_count"] for r in rows]
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1], f"Monotonicity violated at eps={eps_grid[i+1]}"


# ---------------------------------------------------------------------------
# Spectral analysis on GridWorld
# ---------------------------------------------------------------------------


def test_spectral_gridworld() -> None:
    """Spectral analysis runs on GridWorld without error."""
    pomdp = gridworld_pomdp(size=3)
    d_obs = gridworld_geometric_observation_metric()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1,
        include_smaller=True,
    )
    cache = build_fsc_distance_tensor(
        pomdp=pomdp, policies=policies, horizon=2,
        distance_mode="w1", d_obs=d_obs,
    )
    sa = spectral_analysis(cache)
    assert len(sa) > 0
    for depth, info in sa.items():
        assert info["effective_rank_99"] >= 0


# ---------------------------------------------------------------------------
# Sampling-based W1 estimation
# ---------------------------------------------------------------------------


def test_sampling_vs_exact_agreement() -> None:
    """On Tiger (small), sampling-based W1 should be close to exact."""
    pomdp = tiger_listen_only_pomdp()
    d_obs = tiger_discrete_observation_metric()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1,
        include_smaller=True,
    )

    exact_cache = precompute_distance_cache(
        pomdp=pomdp, policies=policies, horizon=2,
        distance_mode="w1", d_obs=d_obs,
    )
    sampled_cache = sampling_based_distance_cache(
        pomdp=pomdp, policies=policies, horizon=2,
        d_obs=d_obs, num_samples=2000, seed=42,
    )

    # Compare distance matrices at each depth
    for depth in exact_cache.max_distance_matrices:
        exact_mat = exact_cache.max_distance_matrices[depth]
        sampled_mat = sampled_cache.max_distance_matrices[depth]
        assert exact_mat.shape == sampled_mat.shape
        # Allow some tolerance since sampling is approximate
        assert np.allclose(exact_mat, sampled_mat, atol=0.1), (
            f"Sampling disagreement at depth {depth}: "
            f"max diff = {np.max(np.abs(exact_mat - sampled_mat)):.4f}"
        )


def test_sample_future_observations_basic() -> None:
    """Basic smoke test for trajectory sampling."""
    pomdp = tiger_listen_only_pomdp()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1,
        include_smaller=True,
    )
    rng = np.random.default_rng(42)
    samples = sample_future_observations(
        pomdp, policies[0], history=(), horizon=2, num_samples=100, rng=rng,
    )
    assert len(samples) == 100
    for s in samples:
        assert len(s) == 2
        assert all(0 <= o < pomdp.num_observations for o in s)


def test_empirical_distribution() -> None:
    samples = [(0, 1), (0, 1), (1, 0), (0, 1)]
    dist = empirical_distribution(samples)
    assert abs(dist[(0, 1)] - 0.75) < 1e-10
    assert abs(dist[(1, 0)] - 0.25) < 1e-10


# ---------------------------------------------------------------------------
# |S|=100 construction and sampling (M1)
# ---------------------------------------------------------------------------


def test_gridworld_10x10_construction() -> None:
    pomdp = gridworld_pomdp(size=10)
    assert pomdp.num_states == 100
    assert pomdp.num_actions == 5
    assert pomdp.num_observations == 4
    for s in range(pomdp.num_states):
        for a in range(pomdp.num_actions):
            assert abs(pomdp.transition[s, a, :].sum() - 1.0) < 1e-10
            assert abs(pomdp.observation[s, a, :].sum() - 1.0) < 1e-10


def test_random_pomdp_100_construction() -> None:
    pomdp = random_structured_pomdp(num_states=100, seed=42)
    assert pomdp.num_states == 100
    assert pomdp.num_actions == 3
    assert pomdp.num_observations == 4


def test_medium_scale_sampling_100() -> None:
    """Sampling-based cache works for |S|=100."""
    pomdp = gridworld_pomdp(size=10)
    d_obs = gridworld_geometric_observation_metric()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1,
        include_smaller=True,
    )
    cache = sampling_based_distance_cache(
        pomdp=pomdp,
        policies=policies,
        horizon=2,
        d_obs=d_obs,
        num_samples=100,
        seed=42,
    )
    assert 2 in cache.max_distance_matrices
    assert cache.max_distance_matrices[2].shape[0] > 0


# ---------------------------------------------------------------------------
# Observation coarsening wrapper (M2)
# ---------------------------------------------------------------------------


def test_coarsen_observations() -> None:
    """Coarsening merges observations correctly."""
    pomdp = gridworld_pomdp(size=3)
    merge_map = {0: 0, 1: 0, 2: 1, 3: 1}
    coarsened = coarsen_observations(pomdp, merge_map)
    assert coarsened.num_observations == 2
    assert coarsened.num_states == pomdp.num_states
    assert coarsened.num_actions == pomdp.num_actions
    # Coarsened observation rows must sum to 1
    for s in range(coarsened.num_states):
        for a in range(coarsened.num_actions):
            assert abs(coarsened.observation[s, a, :].sum() - 1.0) < 1e-10
    # New obs 0 = old obs 0 + old obs 1
    for s in range(pomdp.num_states):
        for a in range(pomdp.num_actions):
            expected = pomdp.observation[s, a, 0] + pomdp.observation[s, a, 1]
            assert abs(coarsened.observation[s, a, 0] - expected) < 1e-10


def test_coarsened_observation_metric() -> None:
    d_obs = gridworld_geometric_observation_metric()
    merge_map = {0: 0, 1: 0, 2: 1, 3: 1}
    new_d_obs, L_C = coarsened_observation_metric(d_obs, merge_map)
    assert new_d_obs.shape == (2, 2)
    assert L_C == 1.0
    assert new_d_obs[0, 0] == 0.0
    assert new_d_obs[1, 1] == 0.0
    assert new_d_obs[0, 1] > 0.0
    assert np.allclose(new_d_obs, new_d_obs.T)


def test_data_processing_monotonicity() -> None:
    """D^W on coarsened POMDP <= L_C * D^W on original (Theorem 5.3)."""
    pomdp = gridworld_pomdp(size=3)
    d_obs = gridworld_geometric_observation_metric()
    merge_map = {0: 0, 1: 0, 2: 1, 3: 1}
    coarsened = coarsen_observations(pomdp, merge_map)
    new_d_obs, L_C = coarsened_observation_metric(d_obs, merge_map)

    policies_orig = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1, include_smaller=True,
    )
    cache_orig = precompute_distance_cache(
        pomdp=pomdp, policies=policies_orig, horizon=2,
        distance_mode="w1", d_obs=d_obs,
    )

    policies_coarsened = enumerate_deterministic_fscs(
        num_actions=coarsened.num_actions,
        num_observations=coarsened.num_observations,
        max_nodes=1, include_smaller=True,
    )
    cache_coarsened = precompute_distance_cache(
        pomdp=coarsened, policies=policies_coarsened, horizon=2,
        distance_mode="w1", d_obs=new_d_obs,
    )

    for d in cache_orig.max_distance_matrices:
        max_orig = float(cache_orig.max_distance_matrices[d].max())
        if d in cache_coarsened.max_distance_matrices:
            max_coarsened = float(cache_coarsened.max_distance_matrices[d].max())
            assert max_coarsened <= L_C * max_orig + 1e-9, (
                f"Monotonicity violated at depth {d}: "
                f"coarsened={max_coarsened:.6f} > L_C*orig={L_C * max_orig:.6f}"
            )


# ---------------------------------------------------------------------------
# m=2 capacity tests (Phase 1d)
# ---------------------------------------------------------------------------


def test_tiger_m2_t4_capacity() -> None:
    """Tiger m=2 produces more classes than m=1 at T=4."""
    pomdp = tiger_full_actions_pomdp()
    d_obs = tiger_discrete_observation_metric()

    policies_m1 = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1, include_smaller=True,
    )
    cache_m1 = precompute_distance_cache(
        pomdp=pomdp, policies=policies_m1, horizon=4,
        distance_mode="w1", d_obs=d_obs,
    )
    part_m1 = compute_partition_from_cache(cache_m1, epsilon=0.0)

    policies_m2 = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=2, include_smaller=True,
    )
    cache_m2 = precompute_distance_cache(
        pomdp=pomdp, policies=policies_m2, horizon=4,
        distance_mode="w1", d_obs=d_obs,
    )
    part_m2 = compute_partition_from_cache(cache_m2, epsilon=0.0)

    assert len(policies_m2) == 147
    assert part_m2.num_classes_total >= part_m1.num_classes_total
    # Known values: m=1 gives 11, m=2 gives 16
    assert part_m1.num_classes_total == 11
    assert part_m2.num_classes_total == 16


def test_gridworld_m2_construction() -> None:
    """GridWorld 3x3 m=2 enumerates 6,405 FSCs correctly."""
    pomdp = gridworld_pomdp(size=3)
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=2, include_smaller=True,
    )
    assert len(policies) == 6405


# ---------------------------------------------------------------------------
# Convergence sweep test (Phase 2c)
# ---------------------------------------------------------------------------


def test_convergence_sweep() -> None:
    """Convergence sweep runs and ARI increases with samples."""
    from experiments.sampling import run_convergence_sweep

    pomdp = gridworld_pomdp(size=3)
    d_obs = gridworld_geometric_observation_metric()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1, include_smaller=True,
    )

    rows = run_convergence_sweep(
        pomdp=pomdp, policies=policies, horizon=2,
        d_obs=d_obs,
        num_samples_grid=(50, 200, 500),
        eps_grid=(0.0, 0.3),
    )
    assert len(rows) == 6  # 3 sample counts x 2 eps values

    # ARI at highest sample count should be high
    for eps in (0.0, 0.3):
        aris = [r["ari_vs_exact"] for r in rows if r["epsilon"] == eps]
        assert aris[-1] >= 0.8, f"ARI too low at eps={eps}: {aris[-1]}"


# ---------------------------------------------------------------------------
# Sampling-based FSC distance tensor tests
# ---------------------------------------------------------------------------


def test_sampling_fsc_distance_tensor_shape() -> None:
    """SpectralCache tensor shape, symmetry, zero diagonal."""
    tiger = tiger_full_actions_pomdp()
    d_obs = tiger_discrete_observation_metric()
    policies = enumerate_deterministic_fscs(
        num_actions=tiger.num_actions,
        num_observations=tiger.num_observations,
        max_nodes=1, include_smaller=True,
    )
    cache = build_sampling_based_fsc_distance_tensor(
        pomdp=tiger, policies=policies, horizon=2,
        d_obs=d_obs, num_samples=200, seed=0,
    )
    for depth, tensor in cache.per_fsc_distances.items():
        n = tensor.shape[0]
        n_fscs = tensor.shape[2]
        assert n_fscs == len(policies)
        # Symmetry
        for p in range(n_fscs):
            np.testing.assert_allclose(tensor[:, :, p], tensor[:, :, p].T, atol=1e-12)
        # Zero diagonal
        for p in range(n_fscs):
            for i in range(n):
                assert tensor[i, i, p] == 0.0


def test_sampling_spectral_vs_exact_tiger() -> None:
    """Sampling-based spectral agrees with exact on small Tiger (atol=0.15)."""
    tiger = tiger_full_actions_pomdp()
    d_obs = tiger_discrete_observation_metric()
    policies = enumerate_deterministic_fscs(
        num_actions=tiger.num_actions,
        num_observations=tiger.num_observations,
        max_nodes=1, include_smaller=True,
    )
    exact_cache = build_fsc_distance_tensor(
        pomdp=tiger, policies=policies, horizon=2,
        distance_mode="w1", d_obs=d_obs,
    )
    samp_cache = build_sampling_based_fsc_distance_tensor(
        pomdp=tiger, policies=policies, horizon=2,
        d_obs=d_obs, num_samples=1000, seed=42,
    )
    for depth in exact_cache.per_fsc_distances:
        np.testing.assert_allclose(
            exact_cache.max_distance_matrices[depth],
            samp_cache.max_distance_matrices[depth],
            atol=0.15,
        )


def test_m2_gridworld_8x8_runs() -> None:
    """Smoke test: m=2 at |S|=64 completes with correct FSC count."""
    gw = gridworld_pomdp(size=8)
    policies = enumerate_deterministic_fscs(
        num_actions=gw.num_actions,
        num_observations=gw.num_observations,
        max_nodes=2, include_smaller=True,
    )
    assert len(policies) == 6405  # 5 + 25*256
    d_obs = gridworld_geometric_observation_metric()
    cache = sampling_based_distance_cache(
        pomdp=gw, policies=policies, horizon=2,
        d_obs=d_obs, num_samples=50, seed=0,
    )
    part = compute_partition_from_cache(cache, epsilon=0.3)
    assert part.num_classes_total > 0


def test_m2_refines_m1() -> None:
    """m=2 class count >= m=1 class count (monotonicity)."""
    tiger = tiger_full_actions_pomdp()
    d_obs = tiger_discrete_observation_metric()
    for m in (1, 2):
        policies = enumerate_deterministic_fscs(
            num_actions=tiger.num_actions,
            num_observations=tiger.num_observations,
            max_nodes=m, include_smaller=True,
        )
        cache = precompute_distance_cache(
            pomdp=tiger, policies=policies, horizon=2,
            distance_mode="w1", d_obs=d_obs,
        )
        part = compute_partition_from_cache(cache, epsilon=0.0)
        if m == 1:
            m1_count = part.num_classes_total
        else:
            m2_count = part.num_classes_total
    assert m2_count >= m1_count, f"m=2 ({m2_count}) should refine m=1 ({m1_count})"


def test_sampling_variance_bounded() -> None:
    """Class count std <= 3 across 5 seeds at 500 samples on Tiger."""
    tiger = tiger_full_actions_pomdp()
    d_obs = tiger_discrete_observation_metric()
    policies = enumerate_deterministic_fscs(
        num_actions=tiger.num_actions,
        num_observations=tiger.num_observations,
        max_nodes=1, include_smaller=True,
    )
    counts = []
    for seed in range(5):
        cache = sampling_based_distance_cache(
            pomdp=tiger, policies=policies, horizon=2,
            d_obs=d_obs, num_samples=500, seed=seed * 1000,
        )
        part = compute_partition_from_cache(cache, epsilon=0.0)
        counts.append(part.num_classes_total)
    std = float(np.std(counts))
    assert std <= 3.0, f"Class count std too high: {std} (counts={counts})"
