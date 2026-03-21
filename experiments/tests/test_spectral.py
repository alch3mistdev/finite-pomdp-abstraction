"""Tests for the spectral quotient approximation module."""

from __future__ import annotations

import numpy as np

from experiments.analysis import run_spectral_partition_comparison
from experiments.benchmarks import (
    gridworld_geometric_observation_metric,
    gridworld_pomdp,
    tiger_discrete_observation_metric,
    tiger_full_actions_pomdp,
)
from experiments.fsc_enum import enumerate_deterministic_fscs
from experiments.spectral import (
    approximate_partition_from_subset,
    build_fsc_distance_tensor,
    greedy_select_fscs,
    partition_agreement,
    spectral_analysis,
    subset_probe_gap_sup,
)


def _build_tiger_cache(m: int = 1, horizon: int = 2):
    pomdp = tiger_full_actions_pomdp()
    d_obs = tiger_discrete_observation_metric()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=m,
        include_smaller=True,
    )
    return build_fsc_distance_tensor(
        pomdp=pomdp,
        policies=policies,
        horizon=horizon,
        distance_mode="w1",
        d_obs=d_obs,
    )


def _build_gridworld_cache(m: int = 2, horizon: int = 2):
    pomdp = gridworld_pomdp(size=3)
    d_obs = gridworld_geometric_observation_metric()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=m,
        include_smaller=True,
    )
    return build_fsc_distance_tensor(
        pomdp=pomdp,
        policies=policies,
        horizon=horizon,
        distance_mode="w1",
        d_obs=d_obs,
    )


def test_tensor_shape_and_symmetry() -> None:
    cache = _build_tiger_cache(m=1, horizon=2)
    for depth, tensor in cache.per_fsc_distances.items():
        n_histories = len(cache.histories_by_depth[depth])
        n_fscs = len(cache.policies)
        assert tensor.shape == (n_histories, n_histories, n_fscs)
        # Symmetry: tensor[i,j,p] == tensor[j,i,p]
        for p in range(n_fscs):
            np.testing.assert_allclose(tensor[:, :, p], tensor[:, :, p].T, atol=1e-12)
        # Zero diagonal
        for i in range(n_histories):
            np.testing.assert_allclose(tensor[i, i, :], 0.0, atol=1e-12)


def test_greedy_selection_monotonic() -> None:
    """More FSCs selected -> partition should be at least as good (ARI non-decreasing)."""
    cache = _build_tiger_cache(m=1, horizon=2)
    eps = 0.3
    n_fscs = len(cache.policies)

    exact = approximate_partition_from_subset(
        cache=cache,
        fsc_indices=list(range(n_fscs)),
        epsilon=eps,
    )

    prev_ari = -1.0
    for k in range(1, n_fscs + 1):
        selected = greedy_select_fscs(cache, k=k)
        approx = approximate_partition_from_subset(cache=cache, fsc_indices=selected, epsilon=eps)
        agreement = partition_agreement(exact, approx)
        ari = agreement["adjusted_rand_index"]
        assert ari >= prev_ari - 1e-12, f"ARI decreased from {prev_ari} to {ari} at k={k}"
        prev_ari = ari


def test_full_subset_exact_match() -> None:
    """When k = all FSCs, approximate partition must equal exact partition."""
    cache = _build_tiger_cache(m=1, horizon=2)
    n_fscs = len(cache.policies)

    for eps in [0.0, 0.2, 0.5]:
        exact = approximate_partition_from_subset(
            cache=cache,
            fsc_indices=list(range(n_fscs)),
            epsilon=eps,
        )
        selected = greedy_select_fscs(cache, k=n_fscs)
        approx = approximate_partition_from_subset(cache=cache, fsc_indices=selected, epsilon=eps)
        agreement = partition_agreement(exact, approx)
        assert agreement["adjusted_rand_index"] == pytest.approx(1.0, abs=1e-12)
        assert agreement["merge_fidelity"] == pytest.approx(1.0, abs=1e-12)
        assert agreement["exact_classes"] == agreement["approx_classes"]


def test_full_subset_probe_gap_zero() -> None:
    cache = _build_tiger_cache(m=1, horizon=2)
    n_fscs = len(cache.policies)
    assert subset_probe_gap_sup(cache, list(range(n_fscs))) == pytest.approx(0.0, abs=1e-12)


def test_greedy_probe_gap_monotonic() -> None:
    cache = _build_tiger_cache(m=1, horizon=2)
    prev_gap = float("inf")
    for k in range(1, len(cache.policies) + 1):
        gap = subset_probe_gap_sup(cache, greedy_select_fscs(cache, k=k))
        assert gap <= prev_gap + 1e-12, f"Probe gap increased from {prev_gap} to {gap} at k={k}"
        prev_gap = gap


def test_spectral_rank_tiger() -> None:
    """Validate low-rank on Tiger (m=1): effective rank at 99% should be < total FSCs."""
    cache = _build_tiger_cache(m=1, horizon=2)
    sa = spectral_analysis(cache)

    # Check at least one depth has low effective rank
    n_fscs = len(cache.policies)
    has_low_rank = False
    for depth, info in sa.items():
        if len(info["singular_values"]) > 0 and info["effective_rank_99"] < n_fscs:
            has_low_rank = True
    assert has_low_rank, "Expected low rank in at least one depth"


def test_max_distance_consistency() -> None:
    """Max-distance matrix from SpectralCache should match tensor.max(axis=2)."""
    cache = _build_tiger_cache(m=1, horizon=2)
    for depth, tensor in cache.per_fsc_distances.items():
        expected = tensor.max(axis=2)
        np.testing.assert_allclose(cache.max_distance_matrices[depth], expected, atol=1e-12)


def test_tiger_k5_exact_probe_recovery() -> None:
    cache = _build_tiger_cache(m=2, horizon=4)
    selected = greedy_select_fscs(cache, k=5)
    assert subset_probe_gap_sup(cache, selected) == pytest.approx(0.0, abs=1e-12)

    exact = approximate_partition_from_subset(cache, list(range(len(cache.policies))), epsilon=0.0)
    approx = approximate_partition_from_subset(cache, selected, epsilon=0.0)
    agreement = partition_agreement(exact, approx)
    assert agreement["adjusted_rand_index"] == pytest.approx(1.0, abs=1e-12)


def test_gridworld_k5_exact_probe_recovery() -> None:
    cache = _build_gridworld_cache(m=2, horizon=2)
    selected = greedy_select_fscs(cache, k=5)
    assert subset_probe_gap_sup(cache, selected) == pytest.approx(0.0, abs=1e-12)

    exact = approximate_partition_from_subset(cache, list(range(len(cache.policies))), epsilon=0.0)
    approx = approximate_partition_from_subset(cache, selected, epsilon=0.0)
    agreement = partition_agreement(exact, approx)
    assert agreement["adjusted_rand_index"] == pytest.approx(1.0, abs=1e-12)


def test_spectral_partition_comparison_outputs_probe_gap_columns() -> None:
    df = run_spectral_partition_comparison(
        eps_grid=(0.0,),
        ms=(1,),
        horizon=2,
        ks=(1,),
        include_gridworld=False,
    )
    assert {"probe_gap_sup", "exact_probe_recovery"}.issubset(df.columns)
    assert np.isfinite(df["probe_gap_sup"]).all()


import pytest  # noqa: E402
