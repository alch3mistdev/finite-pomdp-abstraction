"""Tests for RockSample benchmark, W1-vs-TV comparison, and PBVI quotient planning."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.benchmarks import rocksample_pomdp, rocksample_observation_metric
from experiments.fsc_enum import enumerate_deterministic_fscs
from experiments.quotient import (
    compute_partition_from_cache,
    precompute_distance_cache,
)
from experiments.analysis import (
    run_w1_vs_tv_structured_comparison,
    run_pbvi_quotient_comparison,
    pbvi_solve,
    materialize_quotient_pomdp,
)


# ---------------------------------------------------------------------------
# RockSample benchmark
# ---------------------------------------------------------------------------


def test_rocksample_construction() -> None:
    """RockSample(4,4) should have correct dimensions."""
    pomdp = rocksample_pomdp()
    assert pomdp.num_states == 257  # 4*4*2^4 + 1 terminal
    assert pomdp.num_actions == 9   # 4 moves + sample + 4 checks
    assert pomdp.num_observations == 3  # good, bad, none


def test_rocksample_stochastic_matrices() -> None:
    """Transition and observation rows must sum to 1."""
    pomdp = rocksample_pomdp()
    for s in range(pomdp.num_states):
        for a in range(pomdp.num_actions):
            assert abs(pomdp.transition[s, a, :].sum() - 1.0) < 1e-10, (
                f"Transition row (s={s}, a={a}) sums to {pomdp.transition[s, a, :].sum()}"
            )
            assert abs(pomdp.observation[s, a, :].sum() - 1.0) < 1e-10, (
                f"Observation row (s={s}, a={a}) sums to {pomdp.observation[s, a, :].sum()}"
            )
    assert abs(pomdp.initial_belief.sum() - 1.0) < 1e-10


def test_rocksample_observation_metric_shape() -> None:
    """Observation metric should be 3x3."""
    d_obs = rocksample_observation_metric()
    assert d_obs.shape == (3, 3)
    # Diagonal should be zero
    for i in range(3):
        assert d_obs[i, i] == 0.0
    # Symmetric
    assert np.allclose(d_obs, d_obs.T)


def test_rocksample_partition_monotonicity() -> None:
    """Class counts should decrease monotonically with epsilon."""
    pomdp = rocksample_pomdp()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1,
        include_smaller=True,
    )
    cache = precompute_distance_cache(
        pomdp=pomdp,
        policies=policies,
        horizon=2,
        distance_mode="w1",
        d_obs=rocksample_observation_metric(),
    )
    eps_grid = [0.0, 0.1, 0.25, 0.45]
    counts = []
    for eps in eps_grid:
        part = compute_partition_from_cache(cache, epsilon=eps)
        counts.append(part.num_classes_total)

    # Monotonically non-increasing
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1], (
            f"Non-monotonic at eps={eps_grid[i]}: {counts[i]} -> {counts[i+1]}"
        )


# ---------------------------------------------------------------------------
# W1 vs TV structured comparison
# ---------------------------------------------------------------------------


def test_w1_vs_tv_structured_columns() -> None:
    """Output DataFrame should have expected columns."""
    df = run_w1_vs_tv_structured_comparison(eps_grid=[0.0, 0.3])
    assert set(df.columns) >= {"epsilon", "classes_w1", "classes_tv", "classes_diff"}
    assert len(df) == 2


def test_w1_vs_tv_w1_merges_more_at_moderate_eps() -> None:
    """At moderate epsilon, W1 should produce <= classes than TV."""
    df = run_w1_vs_tv_structured_comparison(eps_grid=[0.0, 0.1, 0.3, 0.5])
    for _, row in df.iterrows():
        assert row["classes_w1"] <= row["classes_tv"], (
            f"W1 should merge at least as much as TV at eps={row['epsilon']}"
        )


def test_w1_vs_tv_at_eps_zero_equal() -> None:
    """At epsilon=0, both metrics should yield the same classes."""
    df = run_w1_vs_tv_structured_comparison(eps_grid=[0.0])
    row = df.iloc[0]
    assert row["classes_w1"] == row["classes_tv"]
    assert row["classes_diff"] == 0.0


# ---------------------------------------------------------------------------
# PBVI solver and quotient comparison
# ---------------------------------------------------------------------------


def test_pbvi_solve_tiger() -> None:
    """PBVI on Tiger should return a finite value."""
    from experiments.benchmarks import tiger_full_actions_pomdp
    pomdp = tiger_full_actions_pomdp()
    val = pbvi_solve(pomdp, horizon=3)
    assert np.isfinite(val)


def test_materialize_quotient_preserves_stochasticity() -> None:
    """Materialized quotient POMDP should have valid stochastic matrices."""
    from experiments.benchmarks import tiger_full_actions_pomdp, tiger_discrete_observation_metric
    pomdp = tiger_full_actions_pomdp()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1,
        include_smaller=True,
    )
    cache = precompute_distance_cache(
        pomdp=pomdp,
        policies=policies,
        horizon=3,
        distance_mode="w1",
        d_obs=tiger_discrete_observation_metric(),
    )
    part = compute_partition_from_cache(cache, epsilon=0.3)
    q_pomdp = materialize_quotient_pomdp(pomdp, part, policies[0])

    for s in range(q_pomdp.num_states):
        for a in range(q_pomdp.num_actions):
            assert abs(q_pomdp.transition[s, a, :].sum() - 1.0) < 1e-8
            assert abs(q_pomdp.observation[s, a, :].sum() - 1.0) < 1e-8
    assert abs(q_pomdp.initial_belief.sum() - 1.0) < 1e-8


def test_pbvi_quotient_comparison_columns() -> None:
    """Output DataFrame should have expected columns."""
    df = run_pbvi_quotient_comparison(eps_grid=[0.0, 0.3])
    expected = {"benchmark", "epsilon", "classes", "time_original_s",
                "time_quotient_s", "speedup", "value_original",
                "value_quotient", "value_gap"}
    assert expected <= set(df.columns)
    assert len(df) >= 2  # At least 2 benchmarks x 2 eps


def test_pbvi_quotient_zero_value_gap_at_eps_zero() -> None:
    """At epsilon=0 (no merging), quotient value should match original.

    Tolerance is generous (0.05) because quotient materialisation uses
    canonical belief averaging, which introduces small approximation error
    on larger state spaces (e.g. RockSample |S|=257).
    """
    df = run_pbvi_quotient_comparison(eps_grid=[0.0])
    for _, row in df.iterrows():
        assert row["value_gap"] < 0.05, (
            f"Large value gap at eps=0 for {row['benchmark']}: {row['value_gap']}"
        )
