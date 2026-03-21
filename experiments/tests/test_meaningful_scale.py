"""Tests for meaningful-scale experiments (|S| >= 1000)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.benchmarks import (
    network_monitoring_pomdp,
    random_structured_pomdp,
    discrete_observation_metric,
)
from experiments.fsc_enum import enumerate_deterministic_fscs
from experiments.quotient import compute_partition_from_cache
from experiments.sampling import sampling_based_distance_cache


def test_network_monitoring_12_construction() -> None:
    """Validate POMDP construction for |S|=4096."""
    pomdp = network_monitoring_pomdp(num_nodes=12)
    assert pomdp.num_states == 4096
    assert pomdp.num_observations == 3
    # Transition rows sum to 1
    for s in range(min(pomdp.num_states, 10)):  # spot check
        for a in range(pomdp.num_actions):
            assert abs(pomdp.transition[s, a, :].sum() - 1.0) < 1e-10
    # Observation rows sum to 1
    for s in range(min(pomdp.num_states, 10)):
        for a in range(pomdp.num_actions):
            assert abs(pomdp.observation[s, a, :].sum() - 1.0) < 1e-10


def test_random_structured_2000_construction() -> None:
    """Validate random POMDP with |S|=2000, |O|=10."""
    pomdp = random_structured_pomdp(num_states=2000, num_actions=5, num_observations=10, seed=42)
    assert pomdp.num_states == 2000
    assert pomdp.num_observations == 10
    assert pomdp.num_actions == 5
    # Spot-check distributions
    for s in range(min(pomdp.num_states, 10)):
        for a in range(pomdp.num_actions):
            assert abs(pomdp.transition[s, a, :].sum() - 1.0) < 1e-10
            assert abs(pomdp.observation[s, a, :].sum() - 1.0) < 1e-10


def test_meaningful_scale_monotonicity() -> None:
    """Class counts must be monotonically non-increasing in epsilon."""
    pomdp = network_monitoring_pomdp(num_nodes=10)
    d_obs = discrete_observation_metric(pomdp.num_observations)
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1,
        include_smaller=True,
    )
    cache = sampling_based_distance_cache(
        pomdp=pomdp, policies=policies, horizon=2,
        d_obs=d_obs, num_samples=100, seed=42,
    )
    eps_values = [0.0, 0.1, 0.3, 0.5, 1.0]
    prev_classes = float("inf")
    for eps in eps_values:
        part = compute_partition_from_cache(cache, epsilon=eps)
        assert part.num_classes_total <= prev_classes, (
            f"Class count increased from {prev_classes} to {part.num_classes_total} "
            f"when epsilon went from previous to {eps}"
        )
        prev_classes = part.num_classes_total


def test_m2_finer_than_m1_rocksample() -> None:
    """m=2 probes produce finer or equal partitions than m=1 (Prop 4.8)."""
    from experiments.benchmarks import rocksample_pomdp, rocksample_observation_metric

    pomdp = rocksample_pomdp()
    d_obs = rocksample_observation_metric()

    results = {}
    for m in (1, 2):
        policies = enumerate_deterministic_fscs(
            num_actions=pomdp.num_actions,
            num_observations=pomdp.num_observations,
            max_nodes=m,
            include_smaller=True,
        )
        cache = sampling_based_distance_cache(
            pomdp=pomdp, policies=policies, horizon=2,
            d_obs=d_obs, num_samples=100, seed=42,
        )
        results[m] = {
            eps: compute_partition_from_cache(cache, epsilon=eps).num_classes_total
            for eps in (0.0, 0.3, 0.5)
        }

    # m=2 should have >= classes at every epsilon
    for eps in (0.0, 0.3, 0.5):
        assert results[2][eps] >= results[1][eps], (
            f"m=2 produced fewer classes ({results[2][eps]}) than m=1 "
            f"({results[1][eps]}) at eps={eps}"
        )
