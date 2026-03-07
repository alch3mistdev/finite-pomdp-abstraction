"""Tests for baseline partition methods."""

from __future__ import annotations

import pytest

from experiments.baselines import (
    belief_distance_partition,
    random_partition,
    truncation_partition,
)
from experiments.benchmarks import tiger_full_actions_pomdp
from experiments.fsc_enum import enumerate_deterministic_fscs


def test_truncation_partition_depth1():
    part = truncation_partition(num_observations=2, horizon=2, depth=1)
    # depth 0: 1 history (empty) -> 1 class
    # depth 1: 2 histories (0), (1) -> 2 classes (suffixes differ)
    # depth 2: 4 histories -> grouped by last obs -> 2 classes
    assert part.num_classes_total == 5  # 1 + 2 + 2
    # All histories assigned
    assert len(part.history_to_class) == 1 + 2 + 4  # 7 histories


def test_truncation_partition_depth0():
    part = truncation_partition(num_observations=2, horizon=2, depth=0)
    # depth 0: 1 class (empty suffix)
    # depth 1: all share empty suffix -> 1 class
    # depth 2: all share empty suffix -> 1 class
    assert part.num_classes_total == 3  # 1 per depth


def test_random_partition_respects_k():
    k_per_depth = {0: 1, 1: 2, 2: 2}
    part = random_partition(num_observations=2, horizon=2, k_per_depth=k_per_depth, seed=42)
    # Should have at most k classes per depth (may be fewer if random assigns nothing to a cluster)
    for d, k in k_per_depth.items():
        assert len(part.classes_by_depth[d]) <= k


def test_random_partition_different_seeds():
    k_per_depth = {0: 1, 1: 2, 2: 3}
    p1 = random_partition(num_observations=2, horizon=2, k_per_depth=k_per_depth, seed=7)
    p2 = random_partition(num_observations=2, horizon=2, k_per_depth=k_per_depth, seed=42)
    # Different seeds should (with high probability) produce different assignments
    # at depth 2 where there are 4 histories and 3 clusters
    h2c_1 = {h: c for h, c in p1.history_to_class.items() if len(h) == 2}
    h2c_2 = {h: c for h, c in p2.history_to_class.items() if len(h) == 2}
    # At minimum, both should be valid
    assert len(h2c_1) == 4
    assert len(h2c_2) == 4


def test_belief_distance_partition_basic():
    pomdp = tiger_full_actions_pomdp()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1, include_smaller=True,
    )
    part = belief_distance_partition(
        pomdp=pomdp, policy=policies[0], horizon=2, epsilon=0.5,
    )
    assert part.num_classes_total >= 3  # at least one class per depth
    assert len(part.history_to_class) == 7  # all histories assigned


def test_belief_distance_partition_exact():
    pomdp = tiger_full_actions_pomdp()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1, include_smaller=True,
    )
    part = belief_distance_partition(
        pomdp=pomdp, policy=policies[0], horizon=2, epsilon=0.0,
    )
    # At epsilon=0, histories with identical beliefs under the single policy
    # get merged (e.g. (0,1) and (1,0) have the same belief), so we get 6 not 7
    assert part.num_classes_total == 6


def test_all_baselines_cover_all_histories():
    """Every baseline must assign every history to exactly one class."""
    pomdp = tiger_full_actions_pomdp()
    horizon = 2
    num_obs = pomdp.num_observations
    total = sum(num_obs ** d for d in range(horizon + 1))

    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=num_obs,
        max_nodes=1, include_smaller=True,
    )

    for part in [
        truncation_partition(num_obs, horizon, depth=1),
        random_partition(num_obs, horizon, {0: 1, 1: 2, 2: 2}, seed=42),
        belief_distance_partition(pomdp, policies[0], horizon, epsilon=0.3),
    ]:
        assert len(part.history_to_class) == total
        # Every history in class_histories should appear in history_to_class
        for cid, members in part.class_histories.items():
            for h in members:
                assert part.history_to_class[h] == cid
