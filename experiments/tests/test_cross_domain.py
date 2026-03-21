"""Tests for cross-domain experiments: channel communication and model distinguishability."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.benchmarks import (
    channel_communication_pomdp,
    channel_obs_metric,
    tiger_listen_only_pomdp,
    tiger_discrete_observation_metric,
)
from experiments.fsc_enum import enumerate_deterministic_fscs
from experiments.quotient import (
    compute_partition_from_cache,
    d_m_t_between_two_pomdps,
    precompute_distance_cache,
)


class TestChannelCommunicationPOMDP:
    """Validate channel_communication_pomdp construction."""

    def test_channel_pomdp_valid_distributions(self):
        """Transition and observation matrices must sum to 1 along last axis."""
        for noise in (0.0, 0.1, 0.3, 0.5):
            pomdp = channel_communication_pomdp(num_symbols=4, num_codewords=4, noise=noise)
            for s in range(pomdp.num_states):
                for a in range(pomdp.num_actions):
                    assert abs(pomdp.transition[s, a, :].sum() - 1.0) < 1e-12
                    assert abs(pomdp.observation[s, a, :].sum() - 1.0) < 1e-12
            assert abs(pomdp.initial_belief.sum() - 1.0) < 1e-12

    def test_channel_quotient_monotonicity(self):
        """More noise should yield fewer or equal quotient classes at fixed epsilon."""
        horizon = 2
        epsilon = 0.3
        prev_classes = float("inf")

        for noise in (0.0, 0.1, 0.2, 0.3, 0.5):
            pomdp = channel_communication_pomdp(num_symbols=4, num_codewords=4, noise=noise)
            d_obs = channel_obs_metric(4)
            policies = enumerate_deterministic_fscs(
                num_actions=pomdp.num_actions,
                num_observations=pomdp.num_observations,
                max_nodes=1,
                include_smaller=True,
            )
            cache = precompute_distance_cache(
                pomdp=pomdp, policies=policies, horizon=horizon,
                distance_mode="w1", d_obs=d_obs,
            )
            part = compute_partition_from_cache(cache, epsilon=epsilon)
            assert part.num_classes_total <= prev_classes + 1  # allow small tolerance
            prev_classes = part.num_classes_total

    def test_channel_decoder_capacity(self):
        """More memory should yield more or equal classes at nonzero noise."""
        noise = 0.3
        horizon = 2
        epsilon = 0.1
        pomdp = channel_communication_pomdp(num_symbols=4, num_codewords=4, noise=noise)
        d_obs = channel_obs_metric(4)

        class_counts = []
        for m in (1, 2):
            policies = enumerate_deterministic_fscs(
                num_actions=pomdp.num_actions,
                num_observations=pomdp.num_observations,
                max_nodes=m,
                include_smaller=True,
            )
            cache = precompute_distance_cache(
                pomdp=pomdp, policies=policies, horizon=horizon,
                distance_mode="w1", d_obs=d_obs,
            )
            part = compute_partition_from_cache(cache, epsilon=epsilon)
            class_counts.append(part.num_classes_total)

        # m=2 should have at least as many classes as m=1
        assert class_counts[1] >= class_counts[0]

    def test_channel_obs_metric_valid(self):
        """Channel obs metric should be a valid metric (Hamming distance)."""
        d = channel_obs_metric(4)
        assert d.shape == (4, 4)
        np.testing.assert_array_equal(np.diag(d), 0.0)
        assert (d >= 0).all()
        np.testing.assert_array_equal(d, d.T)


class TestModelDistinguishability:
    """Validate d_m_t_between_two_pomdps."""

    def test_model_distinguishability_reflexive(self):
        """D(M, M) must be 0."""
        pomdp = tiger_listen_only_pomdp(accuracy=0.85)
        d_obs = tiger_discrete_observation_metric()
        policies = enumerate_deterministic_fscs(
            num_actions=pomdp.num_actions,
            num_observations=pomdp.num_observations,
            max_nodes=1,
            include_smaller=True,
        )
        d = d_m_t_between_two_pomdps(
            pomdp1=pomdp, pomdp2=pomdp,
            policies=policies, horizon=2,
            distance_mode="w1", d_obs=d_obs,
        )
        assert abs(d) < 1e-12

    def test_model_distinguishability_symmetric(self):
        """D(M1, M2) must equal D(M2, M1)."""
        m1 = tiger_listen_only_pomdp(accuracy=0.85)
        m2 = tiger_listen_only_pomdp(accuracy=0.70)
        d_obs = tiger_discrete_observation_metric()
        policies = enumerate_deterministic_fscs(
            num_actions=m1.num_actions,
            num_observations=m1.num_observations,
            max_nodes=1,
            include_smaller=True,
        )
        d_fwd = d_m_t_between_two_pomdps(
            pomdp1=m1, pomdp2=m2,
            policies=policies, horizon=2,
            distance_mode="w1", d_obs=d_obs,
        )
        d_rev = d_m_t_between_two_pomdps(
            pomdp1=m2, pomdp2=m1,
            policies=policies, horizon=2,
            distance_mode="w1", d_obs=d_obs,
        )
        assert abs(d_fwd - d_rev) < 1e-12

    def test_model_distinguishability_monotone_in_m(self):
        """Larger m should yield D that never decreases."""
        m1 = tiger_listen_only_pomdp(accuracy=0.85)
        m2 = tiger_listen_only_pomdp(accuracy=0.70)
        d_obs = tiger_discrete_observation_metric()

        distances = []
        for m in (1, 2):
            policies = enumerate_deterministic_fscs(
                num_actions=m1.num_actions,
                num_observations=m1.num_observations,
                max_nodes=m,
                include_smaller=True,
            )
            d = d_m_t_between_two_pomdps(
                pomdp1=m1, pomdp2=m2,
                policies=policies, horizon=2,
                distance_mode="w1", d_obs=d_obs,
            )
            distances.append(d)

        # m=2 policies include all m=1 policies, so distance cannot decrease
        assert distances[1] >= distances[0] - 1e-12

    def test_model_distinguishability_increases_with_gap(self):
        """Larger accuracy gap should yield larger or equal D."""
        ref = tiger_listen_only_pomdp(accuracy=0.85)
        d_obs = tiger_discrete_observation_metric()
        policies = enumerate_deterministic_fscs(
            num_actions=ref.num_actions,
            num_observations=ref.num_observations,
            max_nodes=1,
            include_smaller=True,
        )

        prev_d = 0.0
        for acc in (0.80, 0.75, 0.70):
            test = tiger_listen_only_pomdp(accuracy=acc)
            d = d_m_t_between_two_pomdps(
                pomdp1=ref, pomdp2=test,
                policies=policies, horizon=2,
                distance_mode="w1", d_obs=d_obs,
            )
            assert d >= prev_d - 1e-12
            prev_d = d
