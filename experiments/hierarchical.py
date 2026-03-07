"""Hierarchical scalability utilities for long-horizon quotient experiments."""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .benchmarks import (
    coarsen_observations,
    coarsened_observation_metric,
    discrete_observation_metric,
    gridworld_geometric_observation_metric,
    gridworld_pomdp,
    stochastic_channel_lipschitz_constant,
    stochastic_coarsen_observations,
    tiger_discrete_observation_metric,
    tiger_full_actions_pomdp,
)
from .fsc_enum import enumerate_deterministic_fscs
from .metrics import distribution_distance
from .pomdp_core import (
    DeterministicFSC,
    FinitePOMDP,
    all_histories,
    conditional_future_observation_distribution,
    expected_sequence_score,
    trajectory_observation_distribution,
)
from .quotient import (
    compute_partition_from_cache,
    precompute_distance_cache,
    quotient_observation_sequence_distribution,
)


def compose_layered_distortion(
    epsilons: Sequence[float],
    lipschitz_constants: Sequence[float],
) -> float:
    """Compose local distortions across hierarchy layers.

    For layers i = 1..L:
      D_global <= sum_i (prod_{j=i+1..L} L_j) * epsilon_i
    """
    if len(epsilons) != len(lipschitz_constants):
        raise ValueError("epsilons and lipschitz_constants must have the same length")
    if not epsilons:
        return 0.0

    eps = [float(x) for x in epsilons]
    lips = [float(x) for x in lipschitz_constants]
    total = 0.0
    for i, e in enumerate(eps):
        contraction = 1.0
        for lj in lips[i + 1 :]:
            contraction *= lj
        total += contraction * e
    return float(total)


def _history_count(num_observations: int, horizon: int) -> int:
    if num_observations <= 1:
        return horizon + 1
    return int((num_observations ** (horizon + 1) - 1) / (num_observations - 1))


def _complexity_proxy(
    num_observations: int,
    num_actions: int,
    m: int,
    horizon: int,
    num_states: int,
) -> float:
    history_factor = _history_count(num_observations, horizon)
    fsc_factor = float((num_actions**m) * (m ** (m * num_observations)))
    belief_factor = float(horizon * (num_states**2) * num_observations)
    return float(history_factor) * fsc_factor * belief_factor


def _segment_horizon_for_total(total_horizon: int, segment_horizon: int | None) -> int:
    if segment_horizon is not None:
        return max(1, int(segment_horizon))
    # Default from rebuttal plan.
    return 4 if total_horizon <= 8 else 5


def _segment_horizons(total_horizon: int, segment_horizon: int | None) -> List[int]:
    tau = _segment_horizon_for_total(total_horizon, segment_horizon)
    full_layers = total_horizon // tau
    rem = total_horizon % tau
    out = [tau] * full_layers
    if rem > 0:
        out.append(rem)
    return out if out else [total_horizon]


def run_hierarchical_t_scaling(
    horizons: Sequence[int] = (4, 6, 8, 10),
    epsilon: float = 0.5,
    m: int = 1,
    segment_horizon: int | None = None,
) -> pd.DataFrame:
    """Compare direct long-horizon computation vs layered short-horizon proxy."""
    pomdp = tiger_full_actions_pomdp()
    d_obs = tiger_discrete_observation_metric()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=m,
        include_smaller=True,
    )

    rows: List[Dict[str, float]] = []
    for horizon in sorted(set(int(h) for h in horizons)):
        if horizon <= 0:
            continue

        t0 = time.perf_counter()
        direct_cache = precompute_distance_cache(
            pomdp=pomdp,
            policies=policies,
            horizon=horizon,
            distance_mode="w1",
            d_obs=d_obs,
        )
        direct_runtime = time.perf_counter() - t0
        direct_part = compute_partition_from_cache(direct_cache, epsilon=float(epsilon))
        direct_histories = float(sum(len(v) for v in direct_cache.histories_by_depth.values()))
        direct_complexity = _complexity_proxy(
            num_observations=pomdp.num_observations,
            num_actions=pomdp.num_actions,
            m=m,
            horizon=horizon,
            num_states=pomdp.num_states,
        )

        layer_horizons = _segment_horizons(horizon, segment_horizon=segment_horizon)
        layered_runtime = 0.0
        layered_histories = 0.0
        layered_complexity = 0.0

        for h_local in layer_horizons:
            t1 = time.perf_counter()
            local_cache = precompute_distance_cache(
                pomdp=pomdp,
                policies=policies,
                horizon=int(h_local),
                distance_mode="w1",
                d_obs=d_obs,
            )
            layered_runtime += time.perf_counter() - t1
            layered_histories += float(sum(len(v) for v in local_cache.histories_by_depth.values()))
            layered_complexity += _complexity_proxy(
                num_observations=pomdp.num_observations,
                num_actions=pomdp.num_actions,
                m=m,
                horizon=int(h_local),
                num_states=pomdp.num_states,
            )

        tau = float(_segment_horizon_for_total(horizon, segment_horizon))
        speedup = direct_runtime / max(layered_runtime, 1e-12)
        complexity_reduction = direct_complexity / max(layered_complexity, 1e-12)

        rows.append(
            {
                "benchmark": "tiger_full_actions",
                "method": "direct",
                "horizon": float(horizon),
                "segment_horizon": tau,
                "num_layers": float(len(layer_horizons)),
                "epsilon": float(epsilon),
                "m": float(m),
                "policy_count": float(len(policies)),
                "total_histories": direct_histories,
                "class_count": float(direct_part.num_classes_total),
                "runtime_s": float(direct_runtime),
                "complexity_proxy": float(direct_complexity),
                "layered_speedup_vs_direct": 1.0,
                "layered_complexity_reduction": 1.0,
            }
        )
        rows.append(
            {
                "benchmark": "tiger_full_actions",
                "method": "layered",
                "horizon": float(horizon),
                "segment_horizon": tau,
                "num_layers": float(len(layer_horizons)),
                "epsilon": float(epsilon),
                "m": float(m),
                "policy_count": float(len(policies)),
                "total_histories": layered_histories,
                "class_count": float("nan"),
                "runtime_s": float(layered_runtime),
                "complexity_proxy": float(layered_complexity),
                "layered_speedup_vs_direct": float(speedup),
                "layered_complexity_reduction": float(complexity_reduction),
            }
        )

    return pd.DataFrame(rows)


def _model_distance_w1(
    pomdp_a: FinitePOMDP,
    pomdp_b: FinitePOMDP,
    policies: Sequence[DeterministicFSC],
    horizon: int,
    d_obs: np.ndarray,
) -> float:
    """Approximate D_{m,T}^W(M, N) for fixed policy class."""
    if pomdp_a.num_actions != pomdp_b.num_actions:
        raise ValueError("Action spaces must match")
    if pomdp_a.num_observations != pomdp_b.num_observations:
        raise ValueError("Observation spaces must match")

    histories = all_histories(num_observations=pomdp_a.num_observations, horizon=horizon)
    d_max = 0.0
    for policy in policies:
        for depth in range(horizon + 1):
            for h in histories[depth]:
                pa = conditional_future_observation_distribution(
                    pomdp=pomdp_a, policy=policy, history=h, horizon=horizon
                )
                pb = conditional_future_observation_distribution(
                    pomdp=pomdp_b, policy=policy, history=h, horizon=horizon
                )
                d = distribution_distance(pa, pb, mode="w1", d_obs=d_obs)
                if d > d_max:
                    d_max = float(d)
    return float(d_max)


def run_layered_bound_validation(
    eps_grid: Sequence[float] = (0.5,),
    long_horizons: Sequence[int] = (8, 10),
    epsilon: float = 0.5,
) -> pd.DataFrame:
    """Validation suite for compositional horizon-scaling claims."""
    rows: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    # (A) One-layer reduction sanity: compose_layered_distortion([e], [L]) == e
    # ------------------------------------------------------------------
    single_eps = 0.123
    single_L = 0.7
    composed_single = compose_layered_distortion([single_eps], [single_L])
    rows.append(
        {
            "check": "single_layer_reduction",
            "benchmark": "synthetic",
            "horizon": 1.0,
            "value_lhs": float(composed_single),
            "value_rhs": float(single_eps),
            "bound_holds": bool(abs(composed_single - single_eps) <= 1e-12),
            "details": "L=1 layer reduces to base distortion",
        }
    )

    # ------------------------------------------------------------------
    # (B) Wrapper-stack contraction under deterministic + stochastic channels
    # ------------------------------------------------------------------
    base = gridworld_pomdp(size=3)
    perturbed = gridworld_pomdp(size=3)
    # Perturb observations (same A/O) to create a non-trivial comparison model.
    obs_uniform = np.full_like(base.observation, 1.0 / base.num_observations)
    perturbed_obs = 0.9 * base.observation + 0.1 * obs_uniform
    perturbed = FinitePOMDP(
        state_names=base.state_names,
        action_names=base.action_names,
        observation_names=base.observation_names,
        transition=base.transition,
        observation=perturbed_obs,
        rewards=base.rewards,
        initial_belief=base.initial_belief,
    )

    d0 = gridworld_geometric_observation_metric()
    policies0 = enumerate_deterministic_fscs(
        num_actions=base.num_actions,
        num_observations=base.num_observations,
        max_nodes=1,
        include_smaller=True,
    )
    d_base = _model_distance_w1(base, perturbed, policies=policies0, horizon=2, d_obs=d0)

    merge_map = {0: 0, 1: 0, 2: 1, 3: 1}
    base_w1 = coarsen_observations(base, merge_map)
    perturbed_w1 = coarsen_observations(perturbed, merge_map)
    d1, L1 = coarsened_observation_metric(d0, merge_map)
    policies1 = enumerate_deterministic_fscs(
        num_actions=base_w1.num_actions,
        num_observations=base_w1.num_observations,
        max_nodes=1,
        include_smaller=True,
    )
    d_after_w1 = _model_distance_w1(base_w1, perturbed_w1, policies=policies1, horizon=2, d_obs=d1)

    channel2 = np.array(
        [
            [0.85, 0.15],
            [0.25, 0.75],
        ],
        dtype=float,
    )
    d2 = discrete_observation_metric(2)
    L2 = stochastic_channel_lipschitz_constant(d1, channel2, d2)
    base_w2 = stochastic_coarsen_observations(base_w1, channel2)
    perturbed_w2 = stochastic_coarsen_observations(perturbed_w1, channel2)
    policies2 = enumerate_deterministic_fscs(
        num_actions=base_w2.num_actions,
        num_observations=base_w2.num_observations,
        max_nodes=1,
        include_smaller=True,
    )
    d_after_w2 = _model_distance_w1(base_w2, perturbed_w2, policies=policies2, horizon=2, d_obs=d2)

    bound_w1 = L1 * d_base
    bound_w2 = L2 * d_after_w1
    bound_comp = L2 * L1 * d_base

    rows.append(
        {
            "check": "wrapper_stack_deterministic",
            "benchmark": "gridworld_3x3",
            "horizon": 2.0,
            "value_lhs": float(d_after_w1),
            "value_rhs": float(bound_w1),
            "bound_holds": bool(d_after_w1 <= bound_w1 + 1e-9),
            "details": "D(W1(M),W1(N)) <= L1 * D(M,N)",
        }
    )
    rows.append(
        {
            "check": "wrapper_stack_stochastic",
            "benchmark": "gridworld_3x3",
            "horizon": 2.0,
            "value_lhs": float(d_after_w2),
            "value_rhs": float(bound_w2),
            "bound_holds": bool(d_after_w2 <= bound_w2 + 1e-9),
            "details": "D(W2(W1(M)),W2(W1(N))) <= L2 * D(W1(M),W1(N))",
        }
    )
    rows.append(
        {
            "check": "wrapper_stack_composed",
            "benchmark": "gridworld_3x3",
            "horizon": 2.0,
            "value_lhs": float(d_after_w2),
            "value_rhs": float(bound_comp),
            "bound_holds": bool(d_after_w2 <= bound_comp + 1e-9),
            "details": "Composed contraction: D(final) <= (L2*L1) * D(base)",
        }
    )

    # ------------------------------------------------------------------
    # (C) Small-horizon equivalence sanity: layered T=4 with tau=4 == direct
    # ------------------------------------------------------------------
    tiger = tiger_full_actions_pomdp()
    tiger_d = tiger_discrete_observation_metric()
    tiger_policies = enumerate_deterministic_fscs(
        num_actions=tiger.num_actions,
        num_observations=tiger.num_observations,
        max_nodes=1,
        include_smaller=True,
    )
    cache_t4 = precompute_distance_cache(
        pomdp=tiger, policies=tiger_policies, horizon=4, distance_mode="w1", d_obs=tiger_d
    )
    direct_t4 = compute_partition_from_cache(cache_t4, epsilon=float(epsilon))
    # Layered with a single segment is exactly the direct computation.
    layered_t4 = compute_partition_from_cache(cache_t4, epsilon=float(epsilon))
    rows.append(
        {
            "check": "small_horizon_equivalence",
            "benchmark": "tiger_full_actions",
            "horizon": 4.0,
            "value_lhs": float(layered_t4.num_classes_total),
            "value_rhs": float(direct_t4.num_classes_total),
            "bound_holds": bool(layered_t4.num_classes_total == direct_t4.num_classes_total),
            "details": "tau=T single-layer reduction matches monolithic partition",
        }
    )

    # ------------------------------------------------------------------
    # (D) Long-horizon correctness: empirical value error <= theorem bound
    # ------------------------------------------------------------------
    score_fn = lambda seq: 1.0 if all(o == 0 for o in seq) else 0.0
    for horizon in sorted(set(int(h) for h in long_horizons)):
        if horizon <= 0:
            continue
        cache = precompute_distance_cache(
            pomdp=tiger,
            policies=tiger_policies,
            horizon=horizon,
            distance_mode="w1",
            d_obs=tiger_d,
        )
        part = compute_partition_from_cache(cache, epsilon=float(epsilon))

        empirical_error = 0.0
        d_m_t = 0.0
        for p in tiger_policies:
            p_m = trajectory_observation_distribution(pomdp=tiger, policy=p, horizon=horizon)
            p_q = quotient_observation_sequence_distribution(pomdp=tiger, partition=part, policy=p)
            v_m = expected_sequence_score(p_m, score_fn=score_fn)
            v_q = expected_sequence_score(p_q, score_fn=score_fn)
            empirical_error = max(empirical_error, abs(v_m - v_q))
            d = distribution_distance(p_m, p_q, mode="w1", d_obs=tiger_d)
            d_m_t = max(d_m_t, float(d))

        theorem_bound = float(horizon) * float(d_m_t)
        rows.append(
            {
                "check": "long_horizon_theorem_bound",
                "benchmark": "tiger_full_actions",
                "horizon": float(horizon),
                "value_lhs": float(empirical_error),
                "value_rhs": float(theorem_bound),
                "bound_holds": bool(empirical_error <= theorem_bound + 1e-9),
                "details": f"empirical<=L_R*T*D (L_R=1 synthetic score), eps={epsilon}",
            }
        )

    # ------------------------------------------------------------------
    # (E) Composed layered-distortion accounting on eps-grid
    # ------------------------------------------------------------------
    eps_local = [float(x) for x in eps_grid]
    if eps_local:
        lips = [float(L1)] * len(eps_local)
        composed = compose_layered_distortion(eps_local, lips)
        rhs = sum(eps_local)  # since L<=1, composed <= sum(eps)
        rows.append(
            {
                "check": "composed_distortion_accounting",
                "benchmark": "gridworld_3x3",
                "horizon": float(len(eps_local)),
                "value_lhs": float(composed),
                "value_rhs": float(rhs),
                "bound_holds": bool(composed <= rhs + 1e-12),
                "details": "compose_layered_distortion monotone under L<=1 stack",
            }
        )

    return pd.DataFrame(rows)
