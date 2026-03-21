"""Deterministic/stochastic FSC enumeration and sampling helpers."""

from __future__ import annotations

from itertools import product
from typing import Iterable, List, Sequence

import numpy as np

from .pomdp_core import DeterministicFSC, StochasticFSC


def enumerate_deterministic_fscs(
    num_actions: int,
    num_observations: int,
    max_nodes: int,
    include_smaller: bool = True,
) -> List[DeterministicFSC]:
    """Enumerate all deterministic FSCs with <= max_nodes nodes.

    Node 0 is used as initial node for all controllers.
    """
    policies: List[DeterministicFSC] = []
    min_nodes = 1 if include_smaller else max_nodes
    for num_nodes in range(min_nodes, max_nodes + 1):
        for action_assignment in product(range(num_actions), repeat=num_nodes):
            # For each (node, obs) pair, choose the next node.
            for transition_flat in product(range(num_nodes), repeat=num_nodes * num_observations):
                trans_rows = []
                idx = 0
                for _node in range(num_nodes):
                    row = tuple(int(transition_flat[idx + o]) for o in range(num_observations))
                    trans_rows.append(row)
                    idx += num_observations
                policies.append(
                    DeterministicFSC(
                        num_nodes=num_nodes,
                        action_for_node=tuple(int(a) for a in action_assignment),
                        next_node_for_observation=tuple(trans_rows),
                        initial_node=0,
                    )
                )
    return policies


def sample_stochastic_fscs(
    num_actions: int,
    num_observations: int,
    num_nodes: int,
    num_samples: int,
    seed: int,
) -> List[StochasticFSC]:
    rng = np.random.default_rng(seed)
    policies: List[StochasticFSC] = []

    for _ in range(num_samples):
        alpha = np.zeros((num_nodes, num_actions), dtype=float)
        beta = np.zeros((num_nodes, num_observations, num_nodes), dtype=float)

        for n in range(num_nodes):
            alpha[n] = rng.dirichlet(np.ones(num_actions, dtype=float))
            for o in range(num_observations):
                beta[n, o] = rng.dirichlet(np.ones(num_nodes, dtype=float))

        policies.append(
            StochasticFSC(
                num_nodes=num_nodes,
                alpha=alpha,
                beta=beta,
                initial_node=0,
            )
        )

    return policies


def enumerate_clock_aware_deterministic_fscs(
    num_actions: int,
    num_observations: int,
    m: int,
    horizon: int,
) -> List[DeterministicFSC]:
    """Enumerate all deterministic clock-aware FSCs with *m* internal nodes.

    A clock-aware FSC has stage-indexed parameters (alpha_tau, beta_tau) for
    tau = 0, ..., T-1.  Each stage has m nodes; alpha_tau maps each node to an
    action and beta_tau maps each (node, observation) pair to a next node.

    The returned DeterministicFSC objects are "unrolled" with m*T nodes total.
    Node index for stage tau, internal node n is tau*m + n.  At the final stage
    (tau = T-1), next-node transitions are set to 0 (unused).
    """
    total_nodes = m * horizon
    if total_nodes == 0:
        return []

    # Build per-stage parameter ranges
    # alpha_tau: m actions => num_actions^m choices
    # beta_tau: m*num_observations next-nodes in {0..m-1} => m^(m*num_observations) choices
    # But at stage T-1, beta is irrelevant (no more transitions)
    stage_alpha_options = list(product(range(num_actions), repeat=m))
    stage_beta_options = list(product(range(m), repeat=m * num_observations))

    # Build stage parameter iterators
    # Each stage has (alpha, beta) except last stage which only has alpha
    stage_param_lists = []
    for tau in range(horizon):
        if tau < horizon - 1:
            stage_param_lists.append(list(product(stage_alpha_options, stage_beta_options)))
        else:
            # Last stage: beta doesn't matter, use single dummy
            dummy_beta = (0,) * (m * num_observations)
            stage_param_lists.append([(alpha, dummy_beta) for alpha in stage_alpha_options])

    policies: List[DeterministicFSC] = []
    for stage_params in product(*stage_param_lists):
        action_for_node = [0] * total_nodes
        next_node_for_obs = [tuple(0 for _ in range(num_observations))] * total_nodes

        for tau, (alpha_tau, beta_tau) in enumerate(stage_params):
            for n in range(m):
                node_idx = tau * m + n
                action_for_node[node_idx] = alpha_tau[n]
                if tau < horizon - 1:
                    row = tuple(
                        (tau + 1) * m + beta_tau[n * num_observations + o]
                        for o in range(num_observations)
                    )
                else:
                    row = tuple(0 for _ in range(num_observations))
                next_node_for_obs[node_idx] = row

        policies.append(
            DeterministicFSC(
                num_nodes=total_nodes,
                action_for_node=tuple(action_for_node),
                next_node_for_observation=tuple(next_node_for_obs),
                initial_node=0,
            )
        )

    return policies


def policy_label(policy: DeterministicFSC) -> str:
    action_bits = "".join(str(a) for a in policy.action_for_node)
    trans_bits = "|".join("".join(str(v) for v in row) for row in policy.next_node_for_observation)
    return f"n{policy.num_nodes}:a{action_bits}:t{trans_bits}"
