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


def policy_label(policy: DeterministicFSC) -> str:
    action_bits = "".join(str(a) for a in policy.action_for_node)
    trans_bits = "|".join("".join(str(v) for v in row) for row in policy.next_node_for_observation)
    return f"n{policy.num_nodes}:a{action_bits}:t{trans_bits}"
