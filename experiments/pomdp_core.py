"""Core finite-horizon POMDP and FSC simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

History = Tuple[int, ...]
ObsSeq = Tuple[int, ...]


@dataclass(frozen=True)
class FinitePOMDP:
    """Finite POMDP with dense transition and observation kernels.

    Shapes:
      - transition: [num_states, num_actions, num_states]
      - observation: [num_states, num_actions, num_observations]
      - rewards: [num_states, num_actions]
      - initial_belief: [num_states]
    """

    state_names: Tuple[str, ...]
    action_names: Tuple[str, ...]
    observation_names: Tuple[str, ...]
    transition: np.ndarray
    observation: np.ndarray
    rewards: np.ndarray
    initial_belief: np.ndarray

    @property
    def num_states(self) -> int:
        return len(self.state_names)

    @property
    def num_actions(self) -> int:
        return len(self.action_names)

    @property
    def num_observations(self) -> int:
        return len(self.observation_names)


class FSCPolicy:
    """Base policy interface for FSC policies."""

    num_nodes: int
    initial_node: int

    def action_distribution(self, node: int) -> np.ndarray:
        raise NotImplementedError

    def node_transition_distribution(self, node: int, observation: int) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class DeterministicFSC(FSCPolicy):
    num_nodes: int
    action_for_node: Tuple[int, ...]
    next_node_for_observation: Tuple[Tuple[int, ...], ...]
    initial_node: int = 0

    def action_distribution(self, node: int) -> np.ndarray:
        probs = np.zeros(max(self.action_for_node) + 1, dtype=float)
        probs[self.action_for_node[node]] = 1.0
        return probs

    def action_index(self, node: int) -> int:
        return self.action_for_node[node]

    def next_node_index(self, node: int, observation: int) -> int:
        return self.next_node_for_observation[node][observation]


@dataclass(frozen=True)
class StochasticFSC(FSCPolicy):
    num_nodes: int
    alpha: np.ndarray  # [num_nodes, num_actions]
    beta: np.ndarray  # [num_nodes, num_observations, num_nodes]
    initial_node: int = 0

    def action_distribution(self, node: int) -> np.ndarray:
        return self.alpha[node]

    def node_transition_distribution(self, node: int, observation: int) -> np.ndarray:
        return self.beta[node, observation]


def all_histories(num_observations: int, horizon: int) -> Dict[int, List[History]]:
    """Enumerate all observation histories grouped by depth."""
    histories: Dict[int, List[History]] = {0: [()]}
    for depth in range(1, horizon + 1):
        prev = histories[depth - 1]
        curr: List[History] = []
        for h in prev:
            for obs in range(num_observations):
                curr.append(h + (obs,))
        histories[depth] = curr
    return histories


def _one_step_joint_by_observation(
    pomdp: FinitePOMDP,
    policy: FSCPolicy,
    joint_state_node: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Return unnormalized next joint distributions for each emitted observation."""
    num_s = pomdp.num_states
    num_a = pomdp.num_actions
    num_o = pomdp.num_observations
    num_n = policy.num_nodes

    out = {obs: np.zeros((num_s, num_n), dtype=float) for obs in range(num_o)}

    for s in range(num_s):
        for n in range(num_n):
            mass = joint_state_node[s, n]
            if mass <= 0.0:
                continue

            if isinstance(policy, DeterministicFSC):
                action_probs = np.zeros(num_a, dtype=float)
                action_probs[policy.action_index(n)] = 1.0
            else:
                action_probs = policy.action_distribution(n)

            for a in range(num_a):
                pa = action_probs[a]
                if pa <= 0.0:
                    continue

                for s_next in range(num_s):
                    pt = pomdp.transition[s, a, s_next]
                    if pt <= 0.0:
                        continue

                    for obs in range(num_o):
                        pz = pomdp.observation[s_next, a, obs]
                        if pz <= 0.0:
                            continue

                        if isinstance(policy, DeterministicFSC):
                            n_next = policy.next_node_index(n, obs)
                            out[obs][s_next, n_next] += mass * pa * pt * pz
                        else:
                            node_probs = policy.node_transition_distribution(n, obs)
                            for n_next in range(num_n):
                                pn = node_probs[n_next]
                                if pn <= 0.0:
                                    continue
                                out[obs][s_next, n_next] += mass * pa * pt * pz * pn

    return out


def joint_posterior_after_history(
    pomdp: FinitePOMDP,
    policy: FSCPolicy,
    history: History,
) -> np.ndarray:
    """Posterior over (state, node) after conditioning on observation history."""
    joint = np.zeros((pomdp.num_states, policy.num_nodes), dtype=float)
    joint[:, policy.initial_node] = pomdp.initial_belief

    for obs in history:
        by_obs = _one_step_joint_by_observation(pomdp, policy, joint)
        next_joint = by_obs[obs]
        z = float(next_joint.sum())
        if z <= 0.0:
            # Safe fallback for impossible histories; avoided by benchmark construction.
            joint = np.full_like(joint, 1.0 / (pomdp.num_states * policy.num_nodes))
        else:
            joint = next_joint / z

    return joint


def belief_after_history(
    pomdp: FinitePOMDP,
    policy: FSCPolicy,
    history: History,
) -> np.ndarray:
    joint = joint_posterior_after_history(pomdp, policy, history)
    belief = joint.sum(axis=1)
    z = float(belief.sum())
    if z <= 0.0:
        return np.full(pomdp.num_states, 1.0 / pomdp.num_states, dtype=float)
    return belief / z


def _future_observation_distribution_from_joint(
    pomdp: FinitePOMDP,
    policy: FSCPolicy,
    joint: np.ndarray,
    steps_remaining: int,
) -> Dict[ObsSeq, float]:
    if steps_remaining == 0:
        return {(): 1.0}

    by_obs = _one_step_joint_by_observation(pomdp, policy, joint)
    dist: Dict[ObsSeq, float] = {}
    for obs, next_joint_unnorm in by_obs.items():
        p_obs = float(next_joint_unnorm.sum())
        if p_obs <= 0.0:
            continue
        next_joint = next_joint_unnorm / p_obs
        suffix_dist = _future_observation_distribution_from_joint(
            pomdp=pomdp,
            policy=policy,
            joint=next_joint,
            steps_remaining=steps_remaining - 1,
        )
        for suffix, p_suffix in suffix_dist.items():
            key = (obs,) + suffix
            dist[key] = dist.get(key, 0.0) + p_obs * p_suffix

    return dist


def conditional_future_observation_distribution(
    pomdp: FinitePOMDP,
    policy: FSCPolicy,
    history: History,
    horizon: int,
) -> Dict[ObsSeq, float]:
    """Compute P(O_{t+1:T} | history) exactly for finite horizon T."""
    depth = len(history)
    if depth > horizon:
        raise ValueError("History length exceeds horizon")
    steps_remaining = horizon - depth
    joint = joint_posterior_after_history(pomdp, policy, history)
    return _future_observation_distribution_from_joint(
        pomdp=pomdp,
        policy=policy,
        joint=joint,
        steps_remaining=steps_remaining,
    )


def trajectory_observation_distribution(
    pomdp: FinitePOMDP,
    policy: FSCPolicy,
    horizon: int,
) -> Dict[ObsSeq, float]:
    """Compute P(O_{1:T}) under policy."""
    return conditional_future_observation_distribution(
        pomdp=pomdp,
        policy=policy,
        history=(),
        horizon=horizon,
    )


def observation_metric_sum(seq_a: Sequence[int], seq_b: Sequence[int], d_obs: np.ndarray) -> float:
    if len(seq_a) != len(seq_b):
        raise ValueError("Observation sequences must have equal length")
    return float(sum(d_obs[int(a), int(b)] for a, b in zip(seq_a, seq_b)))


def expected_per_step_observation_score(
    sequence_distribution: Mapping[ObsSeq, float],
    score_by_observation: Sequence[float],
) -> float:
    """E[sum_t score(o_t)] for a sequence distribution."""
    total = 0.0
    for seq, p in sequence_distribution.items():
        total += p * sum(score_by_observation[o] for o in seq)
    return float(total)


def expected_sequence_score(
    sequence_distribution: Mapping[ObsSeq, float],
    score_fn,
) -> float:
    """E[score_fn(sequence)] for an observation-sequence distribution."""
    return float(sum(p * float(score_fn(seq)) for seq, p in sequence_distribution.items()))


def normalize_distribution(dist: Mapping[ObsSeq, float]) -> Dict[ObsSeq, float]:
    z = float(sum(dist.values()))
    if z <= 0.0:
        return {}
    return {k: v / z for k, v in dist.items() if v > 0.0}
