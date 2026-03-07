"""Benchmark POMDP instances used in the basic experimental package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np

from .pomdp_core import FinitePOMDP


def tiger_listen_only_pomdp(accuracy: float = 0.85) -> FinitePOMDP:
    """Two-state Tiger with listen-only action and binary observations."""
    states = ("tiger_left", "tiger_right")
    actions = ("listen",)
    observations = ("L", "R")

    num_s, num_a, num_o = 2, 1, 2

    transition = np.zeros((num_s, num_a, num_s), dtype=float)
    # Listening does not move tiger.
    transition[:, 0, :] = np.eye(num_s)

    observation = np.zeros((num_s, num_a, num_o), dtype=float)
    # Correct with given accuracy probability.
    observation[0, 0, 0] = accuracy
    observation[0, 0, 1] = 1.0 - accuracy
    observation[1, 0, 0] = 1.0 - accuracy
    observation[1, 0, 1] = accuracy

    # Non-lipschitz-style reward is not used for this benchmark track.
    rewards = np.zeros((num_s, num_a), dtype=float)

    b0 = np.array([0.5, 0.5], dtype=float)

    return FinitePOMDP(
        state_names=states,
        action_names=actions,
        observation_names=observations,
        transition=transition,
        observation=observation,
        rewards=rewards,
        initial_belief=b0,
    )


def tiger_full_actions_pomdp(accuracy: float = 0.85) -> FinitePOMDP:
    """Tiger with listen + open-left + open-right actions.

    Open actions reset the tiger position uniformly and produce uninformative observations.
    """

    states = ("tiger_left", "tiger_right")
    actions = ("listen", "open_left", "open_right")
    observations = ("L", "R")

    num_s, num_a, num_o = 2, 3, 2

    transition = np.zeros((num_s, num_a, num_s), dtype=float)
    # listen: state unchanged
    transition[:, 0, :] = np.eye(num_s)
    # open actions: reset to uniform state
    transition[:, 1, :] = 0.5
    transition[:, 2, :] = 0.5

    observation = np.zeros((num_s, num_a, num_o), dtype=float)
    # listen is informative
    observation[0, 0, 0] = accuracy
    observation[0, 0, 1] = 1.0 - accuracy
    observation[1, 0, 0] = 1.0 - accuracy
    observation[1, 0, 1] = accuracy
    # open actions are uninformative and full-support
    observation[:, 1, :] = 0.5
    observation[:, 2, :] = 0.5

    rewards = np.zeros((num_s, num_a), dtype=float)
    # listen cost
    rewards[:, 0] = -1.0
    # open_left: good if tiger is right
    rewards[0, 1] = -100.0
    rewards[1, 1] = 10.0
    # open_right: good if tiger is left
    rewards[0, 2] = 10.0
    rewards[1, 2] = -100.0

    b0 = np.array([0.5, 0.5], dtype=float)

    return FinitePOMDP(
        state_names=states,
        action_names=actions,
        observation_names=observations,
        transition=transition,
        observation=observation,
        rewards=rewards,
        initial_belief=b0,
    )


def tiger_discrete_observation_metric() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)


def _move(row: int, col: int, action: int, size: int) -> Tuple[int, int]:
    if action == 0:  # up
        return max(0, row - 1), col
    if action == 1:  # down
        return min(size - 1, row + 1), col
    if action == 2:  # left
        return row, max(0, col - 1)
    if action == 3:  # right
        return row, min(size - 1, col + 1)
    return row, col  # stay


def gridworld_pomdp(size: int = 3) -> FinitePOMDP:
    """Stochastic GridWorld with hidden position and noisy quadrant observations.

    Supports arbitrary grid sizes.  The agent occupies one of ``size * size``
    cells, takes 5 actions (up/down/left/right/stay), and receives a noisy
    observation indicating which quadrant (NW/NE/SW/SE) it is in.
    """

    states: List[str] = []
    for r in range(size):
        for c in range(size):
            states.append(f"s_{r}_{c}")
    actions = ("up", "down", "left", "right", "stay")
    observations = ("NW", "NE", "SW", "SE")

    num_s = len(states)
    num_a = len(actions)
    num_o = len(observations)

    transition = np.zeros((num_s, num_a, num_s), dtype=float)

    for s_idx in range(num_s):
        r, c = divmod(s_idx, size)
        for a in range(num_a):
            r1, c1 = _move(r, c, a, size)
            s1 = r1 * size + c1
            # intended move with 0.8, stay with 0.2
            transition[s_idx, a, s1] += 0.8
            transition[s_idx, a, s_idx] += 0.2

    observation = np.zeros((num_s, num_a, num_o), dtype=float)

    mid = size / 2.0

    def quadrant(rr: int, cc: int) -> int:
        top = 0 if rr < mid else 1
        left = 0 if cc < mid else 1
        if top == 0 and left == 0:
            return 0
        if top == 0 and left == 1:
            return 1
        if top == 1 and left == 0:
            return 2
        return 3

    for s_idx in range(num_s):
        r, c = divmod(s_idx, size)
        q = quadrant(r, c)
        for a in range(num_a):
            observation[s_idx, a, :] = 0.1
            observation[s_idx, a, q] = 0.7

    rewards = np.zeros((num_s, num_a), dtype=float)
    b0 = np.full(num_s, 1.0 / num_s, dtype=float)

    return FinitePOMDP(
        state_names=tuple(states),
        action_names=actions,
        observation_names=observations,
        transition=transition,
        observation=observation,
        rewards=rewards,
        initial_belief=b0,
    )


def gridworld_geometric_observation_metric() -> np.ndarray:
    """Normalized Manhattan distance on 2x2 quadrant coordinates."""
    coords = {
        0: (0, 0),  # NW
        1: (0, 1),  # NE
        2: (1, 0),  # SW
        3: (1, 1),  # SE
    }
    num_o = 4
    d = np.zeros((num_o, num_o), dtype=float)
    for i in range(num_o):
        for j in range(num_o):
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            d[i, j] = (abs(x1 - x2) + abs(y1 - y2)) / 2.0
    return d


def discrete_observation_metric(num_observations: int) -> np.ndarray:
    d = np.ones((num_observations, num_observations), dtype=float)
    np.fill_diagonal(d, 0.0)
    return d


def lipschitz_observation_score(num_observations: int) -> np.ndarray:
    """Simple [0, 1] score map to induce a 1-Lipschitz per-step objective."""
    if num_observations == 1:
        return np.array([0.0], dtype=float)
    return np.linspace(0.0, 1.0, num_observations, dtype=float)


def compute_reward_lipschitz_constant(pomdp: FinitePOMDP) -> float:
    """Reward range under discrete observation metric (L_R = max R - min R)."""
    return float(pomdp.rewards.max() - pomdp.rewards.min())


# ---------------------------------------------------------------------------
# Random structured POMDP
# ---------------------------------------------------------------------------

def random_structured_pomdp(
    num_states: int = 20,
    num_actions: int = 3,
    num_observations: int = 4,
    seed: int = 42,
    ergodicity_mix: float = 0.1,
) -> FinitePOMDP:
    """Generate a random POMDP with Dirichlet-sampled kernels.

    Transition rows are ``(1 - ergodicity_mix) * Dir(1) + ergodicity_mix / |S|``
    to ensure ergodicity.  Observation rows use ``Dir(2)`` for non-trivial
    observation structure.  Rewards are zero (abstraction-focused benchmark).
    """
    rng = np.random.default_rng(seed)

    states = tuple(f"s{i}" for i in range(num_states))
    actions = tuple(f"a{i}" for i in range(num_actions))
    observations = tuple(f"o{i}" for i in range(num_observations))

    num_s, num_a, num_o = num_states, num_actions, num_observations

    # Transition kernel
    transition = np.zeros((num_s, num_a, num_s), dtype=float)
    for s in range(num_s):
        for a in range(num_a):
            raw = rng.dirichlet(np.ones(num_s))
            transition[s, a, :] = (1.0 - ergodicity_mix) * raw + ergodicity_mix / num_s

    # Observation kernel (concentration > 1 for non-trivial structure)
    observation = np.zeros((num_s, num_a, num_o), dtype=float)
    for s in range(num_s):
        for a in range(num_a):
            observation[s, a, :] = rng.dirichlet(2.0 * np.ones(num_o))

    rewards = np.zeros((num_s, num_a), dtype=float)
    b0 = np.full(num_s, 1.0 / num_s, dtype=float)

    return FinitePOMDP(
        state_names=states,
        action_names=actions,
        observation_names=observations,
        transition=transition,
        observation=observation,
        rewards=rewards,
        initial_belief=b0,
    )


# ---------------------------------------------------------------------------
# Observation coarsening wrapper (for data-processing monotonicity validation)
# ---------------------------------------------------------------------------

def coarsen_observations(
    pomdp: FinitePOMDP,
    merge_map: Dict[int, int],
) -> FinitePOMDP:
    """Apply an observation-coarsening wrapper: merge observations via *merge_map*.

    Parameters
    ----------
    pomdp : FinitePOMDP
        Original POMDP.
    merge_map : dict
        Maps each old observation index to a new (coarser) observation index.
        E.g. ``{0: 0, 1: 0, 2: 1, 3: 1}`` merges obs 0,1 -> new 0 and obs 2,3 -> new 1.

    Returns
    -------
    FinitePOMDP with the coarsened observation space.
    """
    new_obs_count = max(merge_map.values()) + 1
    new_obs_names = tuple(f"co{i}" for i in range(new_obs_count))

    new_observation = np.zeros(
        (pomdp.num_states, pomdp.num_actions, new_obs_count), dtype=float,
    )
    for old_o, new_o in merge_map.items():
        new_observation[:, :, new_o] += pomdp.observation[:, :, old_o]

    return FinitePOMDP(
        state_names=pomdp.state_names,
        action_names=pomdp.action_names,
        observation_names=new_obs_names,
        transition=pomdp.transition,
        observation=new_observation,
        rewards=pomdp.rewards,
        initial_belief=pomdp.initial_belief,
    )


def coarsened_observation_metric(
    original_d_obs: np.ndarray,
    merge_map: Dict[int, int],
) -> Tuple[np.ndarray, float]:
    """Observation metric on the coarsened space and the Lipschitz constant L_C.

    For deterministic (many-to-one) merging the Lipschitz constant of the
    channel is 1.0 because every old observation maps to exactly one new one,
    so W_1 cannot increase.

    Returns ``(new_d_obs, L_C)``.
    """
    new_obs_count = max(merge_map.values()) + 1
    new_d_obs = np.zeros((new_obs_count, new_obs_count), dtype=float)
    for ni in range(new_obs_count):
        for nj in range(ni + 1, new_obs_count):
            old_i = [k for k, v in merge_map.items() if v == ni]
            old_j = [k for k, v in merge_map.items() if v == nj]
            min_d = min(original_d_obs[oi, oj] for oi in old_i for oj in old_j)
            new_d_obs[ni, nj] = min_d
            new_d_obs[nj, ni] = min_d
    L_C = 1.0  # deterministic coarsening
    return new_d_obs, L_C


def delta_covering_coarsen(
    d_obs: np.ndarray,
    delta: float,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Compute a greedy delta-covering of the observation space.

    Picks the first uncovered observation as a center, assigns all
    observations within distance ``delta`` to that group.

    Returns ``(new_d_obs, merge_map)`` where *merge_map* maps each
    original observation index to its covering-group index, and
    *new_d_obs* is the induced metric on the covering.
    """
    num_obs = d_obs.shape[0]
    assigned = [-1] * num_obs
    centers: List[int] = []

    for i in range(num_obs):
        if assigned[i] >= 0:
            continue
        group_id = len(centers)
        centers.append(i)
        assigned[i] = group_id
        for j in range(i + 1, num_obs):
            if assigned[j] < 0 and d_obs[i, j] <= delta:
                assigned[j] = group_id

    merge_map = {i: assigned[i] for i in range(num_obs)}
    new_d_obs, _ = coarsened_observation_metric(d_obs, merge_map)
    return new_d_obs, merge_map


def random_observation_metric(num_observations: int, seed: int = 42) -> np.ndarray:
    """Generate a valid metric matrix from random points in [0,1]^2."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 1.0, size=(num_observations, 2))
    d = np.zeros((num_observations, num_observations), dtype=float)
    for i in range(num_observations):
        for j in range(num_observations):
            d[i, j] = np.sqrt(np.sum((points[i] - points[j]) ** 2))
    # Normalize to [0, 1]
    if d.max() > 0:
        d /= d.max()
    return d


def tiger_full_actions_pomdp_custom_belief(
    accuracy: float = 0.85,
    b0: Sequence[float] | None = None,
) -> FinitePOMDP:
    """Tiger full-actions with a custom initial belief."""
    pomdp = tiger_full_actions_pomdp(accuracy=accuracy)
    if b0 is not None:
        b0_arr = np.array(b0, dtype=float)
        b0_arr /= b0_arr.sum()
        return FinitePOMDP(
            state_names=pomdp.state_names,
            action_names=pomdp.action_names,
            observation_names=pomdp.observation_names,
            transition=pomdp.transition,
            observation=pomdp.observation,
            rewards=pomdp.rewards,
            initial_belief=b0_arr,
        )
    return pomdp


# ---------------------------------------------------------------------------
# Hallway POMDP
# ---------------------------------------------------------------------------

def hallway_pomdp(
    length: int = 10,
    num_landmarks: int = 3,
    accuracy: float = 0.8,
) -> FinitePOMDP:
    """1D corridor with hidden position and noisy landmark observations.

    Parameters
    ----------
    length : int
        Number of positions in the corridor.
    num_landmarks : int
        Number of distinct landmark types.
    accuracy : float
        Probability of observing the correct landmark.
    """
    states = tuple(f"pos_{i}" for i in range(length))
    actions = ("left", "right", "stay")
    observations = tuple(f"lm_{i}" for i in range(num_landmarks))

    num_s, num_a, num_o = length, 3, num_landmarks

    # Assign landmark types cyclically
    landmark_for_pos = [i % num_landmarks for i in range(length)]

    # Transition: move with 0.8, stay with 0.2
    transition = np.zeros((num_s, num_a, num_s), dtype=float)
    for s in range(num_s):
        # left
        s_left = max(0, s - 1)
        transition[s, 0, s_left] += 0.8
        transition[s, 0, s] += 0.2
        # right
        s_right = min(length - 1, s + 1)
        transition[s, 1, s_right] += 0.8
        transition[s, 1, s] += 0.2
        # stay
        transition[s, 2, s] = 1.0

    # Observation: correct landmark with `accuracy`, uniform over others
    observation = np.zeros((num_s, num_a, num_o), dtype=float)
    noise = (1.0 - accuracy) / max(num_o - 1, 1)
    for s in range(num_s):
        true_lm = landmark_for_pos[s]
        for a in range(num_a):
            observation[s, a, :] = noise
            observation[s, a, true_lm] = accuracy

    rewards = np.zeros((num_s, num_a), dtype=float)
    b0 = np.full(num_s, 1.0 / num_s, dtype=float)

    return FinitePOMDP(
        state_names=states,
        action_names=actions,
        observation_names=observations,
        transition=transition,
        observation=observation,
        rewards=rewards,
        initial_belief=b0,
    )


# ---------------------------------------------------------------------------
# Network Monitoring POMDP
# ---------------------------------------------------------------------------

def network_monitoring_pomdp(
    num_nodes: int = 4,
    num_alerts: int = 3,
    failure_prob: float = 0.1,
    recovery_prob: float = 0.3,
) -> FinitePOMDP:
    """Network with hidden node failure states and noisy alert observations.

    Parameters
    ----------
    num_nodes : int
        Number of network nodes.  State space is 2^num_nodes (up/down per node).
        Keep num_nodes <= 5 for tractability.
    num_alerts : int
        Number of alert levels (observations).
    failure_prob : float
        Per-step probability of a healthy node failing.
    recovery_prob : float
        Per-step probability of a failed node recovering on its own.
    """
    num_s = 2 ** num_nodes
    num_a = num_nodes + 1  # probe_node_0 .. probe_node_{n-1}, do_nothing
    num_o = num_alerts

    states = tuple(f"s{i:0{num_nodes}b}" for i in range(num_s))
    actions = tuple(f"probe_{i}" for i in range(num_nodes)) + ("do_nothing",)
    observations = tuple(f"alert_{i}" for i in range(num_o))

    # Transition: each node independently fails/recovers
    transition = np.zeros((num_s, num_a, num_s), dtype=float)
    for s_idx in range(num_s):
        # Decode node states
        node_up = [(s_idx >> k) & 1 for k in range(num_nodes)]  # 1=up, 0=down
        # Compute per-node transition probabilities
        prob_up = []
        for k in range(num_nodes):
            if node_up[k] == 1:
                prob_up.append(1.0 - failure_prob)  # stays up
            else:
                prob_up.append(recovery_prob)  # recovers to up

        # Build joint transition (same for all actions — probing doesn't change state)
        for s_next in range(num_s):
            p = 1.0
            for k in range(num_nodes):
                next_up = (s_next >> k) & 1
                if next_up == 1:
                    p *= prob_up[k]
                else:
                    p *= 1.0 - prob_up[k]
            for a in range(num_a):
                transition[s_idx, a, s_next] = p

    # Observation: depends on probed node's state
    observation = np.zeros((num_s, num_a, num_o), dtype=float)
    for s_idx in range(num_s):
        node_up = [(s_idx >> k) & 1 for k in range(num_nodes)]
        for a in range(num_a):
            if a < num_nodes:
                # Probing node a
                if node_up[a] == 1:
                    # Healthy: highest alert level most likely
                    observation[s_idx, a, -1] = 0.8
                    remaining = 0.2 / max(num_o - 1, 1)
                    for o in range(num_o - 1):
                        observation[s_idx, a, o] = remaining
                else:
                    # Failed: lowest alert level most likely
                    observation[s_idx, a, 0] = 0.8
                    remaining = 0.2 / max(num_o - 1, 1)
                    for o in range(1, num_o):
                        observation[s_idx, a, o] = remaining
            else:
                # do_nothing: uninformative observation
                observation[s_idx, a, :] = 1.0 / num_o

    rewards = np.zeros((num_s, num_a), dtype=float)
    b0 = np.full(num_s, 1.0 / num_s, dtype=float)

    return FinitePOMDP(
        state_names=states,
        action_names=actions,
        observation_names=observations,
        transition=transition,
        observation=observation,
        rewards=rewards,
        initial_belief=b0,
    )


def stochastic_coarsen_observations(
    pomdp: FinitePOMDP,
    channel: np.ndarray,
) -> FinitePOMDP:
    """Apply a stochastic observation channel: new_obs ~ channel[old_obs, :].

    Parameters
    ----------
    pomdp : FinitePOMDP
        Original POMDP.
    channel : np.ndarray
        Shape [num_old_obs, num_new_obs].  Each row sums to 1.
    """
    num_new_obs = channel.shape[1]
    new_obs_names = tuple(f"so{i}" for i in range(num_new_obs))
    new_observation = np.einsum("sao,on->san", pomdp.observation, channel)
    return FinitePOMDP(
        state_names=pomdp.state_names,
        action_names=pomdp.action_names,
        observation_names=new_obs_names,
        transition=pomdp.transition,
        observation=new_observation,
        rewards=pomdp.rewards,
        initial_belief=pomdp.initial_belief,
    )


def stochastic_channel_lipschitz_constant(
    original_d_obs: np.ndarray,
    channel: np.ndarray,
    new_d_obs: np.ndarray,
) -> float:
    """Lipschitz constant L_C of a stochastic observation channel.

    L_C = max_{o1!=o2} W1(channel[o1,:], channel[o2,:]; new_d_obs) / d_obs(o1,o2).
    """
    from .metrics import wasserstein_distance

    num_old = channel.shape[0]
    l_c = 0.0
    for o1 in range(num_old):
        for o2 in range(o1 + 1, num_old):
            d_old = original_d_obs[o1, o2]
            if d_old <= 0:
                continue
            p = {(i,): float(channel[o1, i]) for i in range(channel.shape[1]) if channel[o1, i] > 0}
            q = {(i,): float(channel[o2, i]) for i in range(channel.shape[1]) if channel[o2, i] > 0}
            w1 = wasserstein_distance(p, q, d_obs=new_d_obs)
            l_c = max(l_c, w1 / d_old)
    return l_c
