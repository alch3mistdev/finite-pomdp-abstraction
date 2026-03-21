"""Exact m=1 clock-aware/open-loop utilities for theory-first paper tables."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .metrics import transport_lp_value
from .pomdp_core import FinitePOMDP, History, ObsSeq

OpenLoopActions = Tuple[int, ...]
ObservableObjective = Callable[[OpenLoopActions, ObsSeq], float]

IMPOSSIBLE_BELIEF_ID = 0
ROUND_DECIMALS = 12


@dataclass(frozen=True)
class OpenLoopPolicy:
    actions: OpenLoopActions

    def label(self, pomdp: FinitePOMDP) -> str:
        return "".join(pomdp.action_names[a] for a in self.actions)


@dataclass(frozen=True)
class ExactHistoryPartition:
    family_name: str
    horizon: int
    histories_by_depth: Dict[int, Tuple[History, ...]]
    labels_by_depth: Dict[int, np.ndarray]
    class_histories: Dict[int, Tuple[History, ...]]
    history_to_class: Dict[History, int]
    class_depth: Dict[int, int]

    @property
    def num_classes_total(self) -> int:
        return len(self.class_histories)

    @property
    def total_histories(self) -> int:
        return sum(len(v) for v in self.histories_by_depth.values())

    def representative(self, class_id: int) -> History:
        return self.class_histories[class_id][0]


class BeliefRegistry:
    """Interns beliefs and caches one-step Bayes updates."""

    def __init__(self, pomdp: FinitePOMDP):
        self.pomdp = pomdp
        self.keys: List[Tuple[float, ...] | None] = [None]
        self.vectors: List[np.ndarray | None] = [None]
        self._index: Dict[Tuple[float, ...], int] = {}
        self._obs_probs: List[np.ndarray | None] = [None]
        self._next_ids: List[List[np.ndarray]] = [
            [np.array([IMPOSSIBLE_BELIEF_ID], dtype=np.int32) for _ in range(pomdp.num_observations)]
            for _ in range(pomdp.num_actions)
        ]

    def _key(self, belief: np.ndarray) -> Tuple[float, ...]:
        return tuple(np.round(belief.astype(float), ROUND_DECIMALS).tolist())

    def register(self, belief: np.ndarray | None) -> int:
        if belief is None:
            return IMPOSSIBLE_BELIEF_ID
        key = self._key(belief)
        existing = self._index.get(key)
        if existing is not None:
            return existing
        belief_id = len(self.keys)
        self._index[key] = belief_id
        self.keys.append(key)
        self.vectors.append(np.asarray(belief, dtype=float))
        self._obs_probs.append(None)
        for action in range(self.pomdp.num_actions):
            for obs in range(self.pomdp.num_observations):
                arr = self._next_ids[action][obs]
                self._next_ids[action][obs] = np.pad(arr, (0, 1), constant_values=-1)
        return belief_id

    def vector(self, belief_id: int) -> np.ndarray | None:
        return self.vectors[belief_id]

    def observation_probabilities(self, belief_id: int, action: int) -> np.ndarray:
        if belief_id == IMPOSSIBLE_BELIEF_ID:
            out = np.zeros(self.pomdp.num_observations, dtype=float)
            out[0] = 1.0
            return out
        cached = self._obs_probs[belief_id]
        if cached is None:
            belief = self.vectors[belief_id]
            assert belief is not None
            per_action = np.zeros((self.pomdp.num_actions, self.pomdp.num_observations), dtype=float)
            for a in range(self.pomdp.num_actions):
                next_state = belief @ self.pomdp.transition[:, a, :]
                per_action[a] = next_state @ self.pomdp.observation[:, a, :]
            self._obs_probs[belief_id] = per_action
            cached = per_action
        return cached[action]

    def next_belief_id(self, belief_id: int, action: int, obs: int) -> int:
        if belief_id == IMPOSSIBLE_BELIEF_ID:
            return IMPOSSIBLE_BELIEF_ID
        cached = self._next_ids[action][obs][belief_id]
        if cached != -1:
            return int(cached)

        belief = self.vectors[belief_id]
        assert belief is not None
        next_state = belief @ self.pomdp.transition[:, action, :]
        weighted = next_state * self.pomdp.observation[:, action, obs]
        z = float(weighted.sum())
        if z <= 0.0:
            next_id = IMPOSSIBLE_BELIEF_ID
        else:
            next_id = self.register(weighted / z)
        self._next_ids[action][obs][belief_id] = next_id
        return next_id

    def ensure_transitions(self, belief_ids: Iterable[int]) -> None:
        for belief_id in belief_ids:
            if belief_id == IMPOSSIBLE_BELIEF_ID:
                continue
            for action in range(self.pomdp.num_actions):
                self.observation_probabilities(belief_id, action)
                for obs in range(self.pomdp.num_observations):
                    self.next_belief_id(belief_id, action, obs)


def _all_histories(num_observations: int, depth: int) -> Tuple[History, ...]:
    if depth == 0:
        return ((),)
    return tuple(tuple(int(o) for o in seq) for seq in product(range(num_observations), repeat=depth))


def _constant_prefix_row_index(num_actions: int, depth: int, action: int) -> int:
    idx = 0
    for _ in range(depth):
        idx = idx * num_actions + action
    return idx


def enumerate_clock_aware_open_loop_policies(num_actions: int, horizon: int) -> List[OpenLoopPolicy]:
    return [OpenLoopPolicy(tuple(int(a) for a in seq)) for seq in product(range(num_actions), repeat=horizon)]


def enumerate_operational_open_loop_policies(num_actions: int, horizon: int) -> List[OpenLoopPolicy]:
    return [OpenLoopPolicy(tuple([int(action)] * horizon)) for action in range(num_actions)]


def build_prefix_history_tables(
    pomdp: FinitePOMDP,
    horizon: int,
) -> Tuple[BeliefRegistry, List[np.ndarray]]:
    """Return belief-id tables of shape [num_prefixes, num_histories] per depth."""

    registry = BeliefRegistry(pomdp)
    init_id = registry.register(pomdp.initial_belief)
    tables = [np.array([[init_id]], dtype=np.int32)]

    for _depth in range(horizon):
        curr = tables[-1]
        registry.ensure_transitions(np.unique(curr))
        rows, cols = curr.shape
        nxt = np.empty((rows * pomdp.num_actions, cols * pomdp.num_observations), dtype=np.int32)
        for action in range(pomdp.num_actions):
            row_slice = slice(action * rows, (action + 1) * rows)
            for obs in range(pomdp.num_observations):
                nxt[row_slice, obs::pomdp.num_observations] = registry._next_ids[action][obs][curr]
        tables.append(nxt)

    return registry, tables


def compute_belief_equivalence_classes(
    pomdp: FinitePOMDP,
    registry: BeliefRegistry,
    tables: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Class id per belief id at each depth for the open-loop clock-aware family."""

    horizon = len(tables) - 1
    class_arrays: List[np.ndarray] = [np.zeros(len(registry.keys), dtype=np.int32) for _ in range(horizon + 1)]
    class_arrays[horizon] = np.ones(len(registry.keys), dtype=np.int32)
    class_arrays[horizon][IMPOSSIBLE_BELIEF_ID] = 0

    for depth in range(horizon - 1, -1, -1):
        registry.ensure_transitions(np.unique(tables[depth]))
        sig_to_class: Dict[Tuple[object, ...], int] = {}
        arr = np.zeros(len(registry.keys), dtype=np.int32)
        next_classes = class_arrays[depth + 1]
        for belief_id in np.unique(tables[depth]):
            if belief_id == IMPOSSIBLE_BELIEF_ID:
                continue
            signature: List[object] = []
            for action in range(pomdp.num_actions):
                obs_probs = np.round(registry.observation_probabilities(int(belief_id), action), ROUND_DECIMALS)
                per_obs: List[Tuple[float, int]] = []
                for obs in range(pomdp.num_observations):
                    next_belief = registry.next_belief_id(int(belief_id), action, obs)
                    per_obs.append((float(obs_probs[obs]), int(next_classes[next_belief])))
                signature.append(tuple(per_obs))
            key = tuple(signature)
            cls = sig_to_class.setdefault(key, len(sig_to_class) + 1)
            arr[int(belief_id)] = cls
        class_arrays[depth] = arr
    return class_arrays


def _labels_from_belief_matrix(class_matrix: np.ndarray) -> np.ndarray:
    """Cluster history columns by exact equality without materializing a transpose copy."""

    num_histories = class_matrix.shape[1]
    hash_a = np.full(num_histories, np.uint64(1469598103934665603), dtype=np.uint64)
    hash_b = np.full(num_histories, np.uint64(1099511628211), dtype=np.uint64)
    for row in class_matrix:
        vals = row.astype(np.uint64) + np.uint64(1)
        hash_a = (hash_a * np.uint64(11400714819323198485)) ^ vals
        hash_b = (hash_b * np.uint64(14029467366897019727)) ^ (vals * np.uint64(1609587929392839161))

    labels = np.empty(num_histories, dtype=np.int32)
    buckets: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    next_label = 0
    for idx in range(num_histories):
        key = (int(hash_a[idx]), int(hash_b[idx]))
        bucket = buckets.setdefault(key, [])
        assigned = None
        for label, rep_idx in bucket:
            if np.array_equal(class_matrix[:, idx], class_matrix[:, rep_idx]):
                assigned = label
                break
        if assigned is None:
            assigned = next_label
            next_label += 1
            bucket.append((assigned, idx))
        labels[idx] = assigned
    return labels


def build_history_partition(
    pomdp: FinitePOMDP,
    tables: Sequence[np.ndarray],
    belief_classes: Sequence[np.ndarray],
    family_name: str,
) -> ExactHistoryPartition:
    horizon = len(tables) - 1
    histories_by_depth: Dict[int, Tuple[History, ...]] = {}
    labels_by_depth: Dict[int, np.ndarray] = {}
    history_to_class: Dict[History, int] = {}
    class_histories: Dict[int, Tuple[History, ...]] = {}
    class_depth: Dict[int, int] = {}

    next_class_id = 0
    num_actions = pomdp.num_actions
    for depth, table in enumerate(tables):
        histories = _all_histories(pomdp.num_observations, depth)
        histories_by_depth[depth] = histories
        belief_matrix = belief_classes[depth][table]
        if family_name == "clk":
            row_indices = np.arange(table.shape[0], dtype=np.int32)
        elif family_name == "op":
            row_indices = np.array(
                [_constant_prefix_row_index(num_actions, depth, action) for action in range(num_actions)],
                dtype=np.int32,
            )
        else:
            raise ValueError(f"unknown family {family_name}")
        depth_labels = _labels_from_belief_matrix(belief_matrix[row_indices, :])
        remap: Dict[int, int] = {}
        for local in depth_labels.tolist():
            if local not in remap:
                remap[local] = next_class_id
                next_class_id += 1
        global_labels = np.array([remap[int(local)] for local in depth_labels], dtype=np.int32)
        labels_by_depth[depth] = global_labels
        for local, global_id in remap.items():
            members = tuple(histories[i] for i, lbl in enumerate(depth_labels.tolist()) if int(lbl) == local)
            class_histories[global_id] = members
            class_depth[global_id] = depth
            for history in members:
                history_to_class[history] = global_id
    return ExactHistoryPartition(
        family_name=family_name,
        horizon=horizon,
        histories_by_depth=histories_by_depth,
        labels_by_depth=labels_by_depth,
        class_histories=class_histories,
        history_to_class=history_to_class,
        class_depth=class_depth,
    )


def build_future_distribution_cache(
    pomdp: FinitePOMDP,
    registry: BeliefRegistry,
    horizon: int,
):
    @lru_cache(maxsize=None)
    def future_distribution(belief_id: int, action_seq: OpenLoopActions) -> Dict[ObsSeq, float]:
        if not action_seq:
            return {(): 1.0}
        if belief_id == IMPOSSIBLE_BELIEF_ID:
            return {(0,) * len(action_seq): 1.0}
        action = int(action_seq[0])
        obs_probs = registry.observation_probabilities(int(belief_id), action)
        out: Dict[ObsSeq, float] = {}
        for obs in range(pomdp.num_observations):
            p_obs = float(obs_probs[obs])
            if p_obs <= 0.0:
                continue
            next_belief = registry.next_belief_id(int(belief_id), action, obs)
            suffix = future_distribution(next_belief, tuple(int(a) for a in action_seq[1:]))
            for tail, p_tail in suffix.items():
                key = (obs,) + tail
                out[key] = out.get(key, 0.0) + p_obs * p_tail
        return out

    return future_distribution


def observation_objective_score(pomdp: FinitePOMDP) -> ObservableObjective:
    if pomdp.num_observations == 1:
        values = np.array([0.0], dtype=float)
    else:
        values = np.linspace(0.0, 1.0, pomdp.num_observations, dtype=float)

    def objective(_actions: OpenLoopActions, obs_seq: ObsSeq) -> float:
        return float(sum(values[int(obs)] for obs in obs_seq))

    return objective


def observation_score_step_rewards(pomdp: FinitePOMDP) -> np.ndarray:
    if pomdp.num_observations == 1:
        values = np.array([0.0], dtype=float)
    else:
        values = np.linspace(0.0, 1.0, pomdp.num_observations, dtype=float)
    return np.tile(values[None, :], (pomdp.num_actions, 1))


def action_observation_objective(pomdp: FinitePOMDP) -> ObservableObjective:
    if pomdp.num_observations == 2 and pomdp.num_actions >= 3:
        preferred = {
            0: 2,  # observation "left" -> open_right
            1: 1,  # observation "right" -> open_left
        }
    else:
        preferred = {obs: (obs % pomdp.num_actions) for obs in range(pomdp.num_observations)}

    def objective(actions: OpenLoopActions, obs_seq: ObsSeq) -> float:
        total = 0.0
        for action, obs in zip(actions, obs_seq):
            total += 1.0 if int(action) == preferred[int(obs)] else 0.0
        return total

    return objective


def action_observation_step_rewards(pomdp: FinitePOMDP) -> np.ndarray:
    if pomdp.num_observations == 2 and pomdp.num_actions >= 3:
        preferred = {
            0: 2,
            1: 1,
        }
    else:
        preferred = {obs: (obs % pomdp.num_actions) for obs in range(pomdp.num_observations)}
    rewards = np.zeros((pomdp.num_actions, pomdp.num_observations), dtype=float)
    for obs, action in preferred.items():
        rewards[int(action), int(obs)] = 1.0
    return rewards


def evaluate_observable_objective_original(
    policy: OpenLoopPolicy,
    future_distribution: Callable[[int, OpenLoopActions], Mapping[ObsSeq, float]],
    objective: ObservableObjective,
    initial_belief_id: int,
) -> float:
    total = 0.0
    for obs_seq, prob in future_distribution(initial_belief_id, policy.actions).items():
        total += float(prob) * float(objective(policy.actions, obs_seq))
    return total


def evaluate_additive_observable_objective_original(
    pomdp: FinitePOMDP,
    registry: BeliefRegistry,
    policy: OpenLoopPolicy,
    initial_belief_id: int,
    step_rewards: np.ndarray,
) -> float:
    @lru_cache(maxsize=None)
    def recurse(belief_id: int, depth: int) -> float:
        if depth == len(policy.actions):
            return 0.0
        action = int(policy.actions[depth])
        obs_probs = registry.observation_probabilities(belief_id, action)
        total = 0.0
        for obs in range(pomdp.num_observations):
            p_obs = float(obs_probs[obs])
            if p_obs <= 0.0:
                continue
            next_belief = registry.next_belief_id(belief_id, action, obs)
            total += p_obs * (float(step_rewards[action, obs]) + recurse(next_belief, depth + 1))
        return total

    return recurse(initial_belief_id, 0)


def latent_value_original(pomdp: FinitePOMDP, policy: OpenLoopPolicy) -> float:
    state_dist = np.asarray(pomdp.initial_belief, dtype=float)
    total = 0.0
    for action in policy.actions:
        total += float(np.dot(state_dist, pomdp.rewards[:, int(action)]))
        state_dist = state_dist @ pomdp.transition[:, int(action), :]
    return total


def _policy_prefix_row_index(policy: OpenLoopPolicy, depth: int, num_actions: int) -> int:
    idx = 0
    for action in policy.actions[:depth]:
        idx = idx * num_actions + int(action)
    return idx


def _policy_representative_history(
    partition: ExactHistoryPartition,
    depth_table: np.ndarray,
    class_id: int,
    prefix_row: int,
) -> History:
    members = partition.class_histories[class_id]
    histories = partition.histories_by_depth[partition.class_depth[class_id]]
    index = {history: idx for idx, history in enumerate(histories)}
    for history in members:
        if depth_table[prefix_row, index[history]] != IMPOSSIBLE_BELIEF_ID:
            return history
    return members[0]


def quotient_observation_distribution(
    pomdp: FinitePOMDP,
    registry: BeliefRegistry,
    tables: Sequence[np.ndarray],
    partition: ExactHistoryPartition,
    policy: OpenLoopPolicy,
) -> Dict[ObsSeq, float]:
    history_index = {
        depth: {history: idx for idx, history in enumerate(histories)}
        for depth, histories in partition.histories_by_depth.items()
    }

    @lru_cache(maxsize=None)
    def recurse(class_id: int) -> Dict[ObsSeq, float]:
        depth = partition.class_depth[class_id]
        if depth == partition.horizon:
            return {(): 1.0}
        prefix_row = _policy_prefix_row_index(policy, depth, pomdp.num_actions)
        rep = _policy_representative_history(partition, tables[depth], class_id, prefix_row)
        belief_id = int(tables[depth][prefix_row, history_index[depth][rep]])
        obs_probs = registry.observation_probabilities(belief_id, int(policy.actions[depth]))
        out: Dict[ObsSeq, float] = {}
        for obs in range(pomdp.num_observations):
            p_obs = float(obs_probs[obs])
            if p_obs <= 0.0:
                continue
            next_history = rep + (obs,)
            next_class = partition.history_to_class[next_history]
            suffix = recurse(next_class)
            for tail, p_tail in suffix.items():
                key = (obs,) + tail
                out[key] = out.get(key, 0.0) + p_obs * p_tail
        return out

    return recurse(partition.history_to_class[()])


def evaluate_observable_objective_quotient(
    pomdp: FinitePOMDP,
    registry: BeliefRegistry,
    tables: Sequence[np.ndarray],
    partition: ExactHistoryPartition,
    policy: OpenLoopPolicy,
    objective: ObservableObjective,
) -> float:
    total = 0.0
    for obs_seq, prob in quotient_observation_distribution(pomdp, registry, tables, partition, policy).items():
        total += float(prob) * float(objective(policy.actions, obs_seq))
    return total


def evaluate_additive_observable_objective_quotient(
    pomdp: FinitePOMDP,
    registry: BeliefRegistry,
    tables: Sequence[np.ndarray],
    partition: ExactHistoryPartition,
    policy: OpenLoopPolicy,
    step_rewards: np.ndarray,
) -> float:
    history_index = {
        depth: {history: idx for idx, history in enumerate(histories)}
        for depth, histories in partition.histories_by_depth.items()
    }

    @lru_cache(maxsize=None)
    def recurse(class_id: int) -> float:
        depth = partition.class_depth[class_id]
        if depth == partition.horizon:
            return 0.0
        prefix_row = _policy_prefix_row_index(policy, depth, pomdp.num_actions)
        rep = _policy_representative_history(partition, tables[depth], class_id, prefix_row)
        belief_id = int(tables[depth][prefix_row, history_index[depth][rep]])
        action = int(policy.actions[depth])
        obs_probs = registry.observation_probabilities(belief_id, action)
        total = 0.0
        for obs in range(pomdp.num_observations):
            p_obs = float(obs_probs[obs])
            if p_obs <= 0.0:
                continue
            next_class = partition.history_to_class[rep + (obs,)]
            total += p_obs * (float(step_rewards[action, obs]) + recurse(next_class))
        return total

    return recurse(partition.history_to_class[()])


def latent_value_quotient(
    pomdp: FinitePOMDP,
    registry: BeliefRegistry,
    tables: Sequence[np.ndarray],
    partition: ExactHistoryPartition,
    policy: OpenLoopPolicy,
) -> float:
    history_index = {
        depth: {history: idx for idx, history in enumerate(histories)}
        for depth, histories in partition.histories_by_depth.items()
    }

    @lru_cache(maxsize=None)
    def recurse(class_id: int) -> float:
        depth = partition.class_depth[class_id]
        if depth == partition.horizon:
            return 0.0
        prefix_row = _policy_prefix_row_index(policy, depth, pomdp.num_actions)
        members = partition.class_histories[class_id]
        belief_ids = [
            int(tables[depth][prefix_row, history_index[depth][history]])
            for history in members
            if int(tables[depth][prefix_row, history_index[depth][history]]) != IMPOSSIBLE_BELIEF_ID
        ]
        if not belief_ids:
            belief = np.asarray(pomdp.initial_belief, dtype=float)
        else:
            belief = np.mean(np.stack([registry.vector(bid) for bid in belief_ids if registry.vector(bid) is not None]), axis=0)
            belief = belief / float(belief.sum())
        action = int(policy.actions[depth])
        immediate = float(np.dot(belief, pomdp.rewards[:, action]))
        obs_probs = registry.observation_probabilities(belief_ids[0] if belief_ids else IMPOSSIBLE_BELIEF_ID, action)
        future = 0.0
        rep = _policy_representative_history(partition, tables[depth], class_id, prefix_row)
        for obs in range(pomdp.num_observations):
            p_obs = float(obs_probs[obs])
            if p_obs <= 0.0:
                continue
            next_class = partition.history_to_class[rep + (obs,)]
            future += p_obs * recurse(next_class)
        return immediate + future

    return recurse(partition.history_to_class[()])


def belief_suffix_wasserstein(
    pomdp: FinitePOMDP,
    registry: BeliefRegistry,
    belief_a: int,
    belief_b: int,
    action_seq: OpenLoopActions,
    d_obs: np.ndarray,
) -> float:
    @lru_cache(maxsize=None)
    def recurse(bid_a: int, bid_b: int, suffix: OpenLoopActions) -> float:
        if not suffix or bid_a == bid_b:
            return 0.0
        action = int(suffix[0])
        obs_probs_a = registry.observation_probabilities(bid_a, action)
        obs_probs_b = registry.observation_probabilities(bid_b, action)
        obs_idx_a = np.flatnonzero(obs_probs_a > 0.0)
        obs_idx_b = np.flatnonzero(obs_probs_b > 0.0)
        if obs_idx_a.size == 0 or obs_idx_b.size == 0:
            return 0.0
        tail = tuple(int(a) for a in suffix[1:])
        cost = np.zeros((len(obs_idx_a), len(obs_idx_b)), dtype=float)
        for i, obs_a in enumerate(obs_idx_a.tolist()):
            next_a = registry.next_belief_id(bid_a, action, int(obs_a))
            for j, obs_b in enumerate(obs_idx_b.tolist()):
                next_b = registry.next_belief_id(bid_b, action, int(obs_b))
                cost[i, j] = float(d_obs[int(obs_a), int(obs_b)]) + recurse(next_a, next_b, tail)
        pv = obs_probs_a[obs_idx_a].astype(float)
        qv = obs_probs_b[obs_idx_b].astype(float)
        pv = pv / float(pv.sum())
        qv = qv / float(qv.sum())
        return transport_lp_value(pv, qv, cost)

    return recurse(int(belief_a), int(belief_b), tuple(int(a) for a in action_seq))


def planning_summary_for_objective(
    pomdp: FinitePOMDP,
    registry: BeliefRegistry,
    tables: Sequence[np.ndarray],
    partition: ExactHistoryPartition,
    policies: Sequence[OpenLoopPolicy],
    evaluate_original: Callable[[OpenLoopPolicy], float],
    evaluate_quotient: Callable[[OpenLoopPolicy], float],
) -> Dict[str, object]:
    import time

    start = time.perf_counter()
    orig_values = {policy: float(evaluate_original(policy)) for policy in policies}
    time_original = time.perf_counter() - start

    start = time.perf_counter()
    quot_values = {policy: float(evaluate_quotient(policy)) for policy in policies}
    time_quotient = time.perf_counter() - start

    best_orig = max(orig_values, key=orig_values.get)
    best_quot = max(quot_values, key=quot_values.get)
    best_orig_value = float(orig_values[best_orig])
    quot_value_on_orig = float(orig_values[best_quot])

    return {
        "best_original_policy": best_orig,
        "best_quotient_policy": best_quot,
        "value_original_best": best_orig_value,
        "value_quotient_best_on_original": quot_value_on_orig,
        "regret": best_orig_value - quot_value_on_orig,
        "same_policy": bool(best_orig == best_quot),
        "time_original_s": time_original,
        "time_quotient_s": time_quotient,
        "total_histories": partition.total_histories,
        "num_classes": partition.num_classes_total,
    }


def family_distance_gap(
    pomdp: FinitePOMDP,
    registry: BeliefRegistry,
    tables: Sequence[np.ndarray],
    depth: int,
    clock_policy_suffixes: Sequence[OpenLoopActions],
    op_policy_suffixes: Sequence[OpenLoopActions],
    d_obs: np.ndarray,
) -> float:
    class_table = tables[depth]
    num_histories = class_table.shape[1]
    if depth == 0:
        clock_table = class_table[[0], :]
        op_table = class_table[[0], :]
    else:
        clock_table = np.unique(class_table, axis=0)
        op_rows = np.array(
            [_constant_prefix_row_index(pomdp.num_actions, depth, action) for action in range(pomdp.num_actions)],
            dtype=np.int32,
        )
        op_table = np.unique(class_table[op_rows, :], axis=0)

    @lru_cache(maxsize=None)
    def max_distance_for_pair(family: str, belief_a: int, belief_b: int, suffixes: Tuple[OpenLoopActions, ...]) -> float:
        if belief_a == belief_b:
            return 0.0
        best = 0.0
        for suffix in suffixes:
            best = max(best, belief_suffix_wasserstein(pomdp, registry, belief_a, belief_b, suffix, d_obs))
        return best

    max_gap = 0.0
    clock_suffix_tuple = tuple(clock_policy_suffixes)
    op_suffix_tuple = tuple(op_policy_suffixes)
    for i in range(num_histories):
        for j in range(i + 1, num_histories):
            clock_d = 0.0
            for row in clock_table:
                clock_d = max(
                    clock_d,
                    max_distance_for_pair(
                        "clk",
                        int(row[i]),
                        int(row[j]),
                        clock_suffix_tuple,
                    ),
                )
            op_d = 0.0
            for row in op_table:
                op_d = max(
                    op_d,
                    max_distance_for_pair(
                        "op",
                        int(row[i]),
                        int(row[j]),
                        op_suffix_tuple,
                    ),
                )
            max_gap = max(max_gap, abs(clock_d - op_d))
    return max_gap


def total_adjusted_rand_index(
    partition_a: ExactHistoryPartition,
    partition_b: ExactHistoryPartition,
) -> float:
    labels_a: List[int] = []
    labels_b: List[int] = []
    for depth in range(partition_a.horizon + 1):
        labels_a.extend(int(v) for v in partition_a.labels_by_depth[depth].tolist())
        labels_b.extend(int(v) for v in partition_b.labels_by_depth[depth].tolist())
    return float(adjusted_rand_score(labels_a, labels_b))
