"""Distance metrics for observation-sequence distributions."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from .pomdp_core import ObsSeq, observation_metric_sum


def total_variation_distance(
    p: Mapping[ObsSeq, float],
    q: Mapping[ObsSeq, float],
) -> float:
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * float(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


def _as_dense_vectors(
    p: Mapping[ObsSeq, float],
    q: Mapping[ObsSeq, float],
) -> Tuple[Tuple[ObsSeq, ...], np.ndarray, np.ndarray]:
    keys = tuple(sorted(set(p.keys()) | set(q.keys())))
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    # Small numerical cleanup.
    pv[pv < 0.0] = 0.0
    qv[qv < 0.0] = 0.0
    sp = pv.sum()
    sq = qv.sum()
    if sp > 0.0:
        pv = pv / sp
    if sq > 0.0:
        qv = qv / sq
    return keys, pv, qv


@lru_cache(maxsize=128)
def _transport_constraints(n_left: int, n_right: int) -> Tuple[csr_matrix, Tuple[Tuple[float, None], ...]]:
    num_vars = n_left * n_right
    rows = []
    cols = []
    data = []
    for i in range(n_left):
        base = i * n_right
        for j in range(n_right):
            rows.append(i)
            cols.append(base + j)
            data.append(1.0)
    for j in range(n_right):
        for i in range(n_left):
            rows.append(n_left + j)
            cols.append(i * n_right + j)
            data.append(1.0)
    a_eq = csr_matrix((data, (rows, cols)), shape=(n_left + n_right, num_vars), dtype=float)
    bounds = tuple((0.0, None) for _ in range(num_vars))
    return a_eq, bounds


def wasserstein_distance(
    p: Mapping[ObsSeq, float],
    q: Mapping[ObsSeq, float],
    d_obs: np.ndarray,
) -> float:
    """Compute W1 with additive sequence metric induced by d_obs."""
    keys_p = tuple(sorted(k for k, v in p.items() if v > 0.0))
    keys_q = tuple(sorted(k for k, v in q.items() if v > 0.0))

    if not keys_p and not keys_q:
        return 0.0

    pv = np.array([p[k] for k in keys_p], dtype=float)
    qv = np.array([q[k] for k in keys_q], dtype=float)
    pv = pv / pv.sum()
    qv = qv / qv.sum()

    n_left = len(keys_p)
    n_right = len(keys_q)
    if n_left == 1 and n_right == 1 and keys_p[0] == keys_q[0]:
        return 0.0

    cost = np.zeros((n_left, n_right), dtype=float)
    for i, seq_i in enumerate(keys_p):
        for j, seq_j in enumerate(keys_q):
            cost[i, j] = observation_metric_sum(seq_i, seq_j, d_obs)

    return transport_lp_value(pv, qv, cost)


def transport_lp_value(
    pv: np.ndarray,
    qv: np.ndarray,
    cost: np.ndarray,
) -> float:
    n_left, n_right = cost.shape
    if n_left == 0 or n_right == 0:
        return 0.0
    if n_left == 1 and n_right == 1:
        return float(cost[0, 0])

    c = cost.reshape(-1)
    a_eq, bounds = _transport_constraints(n_left, n_right)
    b_eq = np.concatenate([pv, qv])

    res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Wasserstein LP failed: {res.message}")
    return float(res.fun)


def distribution_distance(
    p: Mapping[ObsSeq, float],
    q: Mapping[ObsSeq, float],
    mode: str,
    d_obs: np.ndarray,
) -> float:
    mode = mode.lower().strip()
    if mode == "tv":
        return total_variation_distance(p, q)
    if mode == "w1":
        return wasserstein_distance(p, q, d_obs=d_obs)
    raise ValueError(f"Unsupported distance mode: {mode}")
