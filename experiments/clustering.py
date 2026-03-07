"""Shared complete-linkage clustering utility."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def cluster_complete_linkage(distance_matrix: np.ndarray, epsilon: float) -> List[List[int]]:
    """Greedy complete-linkage merging: all pairwise distances within a cluster <= epsilon."""
    n = distance_matrix.shape[0]
    clusters: List[List[int]] = [[i] for i in range(n)]

    merged = True
    tol = 1e-12
    while merged:
        merged = False
        for i in range(len(clusters)):
            if merged:
                break
            for j in range(i + 1, len(clusters)):
                c1 = clusters[i]
                c2 = clusters[j]
                ok = True
                for a in c1:
                    for b in c2:
                        if distance_matrix[a, b] > epsilon + tol:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    clusters[i] = sorted(c1 + c2)
                    del clusters[j]
                    merged = True
                    break

    for cluster in clusters:
        cluster.sort()
    clusters.sort(key=lambda c: c[0])
    return clusters


def cluster_optimal(distance_matrix: np.ndarray, epsilon: float) -> List[List[int]]:
    """Minimum-cardinality partition where all within-cluster distances <= epsilon.

    Uses Bron-Kerbosch to enumerate maximal cliques in the compatibility graph,
    then branch-and-bound to find minimum clique cover.  Falls back to greedy
    complete-linkage for n > 20.
    """
    n = distance_matrix.shape[0]
    if n > 20:
        return cluster_complete_linkage(distance_matrix, epsilon)

    tol = 1e-12

    # Build compatibility sets
    compat: List[set] = [set() for _ in range(n)]
    for i in range(n):
        compat[i].add(i)
        for j in range(i + 1, n):
            if distance_matrix[i, j] <= epsilon + tol:
                compat[i].add(j)
                compat[j].add(i)

    # Enumerate maximal cliques via Bron-Kerbosch
    all_cliques: List[frozenset] = []

    def _bron_kerbosch(r: set, p: set, x: set) -> None:
        if not p and not x:
            if r:
                all_cliques.append(frozenset(r))
            return
        pivot = max(p | x, key=lambda v: len(compat[v] & p))
        for v in list(p - compat[pivot]):
            _bron_kerbosch(r | {v}, p & compat[v], x & compat[v])
            p.remove(v)
            x.add(v)

    _bron_kerbosch(set(), set(range(n)), set())

    # Greedy upper bound for pruning
    greedy = cluster_complete_linkage(distance_matrix, epsilon)
    best: List[frozenset] = [frozenset(c) for c in greedy]

    def _cover(remaining: frozenset, chosen: List[frozenset]) -> None:
        nonlocal best
        if not remaining:
            if len(chosen) < len(best):
                best = list(chosen)
            return
        if len(chosen) >= len(best) - 1:
            return
        target = min(remaining)
        for clique in all_cliques:
            if target in clique:
                _cover(remaining - clique, chosen + [clique])

    _cover(frozenset(range(n)), [])

    clusters = [sorted(c) for c in best]
    clusters.sort(key=lambda c: c[0])
    return clusters


def clustering_optimality_gap(
    distance_matrix: np.ndarray,
    epsilon: float,
) -> Dict[str, object]:
    """Compare greedy vs optimal clustering."""
    greedy = cluster_complete_linkage(distance_matrix, epsilon)
    optimal = cluster_optimal(distance_matrix, epsilon)
    return {
        "greedy_num_clusters": len(greedy),
        "optimal_num_clusters": len(optimal),
        "gap": len(greedy) - len(optimal),
        "greedy_is_optimal": len(greedy) == len(optimal),
    }
