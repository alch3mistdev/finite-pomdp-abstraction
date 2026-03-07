"""Experiment orchestration for the finite-POMDP basic results package."""

from __future__ import annotations

import math
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmarks import (
    coarsen_observations,
    coarsened_observation_metric,
    delta_covering_coarsen,
    compute_reward_lipschitz_constant,
    discrete_observation_metric,
    gridworld_geometric_observation_metric,
    gridworld_pomdp,
    hallway_pomdp,
    lipschitz_observation_score,
    network_monitoring_pomdp,
    random_observation_metric,
    random_structured_pomdp,
    stochastic_channel_lipschitz_constant,
    stochastic_coarsen_observations,
    tiger_discrete_observation_metric,
    tiger_full_actions_pomdp,
    tiger_full_actions_pomdp_custom_belief,
    tiger_listen_only_pomdp,
)
from .fsc_enum import enumerate_deterministic_fscs, sample_stochastic_fscs
from .hierarchical import (
    run_hierarchical_t_scaling as _run_hierarchical_t_scaling,
    run_layered_bound_validation as _run_layered_bound_validation,
)
from .metrics import distribution_distance, wasserstein_distance
from .pomdp_core import (
    DeterministicFSC,
    FinitePOMDP,
    conditional_future_observation_distribution,
    expected_sequence_score,
    trajectory_observation_distribution,
)
from .quotient import (
    compute_class_count_curve,
    compute_partition_from_cache,
    d_m_t_between_original_and_quotient,
    precompute_distance_cache,
    quotient_observation_sequence_distribution,
    value_state_action_original,
    value_state_action_quotient,
)
from .spectral import (
    approximate_partition_from_subset,
    build_fsc_distance_tensor,
    build_sampling_based_fsc_distance_tensor,
    greedy_select_fscs,
    partition_agreement,
    spectral_analysis,
)


@dataclass(frozen=True)
class ExperimentConfig:
    profile: str
    eps_grid: Tuple[float, ...]
    ms: Tuple[int, ...]
    horizons: Tuple[int, ...]
    include_nonlipschitz: bool
    include_hierarchical_scaling: bool
    stochastic_samples: int
    seed: int
    max_horizon: int
    segment_horizon: int | None


@dataclass(frozen=True)
class ExperimentTask:
    name: str
    fn: Callable[..., pd.DataFrame]
    kwargs: Mapping[str, object]


def _build_config(
    profile: str,
    eps_grid: Sequence[float],
    ms: Sequence[int],
    horizons: Sequence[int],
    include_nonlipschitz: bool,
    include_hierarchical_scaling: bool,
    stochastic_samples: int | None,
    seed: int,
    max_horizon: int,
    segment_horizon: int | None,
) -> ExperimentConfig:
    profile = profile.lower().strip()
    if profile not in {"quick", "extended"}:
        raise ValueError("profile must be one of {'quick', 'extended'}")

    if eps_grid:
        eps = tuple(float(x) for x in eps_grid)
    else:
        if profile == "quick":
            eps = tuple(np.round(np.arange(0.0, 0.61, 0.1), 2))
        else:
            eps = tuple(np.round(np.arange(0.0, 0.61, 0.05), 2))

    ms_tuple = tuple(int(m) for m in (ms if ms else (1, 2)))
    default_horizons = (2, 3, 4) if profile == "quick" else (2, 3, 4, 5, 6)
    h_tuple = tuple(int(h) for h in (horizons if horizons else default_horizons))

    if stochastic_samples is None:
        stochastic_samples = 200 if profile == "quick" else 1000

    return ExperimentConfig(
        profile=profile,
        eps_grid=eps,
        ms=ms_tuple,
        horizons=h_tuple,
        include_nonlipschitz=include_nonlipschitz,
        include_hierarchical_scaling=include_hierarchical_scaling,
        stochastic_samples=int(stochastic_samples),
        seed=int(seed),
        max_horizon=int(max_horizon),
        segment_horizon=None if segment_horizon is None else int(segment_horizon),
    )


def _set_thread_caps_if_unset() -> None:
    """Avoid BLAS/OpenMP oversubscription when running many worker processes."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _resolve_parallel_workers(
    parallel: bool,
    workers: int | None,
    worker_fraction: float,
) -> int:
    if not parallel:
        return 1

    env_workers = os.getenv("EXPERIMENT_WORKERS")
    env_fraction = os.getenv("EXPERIMENT_WORKER_FRACTION")

    selected_workers = workers
    if selected_workers is None and env_workers is not None:
        try:
            selected_workers = int(env_workers)
        except ValueError:
            selected_workers = None

    if selected_workers is not None:
        return max(1, selected_workers)

    selected_fraction = float(worker_fraction)
    if env_fraction is not None:
        try:
            selected_fraction = float(env_fraction)
        except ValueError:
            selected_fraction = float(worker_fraction)
    if selected_fraction <= 0:
        selected_fraction = float(worker_fraction)

    cpu_count = os.cpu_count() or 1
    return min(cpu_count, max(1, math.floor(cpu_count * selected_fraction)))


def _run_experiment_task(task: ExperimentTask) -> tuple[str, pd.DataFrame, float]:
    _set_thread_caps_if_unset()
    t0 = time.perf_counter()
    result = task.fn(**dict(task.kwargs))
    elapsed = time.perf_counter() - t0
    return task.name, result, elapsed


def _execute_experiment_tasks(
    tasks: Sequence[ExperimentTask],
    workers: int,
) -> tuple[Dict[str, pd.DataFrame], List[Dict[str, object]]]:
    results: Dict[str, pd.DataFrame] = {}
    elapsed_by_name: Dict[str, float] = {}

    if workers <= 1:
        for task in tasks:
            name, result, elapsed = _run_experiment_task(task)
            results[name] = result
            elapsed_by_name[name] = elapsed
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_experiment_task, task): task.name
                for task in tasks
            }
            for future in as_completed(futures):
                name, result, elapsed = future.result()
                results[name] = result
                elapsed_by_name[name] = elapsed

    timings = [
        {"experiment": task.name, "wall_clock_s": round(elapsed_by_name[task.name], 4)}
        for task in tasks
    ]
    return results, timings


def tiger_reproduction_sanity() -> pd.DataFrame:
    """Reproduce headline Tiger worked-example numbers used in the manuscript."""
    pomdp = tiger_listen_only_pomdp()
    horizon = 2
    metric = tiger_discrete_observation_metric()

    policy = DeterministicFSC(
        num_nodes=1,
        action_for_node=(0,),
        next_node_for_observation=((0, 0),),
        initial_node=0,
    )

    p_l = conditional_future_observation_distribution(pomdp, policy, history=(0,), horizon=horizon)
    p_r = conditional_future_observation_distribution(pomdp, policy, history=(1,), horizon=horizon)
    w1_lr = wasserstein_distance(p_l, p_r, d_obs=metric)

    # Paper convention for the worked figure:
    # exact: all 1 + 2 + 4 history nodes shown separately (7 classes)
    # epsilon=0.5: root + {L,R} + terminal merge = 3 classes
    row = {
        "horizon": 2,
        "w1_L_vs_R": float(w1_lr),
        "paper_exact_class_count": 7,
        "paper_eps_0_5_class_count": 3,
    }
    return pd.DataFrame([row])


def run_capacity_sweep(
    pomdp: FinitePOMDP,
    d_obs: np.ndarray,
    benchmark_name: str,
    eps_grid: Sequence[float],
    ms: Sequence[int],
    horizon: int,
) -> pd.DataFrame:
    """Capacity sweep for any POMDP: class count vs epsilon and memory bound m."""
    rows: List[Dict[str, float]] = []
    total_histories = sum((pomdp.num_observations**d for d in range(horizon + 1)))

    for m in ms:
        policies = enumerate_deterministic_fscs(
            num_actions=pomdp.num_actions,
            num_observations=pomdp.num_observations,
            max_nodes=m,
            include_smaller=True,
        )
        cache = precompute_distance_cache(
            pomdp=pomdp,
            policies=policies,
            horizon=horizon,
            distance_mode="w1",
            d_obs=d_obs,
        )

        curve_rows = compute_class_count_curve(cache=cache, eps_grid=eps_grid)
        for item in curve_rows:
            rows.append(
                {
                    "benchmark": benchmark_name,
                    "horizon": horizon,
                    "m": float(m),
                    "epsilon": float(item["epsilon"]),
                    "class_count": float(item["class_count"]),
                    "compression_ratio": float(item["class_count"]) / float(total_histories),
                    "policy_count": float(len(policies)),
                }
            )

    return pd.DataFrame(rows)


def run_capacity_sweep_tiger(
    eps_grid: Sequence[float],
    ms: Sequence[int],
    horizon: int = 4,
) -> pd.DataFrame:
    return run_capacity_sweep(
        pomdp=tiger_full_actions_pomdp(),
        d_obs=tiger_discrete_observation_metric(),
        benchmark_name="tiger_full_actions",
        eps_grid=eps_grid,
        ms=ms,
        horizon=horizon,
    )


def run_capacity_sweep_gridworld(
    eps_grid: Sequence[float],
    ms: Sequence[int],
    horizon: int = 2,
) -> pd.DataFrame:
    return run_capacity_sweep(
        pomdp=gridworld_pomdp(size=3),
        d_obs=gridworld_geometric_observation_metric(),
        benchmark_name="gridworld_3x3",
        eps_grid=eps_grid,
        ms=ms,
        horizon=horizon,
    )


def _policies_m1_for_pomdp(pomdp) -> List[DeterministicFSC]:
    return enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=1,
        include_smaller=True,
    )


def run_lipschitz_value_bounds(
    eps_grid: Sequence[float],
) -> pd.DataFrame:
    """Value-loss and bound overlays on Tiger full-actions for m=1.

    Reports two tracks:
    - 'synthetic_lipschitz': L_R=1 synthetic reward (observation-score function)
    - 'standard_reward': L_R=110 standard Tiger reward, showing bound vacuousness
    """
    pomdp = tiger_full_actions_pomdp()
    horizon = 2
    d_obs = tiger_discrete_observation_metric()

    policies = _policies_m1_for_pomdp(pomdp)
    cache = precompute_distance_cache(
        pomdp=pomdp,
        policies=policies,
        horizon=horizon,
        distance_mode="w1",
        d_obs=d_obs,
    )

    rows: List[Dict[str, object]] = []

    # --- Track 1: Synthetic L_R=1 ---
    l_r_synthetic = 1.0
    sequence_score = lambda seq: 1.0 if all(o == 0 for o in seq) else 0.0

    for eps in eps_grid:
        partition = compute_partition_from_cache(cache=cache, epsilon=float(eps))

        per_policy = []
        for p in policies:
            p_m = trajectory_observation_distribution(pomdp=pomdp, policy=p, horizon=horizon)
            p_q = quotient_observation_sequence_distribution(pomdp=pomdp, partition=partition, policy=p)
            v_m = expected_sequence_score(p_m, score_fn=sequence_score)
            v_q = expected_sequence_score(p_q, score_fn=sequence_score)
            d = distribution_distance(p_m, p_q, mode="w1", d_obs=d_obs)
            per_policy.append((abs(v_m - v_q), d))

        empirical_value_error = max(v for v, _ in per_policy)
        d_m_t = max(d for _, d in per_policy)
        theorem_bound = l_r_synthetic * horizon * d_m_t
        canonical_bound = l_r_synthetic * horizon * float(eps) * (1.0 + 2.0 * horizon * pomdp.num_states * pomdp.num_observations)

        rows.append({
            "benchmark": "tiger_full_actions",
            "reward_track": "synthetic_lipschitz",
            "horizon": horizon,
            "m": 1.0,
            "epsilon": float(eps),
            "class_count": float(partition.num_classes_total),
            "empirical_value_error": float(empirical_value_error),
            "d_m_t_w1": float(d_m_t),
            "theorem_4_4_style_bound": float(theorem_bound),
            "canonical_quotient_bound": float(canonical_bound),
            "L_R": float(l_r_synthetic),
            "reward_range": 1.0,
            "theorem_bound_vacuous": bool(theorem_bound > 1.0),
            "canonical_bound_vacuous": bool(canonical_bound > 1.0),
            "bound_applicable": True,
        })

    # --- Track 2: Standard Tiger reward L_R=110 ---
    l_r_standard = compute_reward_lipschitz_constant(pomdp)
    reward_range_std = float(pomdp.rewards.max() - pomdp.rewards.min())

    for eps in eps_grid:
        partition = compute_partition_from_cache(cache=cache, epsilon=float(eps))

        per_policy_std = []
        for p in policies:
            v_m = value_state_action_original(pomdp=pomdp, policy=p, horizon=horizon)
            v_q = value_state_action_quotient(pomdp=pomdp, partition=partition, policy=p)
            p_m = trajectory_observation_distribution(pomdp=pomdp, policy=p, horizon=horizon)
            p_q = quotient_observation_sequence_distribution(pomdp=pomdp, partition=partition, policy=p)
            d = distribution_distance(p_m, p_q, mode="w1", d_obs=d_obs)
            per_policy_std.append((abs(v_m - v_q), d))

        empirical_value_error = max(v for v, _ in per_policy_std)
        d_m_t = max(d for _, d in per_policy_std)
        theorem_bound = l_r_standard * horizon * d_m_t
        canonical_bound = l_r_standard * horizon * float(eps) * (1.0 + 2.0 * horizon * pomdp.num_states * pomdp.num_observations)

        rows.append({
            "benchmark": "tiger_full_actions",
            "reward_track": "standard_reward",
            "horizon": horizon,
            "m": 1.0,
            "epsilon": float(eps),
            "class_count": float(partition.num_classes_total),
            "empirical_value_error": float(empirical_value_error),
            "d_m_t_w1": float(d_m_t),
            "theorem_4_4_style_bound": float(theorem_bound),
            "canonical_quotient_bound": float(canonical_bound),
            "L_R": float(l_r_standard),
            "reward_range": float(reward_range_std),
            "theorem_bound_vacuous": bool(theorem_bound > reward_range_std),
            "canonical_bound_vacuous": bool(canonical_bound > reward_range_std),
            "bound_applicable": True,
        })

    return pd.DataFrame(rows)


def run_nonlipschitz_value_track(
    eps_grid: Sequence[float],
) -> pd.DataFrame:
    """Empirical value-loss under standard Tiger rewards.

    Shows actual empirical value error and the vacuous bound that would result
    from applying L_R=110 to the Lipschitz bound formula.
    """
    pomdp = tiger_full_actions_pomdp()
    horizon = 2
    d_obs = tiger_discrete_observation_metric()
    l_r_standard = compute_reward_lipschitz_constant(pomdp)
    reward_range = float(pomdp.rewards.max() - pomdp.rewards.min())

    policies = _policies_m1_for_pomdp(pomdp)
    cache = precompute_distance_cache(
        pomdp=pomdp,
        policies=policies,
        horizon=horizon,
        distance_mode="w1",
        d_obs=d_obs,
    )

    rows: List[Dict[str, object]] = []
    for eps in eps_grid:
        partition = compute_partition_from_cache(cache=cache, epsilon=float(eps))

        errs = []
        dmts = []
        for p in policies:
            v_m = value_state_action_original(pomdp=pomdp, policy=p, horizon=horizon)
            v_q = value_state_action_quotient(pomdp=pomdp, partition=partition, policy=p)
            errs.append(abs(v_m - v_q))
            p_m = trajectory_observation_distribution(pomdp=pomdp, policy=p, horizon=horizon)
            p_q = quotient_observation_sequence_distribution(pomdp=pomdp, partition=partition, policy=p)
            dmts.append(distribution_distance(p_m, p_q, mode="w1", d_obs=d_obs))

        d_m_t = max(dmts) if dmts else 0.0
        hypothetical_bound = l_r_standard * horizon * d_m_t

        rows.append({
            "benchmark": "tiger_full_actions",
            "horizon": horizon,
            "m": 1.0,
            "epsilon": float(eps),
            "class_count": float(partition.num_classes_total),
            "empirical_value_error": float(max(errs)),
            "d_m_t_w1": float(d_m_t),
            "L_R_standard": float(l_r_standard),
            "reward_range": float(reward_range),
            "hypothetical_bound_l_r_110": float(hypothetical_bound),
            "bound_vacuous": bool(hypothetical_bound > reward_range),
            "bound_applicable": False,
            "bound_note": "Standard Tiger reward: L_R=110 makes bound vacuous (exceeds reward range).",
        })

    return pd.DataFrame(rows)


def run_horizon_gap_tiger(
    horizons: Sequence[int],
    epsilon: float = 0.5,
) -> pd.DataFrame:
    """Horizon stress on listen-only Tiger to illustrate gap growth trends."""
    rows: List[Dict[str, float]] = []

    for horizon in horizons:
        pomdp = tiger_listen_only_pomdp()
        d_obs = tiger_discrete_observation_metric()
        sequence_score = lambda seq: 1.0 if all(o == 0 for o in seq) else 0.0

        policies = _policies_m1_for_pomdp(pomdp)
        cache = precompute_distance_cache(
            pomdp=pomdp,
            policies=policies,
            horizon=horizon,
            distance_mode="w1",
            d_obs=d_obs,
        )
        part = compute_partition_from_cache(cache=cache, epsilon=float(epsilon))

        per_policy = []
        for p in policies:
            p_m = trajectory_observation_distribution(pomdp=pomdp, policy=p, horizon=horizon)
            p_q = quotient_observation_sequence_distribution(pomdp=pomdp, partition=part, policy=p)
            v_m = expected_sequence_score(p_m, score_fn=sequence_score)
            v_q = expected_sequence_score(p_q, score_fn=sequence_score)
            d = distribution_distance(p_m, p_q, mode="w1", d_obs=d_obs)
            per_policy.append((abs(v_m - v_q), d))

        empirical_error = max(v for v, _ in per_policy)
        d_m_t = max(d for _, d in per_policy)
        theorem_bound = float(horizon) * d_m_t
        canonical_bound = float(horizon) * float(epsilon) * (1.0 + 2.0 * float(horizon) * pomdp.num_states * pomdp.num_observations)

        rows.append(
            {
                "benchmark": "tiger_listen_only",
                "horizon": float(horizon),
                "epsilon": float(epsilon),
                "class_count": float(part.num_classes_total),
                "empirical_value_error": float(empirical_error),
                "d_m_t_w1": float(d_m_t),
                "theorem_4_4_style_bound": float(theorem_bound),
                "canonical_quotient_bound": float(canonical_bound),
                "canonical_to_empirical_ratio": float(canonical_bound / max(empirical_error, 1e-12)),
            }
        )

    return pd.DataFrame(rows)


def run_gridworld_metric_sensitivity(
    eps_grid: Sequence[float],
) -> pd.DataFrame:
    pomdp = gridworld_pomdp(size=3)
    horizon = 2

    policies = _policies_m1_for_pomdp(pomdp)
    cache_w1_geo = precompute_distance_cache(
        pomdp=pomdp,
        policies=policies,
        horizon=horizon,
        distance_mode="w1",
        d_obs=gridworld_geometric_observation_metric(),
    )
    cache_tv = precompute_distance_cache(
        pomdp=pomdp,
        policies=policies,
        horizon=horizon,
        distance_mode="tv",
        d_obs=discrete_observation_metric(pomdp.num_observations),
    )

    rows: List[Dict[str, float]] = []
    for eps in eps_grid:
        p_w1 = compute_partition_from_cache(cache_w1_geo, epsilon=float(eps))
        p_tv = compute_partition_from_cache(cache_tv, epsilon=float(eps))
        rows.append(
            {
                "benchmark": "gridworld_3x3",
                "horizon": float(horizon),
                "m": 1.0,
                "epsilon": float(eps),
                "class_count_w1_geometric": float(p_w1.num_classes_total),
                "class_count_tv_equivalent": float(p_tv.num_classes_total),
                "delta_tv_minus_w1": float(p_tv.num_classes_total - p_w1.num_classes_total),
            }
        )

    return pd.DataFrame(rows)


def run_stochastic_vs_deterministic_sanity(
    stochastic_samples: int,
    seed: int,
) -> pd.DataFrame:
    pomdp = tiger_full_actions_pomdp()
    horizon = 2
    d_obs = tiger_discrete_observation_metric()

    det_policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=2,
        include_smaller=True,
    )

    def hist_distance(policy) -> float:
        p_l = conditional_future_observation_distribution(pomdp, policy, history=(0,), horizon=horizon)
        p_r = conditional_future_observation_distribution(pomdp, policy, history=(1,), horizon=horizon)
        return distribution_distance(p_l, p_r, mode="w1", d_obs=d_obs)

    max_det = 0.0
    for p in det_policies:
        max_det = max(max_det, hist_distance(p))

    stoch_policies = sample_stochastic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        num_nodes=2,
        num_samples=stochastic_samples,
        seed=seed,
    )

    stoch_vals = [hist_distance(p) for p in stoch_policies]
    max_stoch = max(stoch_vals) if stoch_vals else 0.0

    row = {
        "benchmark": "tiger_full_actions",
        "horizon": float(horizon),
        "m": 2.0,
        "history_pair": "L_vs_R_at_t1",
        "deterministic_policy_count": float(len(det_policies)),
        "stochastic_samples": float(stochastic_samples),
        "max_distance_deterministic": float(max_det),
        "max_distance_stochastic_sample": float(max_stoch),
        "det_minus_stoch_gap": float(max_det - max_stoch),
        "stochastic_exceeds_deterministic": bool(max_stoch > max_det + 1e-9),
    }
    return pd.DataFrame([row])


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure as PNG (raster) and PDF (vector)."""
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))


def _save_plot_capacity(df_capacity: pd.DataFrame, path: Path) -> None:
    target_eps = min(df_capacity["epsilon"].unique(), key=lambda x: abs(x - 0.5))
    df = df_capacity[df_capacity["epsilon"] == target_eps].copy()
    df = df.sort_values("m")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["m"].astype(int).astype(str), df["class_count"], color=["#4E79A7", "#F28E2B"][: len(df)])
    ax.set_title(f"Tiger Full-Actions: Class Count vs Memory (epsilon={target_eps:.2f})")
    ax.set_xlabel("Memory bound m")
    ax.set_ylabel("Quotient class count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _save_plot_value_bounds(df_value: pd.DataFrame, path: Path) -> None:
    df = df_value.sort_values("epsilon")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["epsilon"], df["empirical_value_error"], marker="o", label="Empirical value error")
    ax.plot(df["epsilon"], df["theorem_4_4_style_bound"], marker="s", label="L_R * T * D_{m,T}^W")
    ax.plot(df["epsilon"], df["canonical_quotient_bound"], marker="^", label="Canonical quotient bound")
    ax.set_title("Value Error vs Two Bounds (Lipschitz Track)")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Value / bound")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _save_plot_horizon_gap(df_h: pd.DataFrame, path: Path) -> None:
    df = df_h.sort_values("horizon")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["horizon"], df["empirical_value_error"], marker="o", label="Empirical value error")
    ax.plot(df["horizon"], df["theorem_4_4_style_bound"], marker="s", label="L_R * T * D_{m,T}^W")
    ax.plot(df["horizon"], df["canonical_quotient_bound"], marker="^", label="Canonical quotient bound")
    ax.set_title("Tiger Listen-Only: Gap Growth with Horizon")
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Value / bound")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def run_spectral_rank_analysis(
    ms: Sequence[int],
    horizon: int = 4,
    include_gridworld: bool = True,
) -> pd.DataFrame:
    """Spectral rank analysis of the FSC distance tensor."""
    benchmarks = [
        ("tiger_full_actions", tiger_full_actions_pomdp(), tiger_discrete_observation_metric(), horizon),
    ]
    if include_gridworld:
        benchmarks.append(
            ("gridworld_3x3", gridworld_pomdp(size=3), gridworld_geometric_observation_metric(), min(horizon, 2)),
        )

    rows: List[Dict[str, float]] = []
    for bench_name, pomdp, d_obs, h in benchmarks:
        for m in ms:
            policies = enumerate_deterministic_fscs(
                num_actions=pomdp.num_actions,
                num_observations=pomdp.num_observations,
                max_nodes=m,
                include_smaller=True,
            )
            cache = build_fsc_distance_tensor(
                pomdp=pomdp,
                policies=policies,
                horizon=h,
                distance_mode="w1",
                d_obs=d_obs,
            )
            sa = spectral_analysis(cache)

            for depth, info in sa.items():
                svs = info["singular_values"]
                if len(svs) == 0:
                    continue
                rows.append(
                    {
                        "benchmark": bench_name,
                        "horizon": float(h),
                        "m": float(m),
                        "depth": float(depth),
                        "total_fscs": float(len(policies)),
                        "num_singular_values": float(len(svs)),
                        "top_singular_value": float(svs[0]) if len(svs) > 0 else 0.0,
                        "effective_rank_90": float(info["effective_rank_90"]),
                        "effective_rank_95": float(info["effective_rank_95"]),
                        "effective_rank_99": float(info["effective_rank_99"]),
                    }
                )

    return pd.DataFrame(rows)


def run_spectral_partition_comparison(
    eps_grid: Sequence[float],
    ms: Sequence[int],
    horizon: int = 4,
    ks: Sequence[int] = (1, 3, 5, 10),
    include_gridworld: bool = True,
) -> pd.DataFrame:
    """Compare exact vs approximate partition using k principal FSCs."""
    benchmarks = [
        ("tiger_full_actions", tiger_full_actions_pomdp(), tiger_discrete_observation_metric(), horizon),
    ]
    if include_gridworld:
        benchmarks.append(
            ("gridworld_3x3", gridworld_pomdp(size=3), gridworld_geometric_observation_metric(), min(horizon, 2)),
        )

    rows: List[Dict[str, float]] = []
    for bench_name, pomdp, d_obs, h in benchmarks:
        for m in ms:
            policies = enumerate_deterministic_fscs(
                num_actions=pomdp.num_actions,
                num_observations=pomdp.num_observations,
                max_nodes=m,
                include_smaller=True,
            )
            cache = build_fsc_distance_tensor(
                pomdp=pomdp,
                policies=policies,
                horizon=h,
                distance_mode="w1",
                d_obs=d_obs,
            )

            for eps in eps_grid:
                exact_part = approximate_partition_from_subset(
                    cache=cache,
                    fsc_indices=list(range(len(policies))),
                    epsilon=float(eps),
                )

                for k in ks:
                    if k > len(policies):
                        continue
                    selected = greedy_select_fscs(cache, k=k)
                    approx_part = approximate_partition_from_subset(
                        cache=cache,
                        fsc_indices=selected,
                        epsilon=float(eps),
                    )
                    agreement = partition_agreement(exact_part, approx_part)

                    rows.append(
                        {
                            "benchmark": bench_name,
                            "horizon": float(h),
                            "m": float(m),
                            "epsilon": float(eps),
                            "k": float(k),
                            "total_fscs": float(len(policies)),
                            "exact_classes": float(agreement["exact_classes"]),
                            "approx_classes": float(agreement["approx_classes"]),
                            "adjusted_rand_index": float(agreement["adjusted_rand_index"]),
                            "merge_fidelity": float(agreement["merge_fidelity"]),
                        }
                    )

    return pd.DataFrame(rows)


def _save_plot_spectral_decay(df_rank: pd.DataFrame, path: Path) -> None:
    """Effective rank bar chart grouped by benchmark and m."""
    # Build groups: one per (benchmark, m) at the deepest non-trivial depth
    groups = []
    for bench in sorted(df_rank["benchmark"].unique()):
        for m in sorted(df_rank["m"].unique()):
            sub = df_rank[(df_rank["benchmark"] == bench) & (df_rank["m"] == m) & (df_rank["depth"] > 0)]
            if sub.empty:
                continue
            max_depth = sub["depth"].max()
            row = sub[sub["depth"] == max_depth].iloc[0]
            groups.append({
                "label": f"{bench} m={int(m)} ({int(row['total_fscs'])} FSCs)",
                "ranks": [row["effective_rank_90"], row["effective_rank_95"], row["effective_rank_99"]],
            })

    if not groups:
        return

    thresholds = ["90%", "95%", "99%"]
    x = np.arange(len(thresholds))
    n = len(groups)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, g in enumerate(groups):
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, g["ranks"], width, alpha=0.8, label=g["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.set_title("Effective Rank of FSC Distance Tensor")
    ax.set_xlabel("Explained variance threshold")
    ax.set_ylabel("Effective rank (# FSCs needed)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _save_plot_partition_agreement(df_compare: pd.DataFrame, path: Path) -> None:
    """Partition agreement vs k for fixed epsilon."""
    target_eps = min(df_compare["epsilon"].unique(), key=lambda x: abs(x - 0.3))
    df = df_compare[df_compare["epsilon"] == target_eps].copy()

    fig, ax = plt.subplots(figsize=(6, 4))
    for m in sorted(df["m"].unique()):
        sub = df[df["m"] == m].sort_values("k")
        ax.plot(sub["k"], sub["adjusted_rand_index"], marker="o", label=f"m={int(m)}")
    ax.set_title(f"Partition Agreement vs Principal FSC Count (eps={target_eps:.2f})")
    ax.set_xlabel("k (number of selected FSCs)")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _save_plot_metric_sensitivity(df_m: pd.DataFrame, path: Path) -> None:
    df = df_m.sort_values("epsilon")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["epsilon"], df["class_count_w1_geometric"], marker="o", label="W1 (geometric d_O)")
    ax.plot(df["epsilon"], df["class_count_tv_equivalent"], marker="s", label="TV-equivalent")
    ax.set_title("GridWorld: W1 vs TV Class Count")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Quotient class count")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def run_observation_noise_sensitivity(
    eps_grid: Sequence[float],
    accuracies: Sequence[float] = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
) -> pd.DataFrame:
    """Vary Tiger observation accuracy and report class counts and value errors."""
    horizon = 2
    l_r = 1.0
    sequence_score = lambda seq: 1.0 if all(o == 0 for o in seq) else 0.0

    rows: List[Dict[str, float]] = []
    for acc in accuracies:
        pomdp = tiger_full_actions_pomdp(accuracy=acc)
        d_obs = tiger_discrete_observation_metric()
        policies = _policies_m1_for_pomdp(pomdp)
        cache = precompute_distance_cache(
            pomdp=pomdp, policies=policies, horizon=horizon,
            distance_mode="w1", d_obs=d_obs,
        )

        for eps in eps_grid:
            partition = compute_partition_from_cache(cache=cache, epsilon=float(eps))
            per_policy = []
            for p in policies:
                p_m = trajectory_observation_distribution(pomdp=pomdp, policy=p, horizon=horizon)
                p_q = quotient_observation_sequence_distribution(pomdp=pomdp, partition=partition, policy=p)
                v_m = expected_sequence_score(p_m, score_fn=sequence_score)
                v_q = expected_sequence_score(p_q, score_fn=sequence_score)
                per_policy.append(abs(v_m - v_q))
            rows.append({
                "accuracy": float(acc),
                "epsilon": float(eps),
                "class_count": float(partition.num_classes_total),
                "max_value_error": float(max(per_policy)),
            })

    return pd.DataFrame(rows)


def run_multi_seed_witness(
    seeds: Sequence[int] = (7, 42, 123, 256, 999),
    stochastic_samples: int = 200,
) -> pd.DataFrame:
    """Run stochastic-vs-deterministic witness check across multiple seeds."""
    pomdp = tiger_full_actions_pomdp()
    horizon = 2
    d_obs = tiger_discrete_observation_metric()

    det_policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=2, include_smaller=True,
    )

    def hist_distance(policy) -> float:
        p_l = conditional_future_observation_distribution(pomdp, policy, history=(0,), horizon=horizon)
        p_r = conditional_future_observation_distribution(pomdp, policy, history=(1,), horizon=horizon)
        return distribution_distance(p_l, p_r, mode="w1", d_obs=d_obs)

    max_det = max(hist_distance(p) for p in det_policies)

    rows: List[Dict[str, float]] = []
    for seed in seeds:
        stoch_policies = sample_stochastic_fscs(
            num_actions=pomdp.num_actions,
            num_observations=pomdp.num_observations,
            num_nodes=2, num_samples=stochastic_samples, seed=seed,
        )
        stoch_vals = [hist_distance(p) for p in stoch_policies]
        max_stoch = max(stoch_vals) if stoch_vals else 0.0
        rows.append({
            "seed": float(seed),
            "max_distance_deterministic": float(max_det),
            "max_distance_stochastic": float(max_stoch),
            "gap": float(max_det - max_stoch),
            "stochastic_exceeds_det": float(max_stoch > max_det + 1e-9),
        })

    return pd.DataFrame(rows)


def _save_plot_noise_sensitivity(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for acc in sorted(df["accuracy"].unique()):
        sub = df[df["accuracy"] == acc].sort_values("epsilon")
        axes[0].plot(sub["epsilon"], sub["class_count"], marker="o", label=f"acc={acc:.2f}")
        axes[1].plot(sub["epsilon"], sub["max_value_error"], marker="o", label=f"acc={acc:.2f}")

    axes[0].set_xlabel("epsilon")
    axes[0].set_ylabel("Class count")
    axes[0].set_title("Sensitivity: Class Count vs Observation Accuracy")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("epsilon")
    axes[1].set_ylabel("Max value error")
    axes[1].set_title("Sensitivity: Value Error vs Observation Accuracy")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def run_baseline_comparison(
    eps_grid: Sequence[float],
    include_gridworld: bool = False,
) -> pd.DataFrame:
    """Compare epsilon-quotient against baseline partition methods."""
    from .baselines import (
        belief_distance_partition,
        bisimulation_metric_partition,
        random_partition,
        truncation_partition,
    )

    benchmarks: List[Tuple[str, FinitePOMDP, np.ndarray, int]] = [
        ("tiger_full_actions", tiger_full_actions_pomdp(), tiger_discrete_observation_metric(), 2),
    ]
    if include_gridworld:
        benchmarks.append(
            ("gridworld_3x3", gridworld_pomdp(size=3), gridworld_geometric_observation_metric(), 2),
        )

    all_rows: List[Dict[str, object]] = []

    for bench_name, pomdp, d_obs, horizon in benchmarks:
        sequence_score = lambda seq: 1.0 if all(o == 0 for o in seq) else 0.0

        policies = _policies_m1_for_pomdp(pomdp)
        cache = precompute_distance_cache(
            pomdp=pomdp, policies=policies, horizon=horizon,
            distance_mode="w1", d_obs=d_obs,
        )

        for eps in eps_grid:
            # Epsilon-quotient
            t0 = time.perf_counter()
            eq_part = compute_partition_from_cache(cache=cache, epsilon=float(eps))
            eq_err = _max_value_error(pomdp, eq_part, policies, horizon, sequence_score)
            eq_dmt = _max_d_m_t(pomdp, eq_part, policies, horizon, d_obs)
            eq_time = time.perf_counter() - t0

            all_rows.append({
                "benchmark": bench_name,
                "method": "epsilon_quotient",
                "epsilon": float(eps),
                "class_count": float(eq_part.num_classes_total),
                "max_value_error": float(eq_err),
                "d_m_t_w1": float(eq_dmt),
                "time_s": round(eq_time, 4),
            })

            # Truncation baseline (depth=1)
            t0 = time.perf_counter()
            trunc_part = truncation_partition(
                num_observations=pomdp.num_observations, horizon=horizon, depth=1,
            )
            trunc_err = _max_value_error(pomdp, trunc_part, policies, horizon, sequence_score)
            trunc_dmt = _max_d_m_t(pomdp, trunc_part, policies, horizon, d_obs)
            trunc_time = time.perf_counter() - t0
            all_rows.append({
                "benchmark": bench_name,
                "method": "truncation_d1",
                "epsilon": float(eps),
                "class_count": float(trunc_part.num_classes_total),
                "max_value_error": float(trunc_err),
                "d_m_t_w1": float(trunc_dmt),
                "time_s": round(trunc_time, 4),
            })

            # Random partition (k = epsilon-quotient class count per depth)
            k_per_depth = {d: len(ids) for d, ids in eq_part.classes_by_depth.items()}
            rand_errs = []
            rand_counts = []
            rand_dmts = []
            t0 = time.perf_counter()
            for seed in (7, 42, 123, 256, 999):
                rand_part = random_partition(
                    num_observations=pomdp.num_observations, horizon=horizon,
                    k_per_depth=k_per_depth, seed=seed,
                )
                rand_errs.append(_max_value_error(pomdp, rand_part, policies, horizon, sequence_score))
                rand_counts.append(rand_part.num_classes_total)
                rand_dmts.append(_max_d_m_t(pomdp, rand_part, policies, horizon, d_obs))
            rand_time = (time.perf_counter() - t0) / 5.0
            all_rows.append({
                "benchmark": bench_name,
                "method": "random_partition",
                "epsilon": float(eps),
                "class_count": float(np.mean(rand_counts)),
                "class_count_std": float(np.std(rand_counts)),
                "max_value_error": float(np.mean(rand_errs)),
                "max_value_error_std": float(np.std(rand_errs)),
                "d_m_t_w1": float(np.mean(rand_dmts)),
                "d_m_t_w1_std": float(np.std(rand_dmts)),
                "time_s": round(rand_time, 4),
            })

            # Belief-distance partition
            t0 = time.perf_counter()
            bd_part = belief_distance_partition(
                pomdp=pomdp, policy=policies[0], horizon=horizon, epsilon=float(eps),
            )
            bd_err = _max_value_error(pomdp, bd_part, policies, horizon, sequence_score)
            bd_dmt = _max_d_m_t(pomdp, bd_part, policies, horizon, d_obs)
            bd_time = time.perf_counter() - t0
            all_rows.append({
                "benchmark": bench_name,
                "method": "belief_distance",
                "epsilon": float(eps),
                "class_count": float(bd_part.num_classes_total),
                "max_value_error": float(bd_err),
                "d_m_t_w1": float(bd_dmt),
                "time_s": round(bd_time, 4),
            })

            # Approximate bisimulation baseline
            t0 = time.perf_counter()
            bisim_part = bisimulation_metric_partition(
                pomdp=pomdp, horizon=horizon, epsilon=float(eps),
            )
            bisim_err = _max_value_error(pomdp, bisim_part, policies, horizon, sequence_score)
            bisim_dmt = _max_d_m_t(pomdp, bisim_part, policies, horizon, d_obs)
            bisim_time = time.perf_counter() - t0
            all_rows.append({
                "benchmark": bench_name,
                "method": "bisimulation",
                "epsilon": float(eps),
                "class_count": float(bisim_part.num_classes_total),
                "max_value_error": float(bisim_err),
                "d_m_t_w1": float(bisim_dmt),
                "time_s": round(bisim_time, 4),
            })

    df = pd.DataFrame(all_rows)

    # Flag baselines that match epsilon-quotient exactly
    flag_rows = []
    for (bench, eps), group in df.groupby(["benchmark", "epsilon"]):
        eq_row = group[group["method"] == "epsilon_quotient"]
        if eq_row.empty:
            continue
        eq_cc = float(eq_row["class_count"].iloc[0])
        eq_err = float(eq_row["max_value_error"].iloc[0])
        for _, row in group.iterrows():
            matches = (
                row["method"] != "epsilon_quotient"
                and abs(float(row["class_count"]) - eq_cc) < 0.5
                and abs(float(row["max_value_error"]) - eq_err) < 1e-9
            )
            flag_rows.append({
                "benchmark": row["benchmark"],
                "method": row["method"],
                "epsilon": row["epsilon"],
                "matches_epsilon_quotient": bool(matches),
            })
    if flag_rows:
        flag_df = pd.DataFrame(flag_rows)
        df = df.merge(flag_df, on=["benchmark", "method", "epsilon"], how="left")
        df["matches_epsilon_quotient"] = df["matches_epsilon_quotient"].fillna(False)
    else:
        df["matches_epsilon_quotient"] = False

    return df


def _max_value_error(pomdp, partition, policies, horizon, score_fn) -> float:
    errs = []
    for p in policies:
        p_m = trajectory_observation_distribution(pomdp=pomdp, policy=p, horizon=horizon)
        p_q = quotient_observation_sequence_distribution(pomdp=pomdp, partition=partition, policy=p)
        v_m = expected_sequence_score(p_m, score_fn=score_fn)
        v_q = expected_sequence_score(p_q, score_fn=score_fn)
        errs.append(abs(v_m - v_q))
    return max(errs) if errs else 0.0


def _max_d_m_t(pomdp, partition, policies, horizon, d_obs) -> float:
    dmax = 0.0
    for p in policies:
        p_m = trajectory_observation_distribution(pomdp=pomdp, policy=p, horizon=horizon)
        p_q = quotient_observation_sequence_distribution(pomdp=pomdp, partition=partition, policy=p)
        d = distribution_distance(p_m, p_q, mode="w1", d_obs=d_obs)
        dmax = max(dmax, d)
    return dmax


def _save_plot_baseline_comparison(df: pd.DataFrame, path: Path) -> None:
    markers = {"epsilon_quotient": "o", "truncation_d1": "s", "random_partition": "^", "belief_distance": "D", "bisimulation": "P"}
    colors = {"epsilon_quotient": "#4E79A7", "truncation_d1": "#F28E2B", "random_partition": "#E15759", "belief_distance": "#76B7B2", "bisimulation": "#59A14F"}

    benchmarks = df["benchmark"].unique() if "benchmark" in df.columns else ["tiger_full_actions"]
    n_bench = len(benchmarks)
    fig, axes = plt.subplots(1, n_bench, figsize=(7 * n_bench, 5), squeeze=False)

    for idx, bench in enumerate(benchmarks):
        ax = axes[0, idx]
        sub_bench = df[df["benchmark"] == bench] if "benchmark" in df.columns else df
        for method in sub_bench["method"].unique():
            sub = sub_bench[sub_bench["method"] == method].sort_values("class_count")
            ax.scatter(
                sub["class_count"], sub["max_value_error"],
                marker=markers.get(method, "o"), color=colors.get(method, "gray"),
                label=method.replace("_", " "), s=60, zorder=3,
            )
        ax.set_xlabel("Class count")
        ax.set_ylabel("Max value error")
        ax.set_title(f"Baseline Comparison: {bench}")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def run_ablation_studies(
    eps_grid: Sequence[float],
) -> pd.DataFrame:
    """Ablation studies: W1 vs TV on Tiger, m=1 vs m=2, spectral vs full."""
    rows: List[Dict[str, float]] = []

    # 1. W1 vs TV on Tiger full-actions
    pomdp = tiger_full_actions_pomdp()
    horizon = 2
    d_obs_w1 = tiger_discrete_observation_metric()
    d_obs_tv = discrete_observation_metric(pomdp.num_observations)
    policies_m1 = _policies_m1_for_pomdp(pomdp)

    cache_w1 = precompute_distance_cache(
        pomdp=pomdp, policies=policies_m1, horizon=horizon,
        distance_mode="w1", d_obs=d_obs_w1,
    )
    cache_tv = precompute_distance_cache(
        pomdp=pomdp, policies=policies_m1, horizon=horizon,
        distance_mode="tv", d_obs=d_obs_tv,
    )

    for eps in eps_grid:
        p_w1 = compute_partition_from_cache(cache_w1, epsilon=float(eps))
        p_tv = compute_partition_from_cache(cache_tv, epsilon=float(eps))
        rows.append({
            "ablation": "w1_vs_tv_tiger",
            "epsilon": float(eps),
            "metric": "w1",
            "class_count": float(p_w1.num_classes_total),
            "m": 1.0,
        })
        rows.append({
            "ablation": "w1_vs_tv_tiger",
            "epsilon": float(eps),
            "metric": "tv",
            "class_count": float(p_tv.num_classes_total),
            "m": 1.0,
        })

    # 2. m=1 vs m=2 value error at fixed epsilon=0.3
    eps_fixed = 0.3
    sequence_score = lambda seq: 1.0 if all(o == 0 for o in seq) else 0.0
    for m_val in (1, 2):
        pols = enumerate_deterministic_fscs(
            num_actions=pomdp.num_actions,
            num_observations=pomdp.num_observations,
            max_nodes=m_val, include_smaller=True,
        )
        cache_m = precompute_distance_cache(
            pomdp=pomdp, policies=pols, horizon=horizon,
            distance_mode="w1", d_obs=d_obs_w1,
        )
        part = compute_partition_from_cache(cache_m, epsilon=eps_fixed)

        # Value error using m=1 policies for fair comparison
        err = _max_value_error(pomdp, part, policies_m1, horizon, sequence_score)
        rows.append({
            "ablation": "m1_vs_m2",
            "epsilon": float(eps_fixed),
            "metric": "w1",
            "class_count": float(part.num_classes_total),
            "m": float(m_val),
            "max_value_error": float(err),
        })

    # 3. Spectral (k=5) vs full enumeration
    policies_m2 = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=2, include_smaller=True,
    )

    t0 = time.perf_counter()
    cache_full = precompute_distance_cache(
        pomdp=pomdp, policies=policies_m2, horizon=horizon,
        distance_mode="w1", d_obs=d_obs_w1,
    )
    time_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    scache = build_fsc_distance_tensor(
        pomdp=pomdp, policies=policies_m2, horizon=horizon,
        distance_mode="w1", d_obs=d_obs_w1,
    )
    selected = greedy_select_fscs(scache, k=min(5, len(policies_m2)))
    time_spectral = time.perf_counter() - t0

    for eps in eps_grid:
        part_full = compute_partition_from_cache(cache_full, epsilon=float(eps))
        part_approx = approximate_partition_from_subset(scache, fsc_indices=selected, epsilon=float(eps))
        agreement = partition_agreement(part_full, part_approx)
        rows.append({
            "ablation": "spectral_vs_full",
            "epsilon": float(eps),
            "metric": "w1",
            "class_count_full": float(part_full.num_classes_total),
            "class_count_spectral": float(part_approx.num_classes_total),
            "ari": float(agreement["adjusted_rand_index"]),
            "m": 2.0,
            "time_full_s": float(time_full),
            "time_spectral_s": float(time_spectral),
        })

    return pd.DataFrame(rows)


def run_larger_scale_experiments(
    eps_grid: Sequence[float],
    seed: int = 42,
) -> pd.DataFrame:
    """Run quotient computation on larger POMDPs (|S| >= 20)."""
    rows: List[Dict[str, float]] = []

    # --- 5x5 GridWorld (|S|=25) ---
    for horizon in (2, 3):
        pomdp = gridworld_pomdp(size=5)
        d_obs = gridworld_geometric_observation_metric()
        policies = _policies_m1_for_pomdp(pomdp)
        cache = precompute_distance_cache(
            pomdp=pomdp, policies=policies, horizon=horizon,
            distance_mode="w1", d_obs=d_obs,
        )
        for eps in eps_grid:
            part = compute_partition_from_cache(cache, epsilon=float(eps))
            total_h = sum(
                len(cache.histories_by_depth[d])
                for d in range(cache.horizon + 1)
            )
            rows.append({
                "benchmark": "gridworld_5x5",
                "num_states": 25.0,
                "horizon": float(horizon),
                "m": 1.0,
                "epsilon": float(eps),
                "class_count": float(part.num_classes_total),
                "total_histories": float(total_h),
                "compression_ratio": float(part.num_classes_total) / float(total_h),
            })

    # --- Random structured POMDP (|S|=20), averaged over 3 seeds ---
    for rseed in (seed, seed + 1, seed + 2):
        for horizon in (2, 3):
            pomdp = random_structured_pomdp(num_states=20, seed=rseed)
            d_obs = random_observation_metric(pomdp.num_observations, seed=rseed)
            policies = _policies_m1_for_pomdp(pomdp)
            cache = precompute_distance_cache(
                pomdp=pomdp, policies=policies, horizon=horizon,
                distance_mode="w1", d_obs=d_obs,
            )
            for eps in eps_grid:
                part = compute_partition_from_cache(cache, epsilon=float(eps))
                total_h = sum(
                    len(cache.histories_by_depth[d])
                    for d in range(cache.horizon + 1)
                )
                rows.append({
                    "benchmark": "random_pomdp_20",
                    "num_states": 20.0,
                    "horizon": float(horizon),
                    "m": 1.0,
                    "epsilon": float(eps),
                    "class_count": float(part.num_classes_total),
                    "total_histories": float(total_h),
                    "compression_ratio": float(part.num_classes_total) / float(total_h),
                    "instance_seed": float(rseed),
                })

    # --- Scaling table: GridWorld sizes {3,4,5,6} at T=2 ---
    for gsize in (3, 4, 5, 6):
        pomdp = gridworld_pomdp(size=gsize)
        d_obs = gridworld_geometric_observation_metric()
        policies = _policies_m1_for_pomdp(pomdp)

        t0 = time.perf_counter()
        cache = precompute_distance_cache(
            pomdp=pomdp, policies=policies, horizon=2,
            distance_mode="w1", d_obs=d_obs,
        )
        part = compute_partition_from_cache(cache, epsilon=0.3)
        elapsed = time.perf_counter() - t0

        total_h = sum(
            len(cache.histories_by_depth[d])
            for d in range(cache.horizon + 1)
        )
        rows.append({
            "benchmark": f"gridworld_{gsize}x{gsize}",
            "num_states": float(gsize * gsize),
            "horizon": 2.0,
            "m": 1.0,
            "epsilon": 0.3,
            "class_count": float(part.num_classes_total),
            "total_histories": float(total_h),
            "compression_ratio": float(part.num_classes_total) / float(total_h),
            "wall_clock_s": float(elapsed),
        })

    return pd.DataFrame(rows)


def _save_plot_larger_scale(df: pd.DataFrame, path: Path) -> None:
    """Class count vs epsilon for larger POMDPs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: GridWorld 5x5
    gw = df[df["benchmark"] == "gridworld_5x5"].copy()
    for h in sorted(gw["horizon"].unique()):
        sub = gw[gw["horizon"] == h].sort_values("epsilon")
        axes[0].plot(sub["epsilon"], sub["class_count"], marker="o", label=f"T={int(h)}")
    axes[0].set_title("GridWorld 5x5 (|S|=25, m=1)")
    axes[0].set_xlabel("epsilon")
    axes[0].set_ylabel("Quotient class count")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    # Right: Random POMDP (averaged)
    rp = df[df["benchmark"] == "random_pomdp_20"].copy()
    for h in sorted(rp["horizon"].unique()):
        sub = rp[rp["horizon"] == h].groupby("epsilon").agg(
            mean_cc=("class_count", "mean"),
            std_cc=("class_count", "std"),
        ).reset_index().sort_values("epsilon")
        axes[1].plot(sub["epsilon"], sub["mean_cc"], marker="o", label=f"T={int(h)}")
        axes[1].fill_between(
            sub["epsilon"],
            sub["mean_cc"] - sub["std_cc"],
            sub["mean_cc"] + sub["std_cc"],
            alpha=0.2,
        )
    axes[1].set_title("Random POMDP (|S|=20, m=1, 3 seeds)")
    axes[1].set_xlabel("epsilon")
    axes[1].set_ylabel("Quotient class count")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _save_plot_scaling(df: pd.DataFrame, path: Path) -> None:
    """Wall-clock time and class count vs |S|."""
    scaling = df[df["wall_clock_s"].notna()].sort_values("num_states")
    if scaling.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(scaling["num_states"], scaling["wall_clock_s"], marker="o", color="#4E79A7")
    axes[0].set_xlabel("|S|")
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].set_title("Scaling: Computation Time (T=2, m=1, eps=0.3)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(scaling["num_states"], scaling["class_count"], marker="s", color="#E15759")
    axes[1].set_xlabel("|S|")
    axes[1].set_ylabel("Class count")
    axes[1].set_title("Scaling: Quotient Size (T=2, m=1, eps=0.3)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def run_extended_capacity_experiment(
    eps_grid: Sequence[float],
    seed: int = 42,
) -> pd.DataFrame:
    """Extended m=2 experiments: Tiger T=4, GridWorld 3x3 T=2, GridWorld 5x5 T=2 (sampling)."""
    from .sampling import sampling_based_distance_cache

    rows: List[Dict[str, float]] = []

    # --- Tiger m=2, T=4 (exact, 147 FSCs) ---
    tiger = tiger_full_actions_pomdp()
    d_obs_tiger = tiger_discrete_observation_metric()
    policies_tiger = enumerate_deterministic_fscs(
        num_actions=tiger.num_actions,
        num_observations=tiger.num_observations,
        max_nodes=2, include_smaller=True,
    )
    t0 = time.perf_counter()
    cache_tiger = precompute_distance_cache(
        pomdp=tiger, policies=policies_tiger, horizon=4,
        distance_mode="w1", d_obs=d_obs_tiger,
    )
    time_tiger = time.perf_counter() - t0
    total_h_tiger = sum(tiger.num_observations ** d for d in range(5))
    for eps in eps_grid:
        part = compute_partition_from_cache(cache_tiger, epsilon=float(eps))
        rows.append({
            "benchmark": "tiger_full_actions",
            "horizon": 4.0, "m": 2.0,
            "epsilon": float(eps),
            "class_count": float(part.num_classes_total),
            "compression_ratio": float(part.num_classes_total) / float(total_h_tiger),
            "policy_count": float(len(policies_tiger)),
            "method": "exact",
            "runtime_s": round(time_tiger, 3),
        })

    # --- GridWorld 3x3 m=2, T=2 (exact, 6405 FSCs) ---
    gw3 = gridworld_pomdp(size=3)
    d_obs_gw = gridworld_geometric_observation_metric()
    policies_gw3 = enumerate_deterministic_fscs(
        num_actions=gw3.num_actions, num_observations=gw3.num_observations,
        max_nodes=2, include_smaller=True,
    )
    t0 = time.perf_counter()
    cache_gw3 = precompute_distance_cache(
        pomdp=gw3, policies=policies_gw3, horizon=2,
        distance_mode="w1", d_obs=d_obs_gw,
    )
    time_gw3 = time.perf_counter() - t0
    total_h_gw = sum(gw3.num_observations ** d for d in range(3))
    for eps in eps_grid:
        part = compute_partition_from_cache(cache_gw3, epsilon=float(eps))
        rows.append({
            "benchmark": "gridworld_3x3",
            "horizon": 2.0, "m": 2.0,
            "epsilon": float(eps),
            "class_count": float(part.num_classes_total),
            "compression_ratio": float(part.num_classes_total) / float(total_h_gw),
            "policy_count": float(len(policies_gw3)),
            "method": "exact",
            "runtime_s": round(time_gw3, 3),
        })

    # --- GridWorld 5x5 m=2, T=2 (sampling, 6405 FSCs) ---
    gw5 = gridworld_pomdp(size=5)
    policies_gw5 = enumerate_deterministic_fscs(
        num_actions=gw5.num_actions, num_observations=gw5.num_observations,
        max_nodes=2, include_smaller=True,
    )
    t0 = time.perf_counter()
    cache_gw5 = sampling_based_distance_cache(
        pomdp=gw5, policies=policies_gw5, horizon=2,
        d_obs=d_obs_gw, num_samples=500, seed=seed,
    )
    time_gw5 = time.perf_counter() - t0
    total_h_gw5 = sum(gw5.num_observations ** d for d in range(3))
    for eps in eps_grid:
        part = compute_partition_from_cache(cache_gw5, epsilon=float(eps))
        rows.append({
            "benchmark": "gridworld_5x5",
            "horizon": 2.0, "m": 2.0,
            "epsilon": float(eps),
            "class_count": float(part.num_classes_total),
            "compression_ratio": float(part.num_classes_total) / float(total_h_gw5),
            "policy_count": float(len(policies_gw5)),
            "method": "sampling_500",
            "runtime_s": round(time_gw5, 3),
        })

    return pd.DataFrame(rows)


def run_m2_medium_scale_experiment(
    eps_grid: Sequence[float],
    seed: int = 42,
    num_samples: int = 500,
) -> pd.DataFrame:
    """m=2 experiments at medium scale (|S|=64-100) using sampling-based W1.

    For each benchmark, computes partitions for both m=1 and m=2 and records
    class counts side-by-side. This demonstrates that agent memory capacity
    creates genuine additional partition structure at scale.
    """
    from .sampling import sampling_based_distance_cache

    rows: List[Dict[str, float]] = []

    benchmarks = [
        ("gridworld_8x8", gridworld_pomdp(size=8), gridworld_geometric_observation_metric()),
        ("gridworld_10x10", gridworld_pomdp(size=10), gridworld_geometric_observation_metric()),
    ]
    random_seeds = [seed, seed + 1]
    for rseed in random_seeds:
        benchmarks.append((
            f"random_pomdp_50_s{rseed}",
            random_structured_pomdp(num_states=50, num_actions=3, num_observations=4, seed=rseed),
            random_observation_metric(4, seed=rseed),
        ))
        benchmarks.append((
            f"random_pomdp_100_s{rseed}",
            random_structured_pomdp(num_states=100, num_actions=3, num_observations=4, seed=rseed),
            random_observation_metric(4, seed=rseed),
        ))

    for bench_name, pomdp, d_obs in benchmarks:
        for m in (1, 2):
            policies = enumerate_deterministic_fscs(
                num_actions=pomdp.num_actions,
                num_observations=pomdp.num_observations,
                max_nodes=m, include_smaller=True,
            )
            t0 = time.perf_counter()
            cache = sampling_based_distance_cache(
                pomdp=pomdp, policies=policies, horizon=2,
                d_obs=d_obs, num_samples=num_samples, seed=seed,
            )
            elapsed = time.perf_counter() - t0

            total_h = sum(len(cache.histories_by_depth[d]) for d in range(cache.horizon + 1))
            for eps in eps_grid:
                part = compute_partition_from_cache(cache, epsilon=float(eps))
                rows.append({
                    "benchmark": bench_name,
                    "num_states": float(pomdp.num_states),
                    "horizon": 2.0,
                    "m": float(m),
                    "epsilon": float(eps),
                    "class_count": float(part.num_classes_total),
                    "total_histories": float(total_h),
                    "compression_ratio": float(part.num_classes_total) / float(total_h),
                    "policy_count": float(len(policies)),
                    "method": "sampling",
                    "num_samples": float(num_samples),
                    "runtime_s": round(elapsed, 3),
                })

    return pd.DataFrame(rows)


def run_spectral_rank_at_scale(
    seed: int = 42,
    num_samples: int = 500,
) -> pd.DataFrame:
    """Spectral low-rank analysis on larger benchmarks (GridWorld 5x5 m=2, 8x8 m=2).

    Uses sampling-based distance tensor to test whether the spectral low-rank
    structure observed on Tiger/GW3x3 persists at larger scale.
    """
    rows: List[Dict[str, float]] = []

    benchmarks = [
        ("gridworld_5x5", gridworld_pomdp(size=5), gridworld_geometric_observation_metric(), 2),
        ("gridworld_8x8", gridworld_pomdp(size=8), gridworld_geometric_observation_metric(), 2),
    ]

    for bench_name, pomdp, d_obs, horizon in benchmarks:
        for m in (1, 2):
            policies = enumerate_deterministic_fscs(
                num_actions=pomdp.num_actions,
                num_observations=pomdp.num_observations,
                max_nodes=m, include_smaller=True,
            )

            t0 = time.perf_counter()
            cache = build_sampling_based_fsc_distance_tensor(
                pomdp=pomdp, policies=policies, horizon=horizon,
                d_obs=d_obs, num_samples=num_samples, seed=seed,
            )
            tensor_time = time.perf_counter() - t0

            sa = spectral_analysis(cache)
            for depth, info in sa.items():
                svs = info["singular_values"]
                for rank_idx, sv in enumerate(svs):
                    rows.append({
                        "benchmark": bench_name,
                        "m": float(m),
                        "horizon": float(horizon),
                        "depth": float(depth),
                        "rank_idx": float(rank_idx),
                        "singular_value": float(sv),
                        "explained_variance_ratio": float(info["explained_variance_ratio"][rank_idx]),
                        "cumulative_variance_ratio": float(info["cumulative_variance_ratio"][rank_idx]),
                        "effective_rank_90": float(info["effective_rank_90"]),
                        "effective_rank_95": float(info["effective_rank_95"]),
                        "effective_rank_99": float(info["effective_rank_99"]),
                        "n_fscs": float(len(policies)),
                        "tensor_time_s": round(tensor_time, 3),
                    })

            # Also test greedy FSC selection + partition recovery
            for k in (3, 5, 10):
                if k > len(policies):
                    continue
                selected = greedy_select_fscs(cache, k=k)
                for eps in (0.0, 0.1, 0.3, 0.5):
                    exact_part = approximate_partition_from_subset(
                        cache, fsc_indices=list(range(len(policies))), epsilon=eps,
                    )
                    approx_part = approximate_partition_from_subset(
                        cache, fsc_indices=selected, epsilon=eps,
                    )
                    agreement = partition_agreement(exact_part, approx_part)
                    rows.append({
                        "benchmark": bench_name,
                        "m": float(m),
                        "horizon": float(horizon),
                        "depth": -1.0,  # sentinel for partition comparison rows
                        "rank_idx": -1.0,
                        "singular_value": float("nan"),
                        "explained_variance_ratio": float("nan"),
                        "cumulative_variance_ratio": float("nan"),
                        "effective_rank_90": float("nan"),
                        "effective_rank_95": float("nan"),
                        "effective_rank_99": float("nan"),
                        "n_fscs": float(len(policies)),
                        "tensor_time_s": round(tensor_time, 3),
                        "k": float(k),
                        "epsilon": float(eps),
                        "ari": float(agreement["adjusted_rand_index"]),
                        "merge_fidelity": float(agreement["merge_fidelity"]),
                        "exact_classes": float(agreement["exact_classes"]),
                        "approx_classes": float(agreement["approx_classes"]),
                    })

    return pd.DataFrame(rows)


def run_sampling_variance_analysis(
    eps_grid: Sequence[float],
    n_replications: int = 10,
    num_samples: int = 500,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Assess sampling variance on medium-scale benchmarks.

    Runs multiple independent replications with different seeds to report
    mean +/- std of partition class counts.
    """
    from .sampling import sampling_based_distance_cache

    rows: List[Dict[str, float]] = []

    benchmarks = [
        ("gridworld_10x10", gridworld_pomdp(size=10), gridworld_geometric_observation_metric()),
    ]

    for bench_name, pomdp, d_obs in benchmarks:
        for m in (1, 2):
            policies = enumerate_deterministic_fscs(
                num_actions=pomdp.num_actions,
                num_observations=pomdp.num_observations,
                max_nodes=m, include_smaller=True,
            )

            for rep in range(n_replications):
                rep_seed = base_seed + rep * 1000
                cache = sampling_based_distance_cache(
                    pomdp=pomdp, policies=policies, horizon=2,
                    d_obs=d_obs, num_samples=num_samples, seed=rep_seed,
                )
                for eps in eps_grid:
                    part = compute_partition_from_cache(cache, epsilon=float(eps))
                    rows.append({
                        "benchmark": bench_name,
                        "m": float(m),
                        "epsilon": float(eps),
                        "replication": float(rep),
                        "seed": float(rep_seed),
                        "class_count": float(part.num_classes_total),
                        "num_samples": float(num_samples),
                        "policy_count": float(len(policies)),
                    })

    return pd.DataFrame(rows)


def run_hierarchical_t_scaling(
    max_horizon: int = 10,
    segment_horizon: int | None = None,
    epsilon: float = 0.5,
    m: int = 1,
) -> pd.DataFrame:
    """Public wrapper for hierarchical long-horizon scaling experiments."""
    canonical = (4, 6, 8, 10)
    horizons = tuple(h for h in canonical if h <= int(max_horizon))
    if not horizons:
        horizons = (int(max_horizon),)
    return _run_hierarchical_t_scaling(
        horizons=horizons,
        epsilon=epsilon,
        m=m,
        segment_horizon=segment_horizon,
    )


def run_layered_bound_validation(
    eps_grid: Sequence[float],
    max_horizon: int = 10,
    epsilon: float = 0.5,
) -> pd.DataFrame:
    """Public wrapper for compositional-bound validation experiments."""
    long_horizons = tuple(h for h in (8, 10) if h <= int(max_horizon))
    if not long_horizons:
        long_horizons = (int(max_horizon),)
    return _run_layered_bound_validation(
        eps_grid=eps_grid,
        long_horizons=long_horizons,
        epsilon=epsilon,
    )


def run_principal_fsc_horizon_scaling(
    epsilon: float = 0.5,
    calibration_horizon: int = 6,
    horizons_full: Sequence[int] = (4, 5, 6, 7),
    horizons_subset: Sequence[int] = (7, 8),
    ari_target: float = 0.999,
    default_k: int = 10,
    max_k_search: int = 20,
) -> pd.DataFrame:
    """Long-horizon scaling with principal FSC subsets for m=2."""
    pomdp = tiger_full_actions_pomdp()
    d_obs = tiger_discrete_observation_metric()
    policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=2,
        include_smaller=True,
    )

    rows: List[Dict[str, float]] = []

    # Calibrate smallest k reaching ARI target on calibration horizon.
    t0 = time.perf_counter()
    cal_tensor = build_fsc_distance_tensor(
        pomdp=pomdp,
        policies=policies,
        horizon=int(calibration_horizon),
        distance_mode="w1",
        d_obs=d_obs,
    )
    tensor_time = time.perf_counter() - t0
    exact_cal = approximate_partition_from_subset(
        cal_tensor,
        fsc_indices=list(range(len(policies))),
        epsilon=float(epsilon),
    )

    selected_indices: List[int] | None = None
    selected_k = int(default_k)
    selected_ari = 0.0
    found_k: int | None = None
    for k in range(1, min(int(max_k_search), len(policies)) + 1):
        idxs = greedy_select_fscs(cal_tensor, k=k)
        approx = approximate_partition_from_subset(
            cal_tensor,
            fsc_indices=idxs,
            epsilon=float(epsilon),
        )
        agr = partition_agreement(exact_cal, approx)
        ari = float(agr["adjusted_rand_index"])
        if ari >= float(ari_target):
            found_k = int(k)
            break

    if found_k is None:
        found_k = int(min(default_k, len(policies)))
    selected_k = int(min(len(policies), max(int(default_k), int(found_k))))
    selected_indices = greedy_select_fscs(cal_tensor, k=selected_k)
    approx = approximate_partition_from_subset(
        cal_tensor,
        fsc_indices=selected_indices,
        epsilon=float(epsilon),
    )
    selected_ari = float(partition_agreement(exact_cal, approx)["adjusted_rand_index"])

    rows.append(
        {
            "benchmark": "tiger_full_actions",
            "method": "calibration",
            "horizon": float(calibration_horizon),
            "epsilon": float(epsilon),
            "m": 2.0,
            "policy_count": float(len(policies)),
            "k_selected": float(selected_k),
            "ari_vs_full": float(selected_ari),
            "class_count": float(exact_cal.num_classes_total),
            "runtime_s": float(tensor_time),
            "speedup_vs_full": float("nan"),
            "ari_target": float(ari_target),
            "calibration_horizon": float(calibration_horizon),
        }
    )

    full_runtime: Dict[int, float] = {}
    full_partitions: Dict[int, object] = {}

    for horizon in sorted(set(int(h) for h in horizons_full)):
        t1 = time.perf_counter()
        cache = precompute_distance_cache(
            pomdp=pomdp,
            policies=policies,
            horizon=horizon,
            distance_mode="w1",
            d_obs=d_obs,
        )
        elapsed = time.perf_counter() - t1
        part = compute_partition_from_cache(cache, epsilon=float(epsilon))
        full_runtime[horizon] = float(elapsed)
        full_partitions[horizon] = part
        rows.append(
            {
                "benchmark": "tiger_full_actions",
                "method": "full_exact",
                "horizon": float(horizon),
                "epsilon": float(epsilon),
                "m": 2.0,
                "policy_count": float(len(policies)),
                "k_selected": float(len(policies)),
                "ari_vs_full": 1.0,
                "class_count": float(part.num_classes_total),
                "runtime_s": float(elapsed),
                "speedup_vs_full": 1.0,
                "ari_target": float(ari_target),
                "calibration_horizon": float(calibration_horizon),
            }
        )

    selected_policies = [policies[i] for i in selected_indices]
    for horizon in sorted(set(int(h) for h in horizons_subset)):
        t2 = time.perf_counter()
        sub_cache = precompute_distance_cache(
            pomdp=pomdp,
            policies=selected_policies,
            horizon=horizon,
            distance_mode="w1",
            d_obs=d_obs,
        )
        elapsed = time.perf_counter() - t2
        sub_part = compute_partition_from_cache(sub_cache, epsilon=float(epsilon))

        ari_vs_full = float("nan")
        speedup = float("nan")
        if horizon in full_partitions:
            agr = partition_agreement(full_partitions[horizon], sub_part)
            ari_vs_full = float(agr["adjusted_rand_index"])
            speedup = float(full_runtime[horizon] / max(elapsed, 1e-12))

        rows.append(
            {
                "benchmark": "tiger_full_actions",
                "method": "principal_subset",
                "horizon": float(horizon),
                "epsilon": float(epsilon),
                "m": 2.0,
                "policy_count": float(len(selected_policies)),
                "k_selected": float(selected_k),
                "ari_vs_full": float(ari_vs_full),
                "class_count": float(sub_part.num_classes_total),
                "runtime_s": float(elapsed),
                "speedup_vs_full": float(speedup),
                "ari_target": float(ari_target),
                "calibration_horizon": float(calibration_horizon),
            }
        )

    return pd.DataFrame(rows)


def _save_plot_m1_vs_m2_comparison(df: pd.DataFrame, path: Path) -> None:
    """Grouped bar chart: class count at eps=0.0 and eps=0.3 for m=1 vs m=2."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, eps in enumerate([0.0, 0.3]):
        ax = axes[ax_idx]
        subset = df[np.isclose(df["epsilon"], eps)]

        benchmarks = sorted(subset["benchmark"].unique())
        x = np.arange(len(benchmarks))
        width = 0.35

        m1_vals = []
        m2_vals = []
        for b in benchmarks:
            bdata = subset[subset["benchmark"] == b]
            m1_row = bdata[np.isclose(bdata["m"], 1.0)]
            m2_row = bdata[np.isclose(bdata["m"], 2.0)]
            m1_vals.append(float(m1_row["class_count"].mean()) if len(m1_row) > 0 else 0)
            m2_vals.append(float(m2_row["class_count"].mean()) if len(m2_row) > 0 else 0)

        ax.bar(x - width / 2, m1_vals, width, label="m=1", color="#4c72b0")
        ax.bar(x + width / 2, m2_vals, width, label="m=2", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace("_", "\n") for b in benchmarks], fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("Class Count")
        ax.set_title(f"m=1 vs m=2 Partition Size (ε={eps})")
        ax.legend()

    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _save_plot_runtime_vs_horizon_log(df: pd.DataFrame, path: Path) -> None:
    """Log-scale runtime curve for direct vs layered vs principal methods."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = ["direct", "layered", "full_exact", "principal_subset"]
    for method in methods:
        sub = df[df["method"] == method].sort_values("horizon")
        if sub.empty:
            continue
        ax.plot(
            sub["horizon"],
            sub["runtime_s"],
            marker="o",
            label=method.replace("_", " "),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.set_title("Runtime vs Horizon: Direct, Layered, Principal-FSC")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _save_plot_layered_bound_validation(df: pd.DataFrame, path: Path) -> None:
    """Visualize bound checks as lhs/rhs bars for all validation checks."""
    if df.empty:
        return
    num_rows = len(df)
    fig_h = max(3.0, 0.35 * num_rows)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(num_rows)
    lhs = df["value_lhs"].astype(float).to_numpy()
    rhs = df["value_rhs"].astype(float).to_numpy()
    labels = [
        f"{c} (T={int(h)})"
        for c, h in zip(df["check"].astype(str).tolist(), df["horizon"].astype(float).tolist())
    ]
    ax.barh(y - 0.2, lhs, height=0.35, label="LHS")
    ax.barh(y + 0.2, rhs, height=0.35, label="RHS bound")
    for idx, ok in enumerate(df["bound_holds"].astype(bool).tolist()):
        mark = "OK" if ok else "FAIL"
        ax.text(max(lhs[idx], rhs[idx]) * 1.01 + 1e-9, y[idx], mark, va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Value")
    ax.set_title("Layered Bound Validation Checks")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def run_convergence_analysis(
    eps_grid: Sequence[float],
    seed: int = 42,
) -> pd.DataFrame:
    """Run convergence sweep on GridWorld 3x3 (m=1, T=2) for convergence.csv."""
    from .sampling import run_convergence_sweep

    pomdp = gridworld_pomdp(size=3)
    d_obs = gridworld_geometric_observation_metric()
    policies = _policies_m1_for_pomdp(pomdp)

    exact_cache = precompute_distance_cache(
        pomdp=pomdp, policies=policies, horizon=2,
        distance_mode="w1", d_obs=d_obs,
    )

    rows = run_convergence_sweep(
        pomdp=pomdp, policies=policies, horizon=2,
        d_obs=d_obs,
        num_samples_grid=(50, 100, 250, 500, 1000),
        eps_grid=eps_grid,
        exact_cache=exact_cache,
        seed=seed,
    )

    for r in rows:
        r["benchmark"] = "gridworld_3x3"

    return pd.DataFrame(rows)


def run_rate_distortion_evaluation(
    eps_grid: Sequence[float],
) -> pd.DataFrame:
    """Evaluate rate-distortion bounds for multiple benchmarks (m=1, T=2)."""
    benchmarks = [
        ("Tiger", tiger_full_actions_pomdp(), tiger_discrete_observation_metric()),
        ("GridWorld 3x3", gridworld_pomdp(size=3), gridworld_geometric_observation_metric()),
    ]
    horizon = 2

    rows: List[Dict[str, object]] = []
    for bench_name, pomdp, d_obs in benchmarks:
        policies = _policies_m1_for_pomdp(pomdp)
        cache = precompute_distance_cache(
            pomdp=pomdp, policies=policies, horizon=horizon,
            distance_mode="w1", d_obs=d_obs,
        )

        for eps in eps_grid:
            part = compute_partition_from_cache(cache, epsilon=float(eps))
            n_classes = part.num_classes_total
            achievable_rate = float(np.log2(n_classes)) if n_classes > 0 else 0.0
            rows.append({
                "benchmark": bench_name,
                "epsilon": float(eps),
                "class_count": float(n_classes),
                "achievable_rate_bits": achievable_rate,
            })

    return pd.DataFrame(rows)


def _save_plot_rate_distortion(df: pd.DataFrame, path: Path) -> None:
    """Rate-distortion curve: achievable rate vs epsilon, multi-benchmark."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if "benchmark" in df.columns:
        for bench in sorted(df["benchmark"].unique()):
            sub = df[df["benchmark"] == bench].sort_values("epsilon")
            ax.plot(sub["epsilon"], sub["achievable_rate_bits"], marker="o", label=bench)
        ax.legend(frameon=False)
    else:
        df = df.sort_values("epsilon")
        ax.plot(df["epsilon"], df["achievable_rate_bits"], marker="o", color="#4E79A7")
    ax.set_xlabel("Distortion (epsilon)")
    ax.set_ylabel("Rate (bits) = log2 |Q~|")
    ax.set_title("Rate-Distortion (m=1, T=2)")
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def _save_plot_reward_planning(df: pd.DataFrame, path: Path) -> None:
    """Bar chart: value gap and compression ratio for reward-based planning."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: value gap vs epsilon, grouped by (m, T)
    ax = axes[0]
    groups = df.groupby(["m", "horizon"])
    for (m_val, h), sub in groups:
        sub = sub.sort_values("epsilon")
        label = f"m={int(m_val)}, T={int(h)}"
        ax.plot(sub["epsilon"], sub["value_gap"].abs(), marker="o", label=label)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Value gap (original)")
    ax.set_title("Quotient planning value gap")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    # Right panel: compression ratio vs epsilon
    ax = axes[1]
    for (m_val, h), sub in groups:
        sub = sub.sort_values("epsilon")
        label = f"m={int(m_val)}, T={int(h)}"
        ax.plot(sub["epsilon"], sub["compression_ratio"], marker="s", label=label)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Compression ratio |Q|/|H|")
    ax.set_title("History compression")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def run_hyperparameter_sensitivity(
    eps_grid: Sequence[float],
    ms: Sequence[int] = (1, 2),
    horizons: Sequence[int] = (2, 3, 4),
) -> pd.DataFrame:
    """Sweep (epsilon, m, T) grid for heatmap data."""
    rows: List[Dict[str, float]] = []

    for m in ms:
        for horizon in horizons:
            pomdp = tiger_full_actions_pomdp()
            d_obs = tiger_discrete_observation_metric()
            policies = enumerate_deterministic_fscs(
                num_actions=pomdp.num_actions,
                num_observations=pomdp.num_observations,
                max_nodes=m, include_smaller=True,
            )
            cache = precompute_distance_cache(
                pomdp=pomdp, policies=policies, horizon=horizon,
                distance_mode="w1", d_obs=d_obs,
            )

            for eps in eps_grid:
                part = compute_partition_from_cache(cache, epsilon=float(eps))
                rows.append({
                    "m": float(m),
                    "horizon": float(horizon),
                    "epsilon": float(eps),
                    "class_count": float(part.num_classes_total),
                })

    return pd.DataFrame(rows)


def _save_plot_hyperparameter_heatmap(df: pd.DataFrame, path: Path) -> None:
    ms = sorted(df["m"].unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(6 * len(ms), 4), squeeze=False)

    for idx, m in enumerate(ms):
        ax = axes[0, idx]
        sub = df[df["m"] == m]
        eps_vals = sorted(sub["epsilon"].unique())
        h_vals = sorted(sub["horizon"].unique())
        grid = np.full((len(h_vals), len(eps_vals)), np.nan)
        for i, h in enumerate(h_vals):
            for j, e in enumerate(eps_vals):
                row = sub[(sub["horizon"] == h) & (sub["epsilon"] == e)]
                if not row.empty:
                    grid[i, j] = row["class_count"].iloc[0]

        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="YlOrRd")
        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels([f"{e:.1f}" for e in eps_vals], fontsize=7, rotation=45)
        ax.set_yticks(range(len(h_vals)))
        ax.set_yticklabels([f"{int(h)}" for h in h_vals])
        ax.set_xlabel("epsilon")
        ax.set_ylabel("Horizon T")
        ax.set_title(f"Class Count (m={int(m)})")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    _save_fig(fig, path)
    plt.close(fig)


def run_medium_scale_experiment(
    eps_grid: Sequence[float],
    seed: int = 42,
) -> pd.DataFrame:
    """Demonstrate approximate quotient on a POMDP with |S| >= 50 using sampling.

    Uses sampling-based distance estimation (sampling.py) to handle the larger
    state space where exact enumeration of observation distributions is costly.
    """
    from .sampling import sampling_based_distance_cache

    rows: List[Dict[str, float]] = []

    # GridWorld (|S|=64 and |S|=100) with sampling-based distance estimation
    for gsize in (8, 10):
        pomdp = gridworld_pomdp(size=gsize)
        d_obs = gridworld_geometric_observation_metric()
        policies = _policies_m1_for_pomdp(pomdp)
        horizon = 2

        t0 = time.perf_counter()
        cache = sampling_based_distance_cache(
            pomdp=pomdp,
            policies=policies,
            horizon=horizon,
            d_obs=d_obs,
            num_samples=500,
            seed=seed,
        )
        cache_time = time.perf_counter() - t0

        for eps in eps_grid:
            part = compute_partition_from_cache(cache, epsilon=float(eps))
            total_h = sum(
                len(cache.histories_by_depth[d])
                for d in range(cache.horizon + 1)
            )
            rows.append({
                "benchmark": f"gridworld_{gsize}x{gsize}",
                "num_states": float(gsize * gsize),
                "horizon": float(horizon),
                "m": 1.0,
                "epsilon": float(eps),
                "class_count": float(part.num_classes_total),
                "total_histories": float(total_h),
                "compression_ratio": float(part.num_classes_total) / float(total_h),
                "method": "sampling",
                "num_samples": 500.0,
                "cache_time_s": round(cache_time, 4),
            })

    # Random structured POMDP with |S|=50
    for rseed in (seed, seed + 1, seed + 2):
        pomdp = random_structured_pomdp(num_states=50, num_actions=3, num_observations=4, seed=rseed)
        d_obs = random_observation_metric(pomdp.num_observations, seed=rseed)
        policies = _policies_m1_for_pomdp(pomdp)
        horizon = 2

        t0 = time.perf_counter()
        cache = sampling_based_distance_cache(
            pomdp=pomdp,
            policies=policies,
            horizon=horizon,
            d_obs=d_obs,
            num_samples=500,
            seed=rseed,
        )
        cache_time = time.perf_counter() - t0

        for eps in eps_grid:
            part = compute_partition_from_cache(cache, epsilon=float(eps))
            total_h = sum(
                len(cache.histories_by_depth[d])
                for d in range(cache.horizon + 1)
            )
            rows.append({
                "benchmark": "random_pomdp_50",
                "num_states": 50.0,
                "horizon": float(horizon),
                "m": 1.0,
                "epsilon": float(eps),
                "class_count": float(part.num_classes_total),
                "total_histories": float(total_h),
                "compression_ratio": float(part.num_classes_total) / float(total_h),
                "method": "sampling",
                "num_samples": 500.0,
                "instance_seed": float(rseed),
                "cache_time_s": round(cache_time, 4),
            })

    # Random structured POMDP with |S|=100
    for rseed in (seed, seed + 1, seed + 2):
        pomdp = random_structured_pomdp(
            num_states=100, num_actions=3, num_observations=4, seed=rseed,
        )
        d_obs = random_observation_metric(pomdp.num_observations, seed=rseed)
        policies = _policies_m1_for_pomdp(pomdp)
        horizon = 2

        t0 = time.perf_counter()
        cache = sampling_based_distance_cache(
            pomdp=pomdp,
            policies=policies,
            horizon=horizon,
            d_obs=d_obs,
            num_samples=500,
            seed=rseed,
        )
        cache_time = time.perf_counter() - t0

        for eps in eps_grid:
            part = compute_partition_from_cache(cache, epsilon=float(eps))
            total_h = sum(
                len(cache.histories_by_depth[d])
                for d in range(cache.horizon + 1)
            )
            rows.append({
                "benchmark": "random_pomdp_100",
                "num_states": 100.0,
                "horizon": float(horizon),
                "m": 1.0,
                "epsilon": float(eps),
                "class_count": float(part.num_classes_total),
                "total_histories": float(total_h),
                "compression_ratio": float(part.num_classes_total) / float(total_h),
                "method": "sampling",
                "num_samples": 500.0,
                "instance_seed": float(rseed),
                "cache_time_s": round(cache_time, 4),
            })

    return pd.DataFrame(rows)


def run_data_processing_experiment(
    eps_grid: Sequence[float],
) -> pd.DataFrame:
    """Validate data-processing monotonicity (Theorem 5.3/7.1) experimentally.

    Coarsens GridWorld observations from 4 (NW/NE/SW/SE) to 2 (North/South)
    and verifies that D^W on the coarsened POMDP is <= L_C * D^W on the original
    for every tested epsilon.
    """
    rows: List[Dict[str, object]] = []

    merge_map = {0: 0, 1: 0, 2: 1, 3: 1}  # NW,NE -> 0; SW,SE -> 1

    for gsize in (3, 5):
        pomdp = gridworld_pomdp(size=gsize)
        d_obs = gridworld_geometric_observation_metric()

        coarsened = coarsen_observations(pomdp, merge_map)
        new_d_obs, L_C = coarsened_observation_metric(d_obs, merge_map)

        # Original distance cache
        policies_orig = _policies_m1_for_pomdp(pomdp)
        cache_orig = precompute_distance_cache(
            pomdp=pomdp,
            policies=policies_orig,
            horizon=2,
            distance_mode="w1",
            d_obs=d_obs,
        )

        # Coarsened distance cache
        policies_coarsened = _policies_m1_for_pomdp(coarsened)
        cache_coarsened = precompute_distance_cache(
            pomdp=coarsened,
            policies=policies_coarsened,
            horizon=2,
            distance_mode="w1",
            d_obs=new_d_obs,
        )

        # Max pairwise pseudometric distances
        max_d_orig = max(
            float(cache_orig.max_distance_matrices[d].max())
            for d in cache_orig.max_distance_matrices
            if cache_orig.max_distance_matrices[d].size > 0
        ) if cache_orig.max_distance_matrices else 0.0
        max_d_coarsened = max(
            float(cache_coarsened.max_distance_matrices[d].max())
            for d in cache_coarsened.max_distance_matrices
            if cache_coarsened.max_distance_matrices[d].size > 0
        ) if cache_coarsened.max_distance_matrices else 0.0

        for eps in eps_grid:
            part_orig = compute_partition_from_cache(cache_orig, epsilon=float(eps))
            part_coarsened = compute_partition_from_cache(cache_coarsened, epsilon=float(eps))

            monotonicity_holds = max_d_coarsened <= L_C * max_d_orig + 1e-9

            rows.append({
                "benchmark": f"gridworld_{gsize}x{gsize}",
                "channel_type": "deterministic",
                "epsilon": float(eps),
                "original_classes": float(part_orig.num_classes_total),
                "coarsened_classes": float(part_coarsened.num_classes_total),
                "original_max_d": round(max_d_orig, 6),
                "coarsened_max_d": round(max_d_coarsened, 6),
                "L_C": float(L_C),
                "monotonicity_holds": bool(monotonicity_holds),
            })

    # --- Stochastic channel test ---
    stochastic_channel = np.array([
        [0.8, 0.2],  # NW
        [0.7, 0.3],  # NE
        [0.2, 0.8],  # SW
        [0.3, 0.7],  # SE
    ], dtype=float)
    new_d_obs_stoch = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    for gsize in (3, 5):
        pomdp = gridworld_pomdp(size=gsize)
        d_obs = gridworld_geometric_observation_metric()

        coarsened = stochastic_coarsen_observations(pomdp, stochastic_channel)
        L_C = stochastic_channel_lipschitz_constant(d_obs, stochastic_channel, new_d_obs_stoch)

        policies_orig = _policies_m1_for_pomdp(pomdp)
        cache_orig = precompute_distance_cache(
            pomdp=pomdp, policies=policies_orig, horizon=2,
            distance_mode="w1", d_obs=d_obs,
        )

        policies_coarsened = _policies_m1_for_pomdp(coarsened)
        cache_coarsened = precompute_distance_cache(
            pomdp=coarsened, policies=policies_coarsened, horizon=2,
            distance_mode="w1", d_obs=new_d_obs_stoch,
        )

        max_d_orig = max(
            float(cache_orig.max_distance_matrices[d].max())
            for d in cache_orig.max_distance_matrices
            if cache_orig.max_distance_matrices[d].size > 0
        ) if cache_orig.max_distance_matrices else 0.0
        max_d_coarsened = max(
            float(cache_coarsened.max_distance_matrices[d].max())
            for d in cache_coarsened.max_distance_matrices
            if cache_coarsened.max_distance_matrices[d].size > 0
        ) if cache_coarsened.max_distance_matrices else 0.0

        for eps in eps_grid:
            part_orig = compute_partition_from_cache(cache_orig, epsilon=float(eps))
            part_coarsened = compute_partition_from_cache(cache_coarsened, epsilon=float(eps))

            monotonicity_holds = max_d_coarsened <= L_C * max_d_orig + 1e-9

            rows.append({
                "benchmark": f"gridworld_{gsize}x{gsize}",
                "channel_type": "stochastic",
                "epsilon": float(eps),
                "original_classes": float(part_orig.num_classes_total),
                "coarsened_classes": float(part_coarsened.num_classes_total),
                "original_max_d": round(max_d_orig, 6),
                "coarsened_max_d": round(max_d_coarsened, 6),
                "L_C": round(float(L_C), 6),
                "monotonicity_holds": bool(monotonicity_holds),
            })

    return pd.DataFrame(rows)


def run_observation_sensitivity_experiment(
    eps_grid: Sequence[float],
    delta_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> pd.DataFrame:
    """Validate observation-sensitivity monotonicity (Proposition obs-resolution).

    For each delta_O value, compute the delta-covering of the observation space,
    build the coarsened POMDP, and compare partitions against full resolution.
    """
    rows: List[Dict[str, object]] = []

    benchmarks = [
        ("gridworld_3x3", gridworld_pomdp(size=3), gridworld_geometric_observation_metric()),
        ("tiger", tiger_full_actions_pomdp(), tiger_discrete_observation_metric()),
    ]

    for bench_name, pomdp, d_obs in benchmarks:
        # Full-resolution reference
        policies_full = _policies_m1_for_pomdp(pomdp)
        cache_full = precompute_distance_cache(
            pomdp=pomdp,
            policies=policies_full,
            horizon=2,
            distance_mode="w1",
            d_obs=d_obs,
        )

        max_d_full = max(
            float(cache_full.max_distance_matrices[d].max())
            for d in cache_full.max_distance_matrices
            if cache_full.max_distance_matrices[d].size > 0
        ) if cache_full.max_distance_matrices else 0.0

        for delta_o in delta_grid:
            new_d_obs, merge_map = delta_covering_coarsen(d_obs, delta_o)
            coarsened = coarsen_observations(pomdp, merge_map)
            num_obs_coarsened = coarsened.num_observations

            policies_coarsened = _policies_m1_for_pomdp(coarsened)
            cache_coarsened = precompute_distance_cache(
                pomdp=coarsened,
                policies=policies_coarsened,
                horizon=2,
                distance_mode="w1",
                d_obs=new_d_obs,
            )

            max_d_coarsened = max(
                float(cache_coarsened.max_distance_matrices[d].max())
                for d in cache_coarsened.max_distance_matrices
                if cache_coarsened.max_distance_matrices[d].size > 0
            ) if cache_coarsened.max_distance_matrices else 0.0

            T = 2
            monotonicity_holds = max_d_coarsened <= max_d_full + T * delta_o + 1e-9

            for eps in eps_grid:
                part_full = compute_partition_from_cache(cache_full, epsilon=float(eps))
                part_coarsened = compute_partition_from_cache(cache_coarsened, epsilon=float(eps))

                rows.append({
                    "benchmark": bench_name,
                    "delta_o": float(delta_o),
                    "epsilon": float(eps),
                    "classes_full": float(part_full.num_classes_total),
                    "classes_coarsened": float(part_coarsened.num_classes_total),
                    "num_obs_original": float(pomdp.num_observations),
                    "num_obs_coarsened": float(num_obs_coarsened),
                    "max_distance_full": round(max_d_full, 6),
                    "max_distance_coarsened": round(max_d_coarsened, 6),
                    "monotonicity_holds": bool(monotonicity_holds),
                })

    return pd.DataFrame(rows)


def run_clustering_optimality_check(
    eps_grid: Sequence[float],
) -> pd.DataFrame:
    """Verify greedy complete-linkage matches optimal for small instances."""
    from .clustering import clustering_optimality_gap

    pomdp = tiger_full_actions_pomdp()
    d_obs = tiger_discrete_observation_metric()

    rows: List[Dict[str, object]] = []
    for horizon in (2, 3, 4):
        policies = _policies_m1_for_pomdp(pomdp)
        cache = precompute_distance_cache(
            pomdp=pomdp, policies=policies, horizon=horizon,
            distance_mode="w1", d_obs=d_obs,
        )
        for eps in eps_grid:
            for depth in range(horizon + 1):
                mat = cache.max_distance_matrices[depth]
                gap_info = clustering_optimality_gap(mat, float(eps))
                rows.append({
                    "benchmark": "tiger_full_actions",
                    "horizon": horizon,
                    "depth": depth,
                    "epsilon": float(eps),
                    "n_items": int(mat.shape[0]),
                    "greedy_clusters": gap_info["greedy_num_clusters"],
                    "optimal_clusters": gap_info["optimal_num_clusters"],
                    "gap": gap_info["gap"],
                    "greedy_is_optimal": gap_info["greedy_is_optimal"],
                })

    return pd.DataFrame(rows)


def run_effective_dimension(
    eps_grid: Sequence[float],
) -> pd.DataFrame:
    """Compute effective dimension d_eff for each benchmark per Remark 5.9.

    Reports max_h ||b_h||_0 (support size) and exp(H(b_h)) (entropic dimension)
    for tighter bounds when beliefs are low-dimensional.
    """
    from .pomdp_core import all_histories, belief_after_history

    benchmarks: List[Tuple[str, FinitePOMDP, np.ndarray, int]] = [
        ("tiger_full_actions", tiger_full_actions_pomdp(), tiger_discrete_observation_metric(), 2),
        ("gridworld_3x3", gridworld_pomdp(size=3), gridworld_geometric_observation_metric(), 2),
        ("gridworld_5x5", gridworld_pomdp(size=5), gridworld_geometric_observation_metric(), 2),
    ]

    rows: List[Dict[str, float]] = []
    for bench_name, pomdp, d_obs, horizon in benchmarks:
        policies = _policies_m1_for_pomdp(pomdp)
        policy = policies[0]
        histories_by_d = all_histories(num_observations=pomdp.num_observations, horizon=horizon)

        cache = precompute_distance_cache(
            pomdp=pomdp, policies=policies, horizon=horizon,
            distance_mode="w1", d_obs=d_obs,
        )

        # Compute belief properties across all histories
        max_support_size = 0
        max_entropic_dim = 0.0
        all_supports = []
        all_entropies = []

        for d in range(horizon + 1):
            for h in histories_by_d[d]:
                b = belief_after_history(pomdp=pomdp, policy=policy, history=h)
                support = int(np.sum(b > 1e-12))
                max_support_size = max(max_support_size, support)
                all_supports.append(support)

                # Shannon entropy
                mask = b > 1e-12
                entropy = -float(np.sum(b[mask] * np.log(b[mask])))
                entropic_dim = float(np.exp(entropy))
                max_entropic_dim = max(max_entropic_dim, entropic_dim)
                all_entropies.append(entropic_dim)

        for eps in eps_grid:
            part = compute_partition_from_cache(cache, epsilon=float(eps))
            l_r = 1.0
            standard_bound = l_r * horizon * float(eps) * (1.0 + 2.0 * horizon * pomdp.num_states * pomdp.num_observations)
            # Tighter bound using effective dimension instead of |S|
            d_eff = max_entropic_dim
            tighter_bound = l_r * horizon * float(eps) * (1.0 + 2.0 * horizon * d_eff * pomdp.num_observations)

            rows.append({
                "benchmark": bench_name,
                "num_states": float(pomdp.num_states),
                "horizon": float(horizon),
                "epsilon": float(eps),
                "class_count": float(part.num_classes_total),
                "max_support_size": float(max_support_size),
                "mean_support_size": float(np.mean(all_supports)),
                "max_entropic_dim": float(max_entropic_dim),
                "mean_entropic_dim": float(np.mean(all_entropies)),
                "standard_canonical_bound": float(standard_bound),
                "tighter_d_eff_bound": float(tighter_bound),
                "bound_improvement_ratio": float(standard_bound / max(tighter_bound, 1e-12)) if eps > 0 else 1.0,
            })

    return pd.DataFrame(rows)


def run_baseline_sensitivity(
    eps_grid: Sequence[float],
) -> pd.DataFrame:
    """Sensitivity analysis for bisimulation baseline hyperparameters.

    Sweeps discount gamma in {0.5, 0.7, 0.9, 0.99} and iterations in {5, 10, 20, 50}.
    Reports class count and computation time per configuration.
    """
    from .baselines import bisimulation_metric_partition

    pomdp = tiger_full_actions_pomdp()
    horizon = 2
    d_obs = tiger_discrete_observation_metric()
    policies = _policies_m1_for_pomdp(pomdp)
    sequence_score = lambda seq: 1.0 if all(o == 0 for o in seq) else 0.0

    gammas = (0.5, 0.7, 0.9, 0.99)
    iterations_list = (5, 10, 20, 50)

    rows: List[Dict[str, float]] = []
    for gamma in gammas:
        for n_iter in iterations_list:
            for eps in eps_grid:
                t0 = time.perf_counter()
                bisim_part = bisimulation_metric_partition(
                    pomdp=pomdp, horizon=horizon, epsilon=float(eps),
                    num_iterations=n_iter, discount=gamma,
                )
                elapsed = time.perf_counter() - t0

                bisim_err = _max_value_error(pomdp, bisim_part, policies, horizon, sequence_score)

                rows.append({
                    "benchmark": "tiger_full_actions",
                    "gamma": float(gamma),
                    "iterations": float(n_iter),
                    "epsilon": float(eps),
                    "class_count": float(bisim_part.num_classes_total),
                    "max_value_error": float(bisim_err),
                    "time_s": round(elapsed, 4),
                })

    return pd.DataFrame(rows)


def run_initial_belief_sensitivity(
    eps_grid: Sequence[float],
    beliefs: Sequence[Sequence[float]] | None = None,
) -> pd.DataFrame:
    """Vary Tiger initial belief and report partition stability."""
    if beliefs is None:
        beliefs = [
            [0.5, 0.5],
            [0.3, 0.7],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.9, 0.1],
        ]

    horizon = 2
    d_obs = tiger_discrete_observation_metric()

    # Baseline partition at uniform belief
    baseline_pomdp = tiger_full_actions_pomdp_custom_belief(b0=[0.5, 0.5])
    baseline_pols = _policies_m1_for_pomdp(baseline_pomdp)
    baseline_cache = precompute_distance_cache(
        pomdp=baseline_pomdp, policies=baseline_pols, horizon=horizon,
        distance_mode="w1", d_obs=d_obs,
    )

    rows: List[Dict[str, object]] = []
    for b0 in beliefs:
        pomdp = tiger_full_actions_pomdp_custom_belief(b0=b0)
        policies = _policies_m1_for_pomdp(pomdp)
        cache = precompute_distance_cache(
            pomdp=pomdp, policies=policies, horizon=horizon,
            distance_mode="w1", d_obs=d_obs,
        )

        for eps in eps_grid:
            part = compute_partition_from_cache(cache, epsilon=float(eps))
            baseline_part = compute_partition_from_cache(baseline_cache, epsilon=float(eps))

            # Compute partition agreement with baseline (uniform belief)
            from .spectral import partition_agreement as _pa
            agreement = _pa(baseline_part, part)

            rows.append({
                "b0_left": float(b0[0]),
                "b0_right": float(b0[1]),
                "epsilon": float(eps),
                "class_count": float(part.num_classes_total),
                "baseline_class_count": float(baseline_part.num_classes_total),
                "adjusted_rand_index": float(agreement["adjusted_rand_index"]),
            })

    return pd.DataFrame(rows)


def run_new_benchmark_experiments(
    eps_grid: Sequence[float],
    ms: Sequence[int] = (1,),
) -> pd.DataFrame:
    """Capacity sweep on Hallway and Network Monitoring benchmarks."""
    benchmarks: List[Tuple[str, FinitePOMDP, np.ndarray, int]] = [
        ("hallway_10", hallway_pomdp(length=10, num_landmarks=3), discrete_observation_metric(3), 2),
        ("hallway_20", hallway_pomdp(length=20, num_landmarks=3), discrete_observation_metric(3), 2),
        ("network_4", network_monitoring_pomdp(num_nodes=4, num_alerts=3), discrete_observation_metric(3), 2),
        ("network_5", network_monitoring_pomdp(num_nodes=5, num_alerts=3), discrete_observation_metric(3), 2),
    ]

    rows: List[Dict[str, object]] = []
    for bench_name, pomdp, d_obs, horizon in benchmarks:
        for m_val in ms:
            policies = enumerate_deterministic_fscs(
                num_actions=pomdp.num_actions,
                num_observations=pomdp.num_observations,
                max_nodes=m_val, include_smaller=True,
            )
            cache = precompute_distance_cache(
                pomdp=pomdp, policies=policies, horizon=horizon,
                distance_mode="w1", d_obs=d_obs,
            )
            total_h = sum(len(cache.histories_by_depth[d]) for d in range(cache.horizon + 1))

            for eps in eps_grid:
                part = compute_partition_from_cache(cache, epsilon=float(eps))
                rows.append({
                    "benchmark": bench_name,
                    "num_states": float(pomdp.num_states),
                    "num_actions": float(pomdp.num_actions),
                    "num_observations": float(pomdp.num_observations),
                    "m": float(m_val),
                    "horizon": float(horizon),
                    "epsilon": float(eps),
                    "class_count": float(part.num_classes_total),
                    "total_histories": float(total_h),
                    "compression_ratio": float(part.num_classes_total) / float(total_h) if total_h > 0 else 1.0,
                })

    return pd.DataFrame(rows)


def run_reward_planning_experiment(
    eps_grid: Sequence[float],
    ms: Sequence[int] = (1, 2),
    horizons: Sequence[int] = (2, 3),
) -> pd.DataFrame:
    """Compare exact vs quotient-based policy selection using real POMDP rewards.

    For each (benchmark, m, T, epsilon):
      1. Enumerate all deterministic FSCs.
      2. *Exact planning*: evaluate every FSC on the original POMDP with
         ``value_state_action_original`` and pick the best.
      3. *Quotient planning*: evaluate every FSC on the ε-quotient with
         ``value_state_action_quotient`` and pick the best.
      4. Report the value gap (exact-best value minus quotient-best value,
         both measured on the original), plus timing and compression.
    """
    # GridWorld with goal reward: +1 for being in the corner cell, 0 otherwise.
    gw = gridworld_pomdp(size=3)
    goal_rewards = np.zeros_like(gw.rewards)
    goal_state = gw.num_states - 1  # bottom-right corner
    goal_rewards[goal_state, :] = 1.0
    gw_goal = FinitePOMDP(
        state_names=gw.state_names,
        action_names=gw.action_names,
        observation_names=gw.observation_names,
        transition=gw.transition,
        observation=gw.observation,
        rewards=goal_rewards,
        initial_belief=gw.initial_belief,
    )

    benchmarks: List[Tuple[str, FinitePOMDP, np.ndarray]] = [
        ("tiger", tiger_full_actions_pomdp(), tiger_discrete_observation_metric()),
        ("gridworld_goal", gw_goal, gridworld_geometric_observation_metric()),
    ]

    max_policies = 500  # skip (benchmark, m) combos with too many FSCs

    rows: List[Dict[str, object]] = []
    for bench_name, pomdp, d_obs in benchmarks:
        for m_val in ms:
            policies = enumerate_deterministic_fscs(
                num_actions=pomdp.num_actions,
                num_observations=pomdp.num_observations,
                max_nodes=m_val, include_smaller=True,
            )
            if len(policies) > max_policies:
                continue
            for horizon in horizons:
                cache = precompute_distance_cache(
                    pomdp=pomdp, policies=policies, horizon=horizon,
                    distance_mode="w1", d_obs=d_obs,
                )
                total_h = sum(
                    len(cache.histories_by_depth[d])
                    for d in range(horizon + 1)
                )

                # Exact planning on original
                t0 = time.perf_counter()
                orig_vals = [
                    value_state_action_original(pomdp=pomdp, policy=p, horizon=horizon)
                    for p in policies
                ]
                best_orig_idx = int(np.argmax(orig_vals))
                best_orig_val = orig_vals[best_orig_idx]
                time_orig = time.perf_counter() - t0

                for eps in eps_grid:
                    partition = compute_partition_from_cache(cache, epsilon=float(eps))

                    # Quotient planning
                    t0 = time.perf_counter()
                    quot_vals = [
                        value_state_action_quotient(
                            pomdp=pomdp, partition=partition, policy=p,
                        )
                        for p in policies
                    ]
                    best_quot_idx = int(np.argmax(quot_vals))
                    time_quot = time.perf_counter() - t0

                    # Evaluate quotient-best policy on original
                    val_quot_on_orig = orig_vals[best_quot_idx]

                    rows.append({
                        "benchmark": bench_name,
                        "m": int(m_val),
                        "horizon": int(horizon),
                        "epsilon": float(eps),
                        "num_policies": int(len(policies)),
                        "num_classes": int(partition.num_classes_total),
                        "total_histories": int(total_h),
                        "compression_ratio": round(
                            float(partition.num_classes_total) / float(total_h), 4
                        ),
                        "time_original_s": round(time_orig, 6),
                        "time_quotient_s": round(time_quot, 6),
                        "speedup_ratio": round(
                            time_orig / max(time_quot, 1e-9), 2
                        ),
                        "value_exact_best": round(best_orig_val, 6),
                        "value_quotient_best_on_orig": round(val_quot_on_orig, 6),
                        "value_gap": round(best_orig_val - val_quot_on_orig, 6),
                        "relative_value_gap": round(
                            abs(best_orig_val - val_quot_on_orig)
                            / max(abs(best_orig_val), 1e-9),
                            6,
                        ),
                        "same_policy": bool(best_orig_idx == best_quot_idx),
                    })

    return pd.DataFrame(rows)


def run_planning_speedup_experiment(
    eps_grid: Sequence[float],
    ms: Sequence[int] = (1,),
) -> pd.DataFrame:
    """Compare planning time on original vs quotient POMDP.

    For each benchmark, enumerate all deterministic FSCs, find the best one
    on the original vs the quotient, and compare wall-clock time and value gap.
    """
    sequence_score = lambda seq: 1.0 if all(o == 0 for o in seq) else 0.0
    horizon = 2

    benchmarks: List[Tuple[str, FinitePOMDP, np.ndarray]] = [
        ("tiger_full_actions", tiger_full_actions_pomdp(), tiger_discrete_observation_metric()),
        ("hallway_10", hallway_pomdp(length=10, num_landmarks=3), discrete_observation_metric(3)),
    ]

    rows: List[Dict[str, object]] = []
    for bench_name, pomdp, d_obs in benchmarks:
        for m_val in ms:
            policies = enumerate_deterministic_fscs(
                num_actions=pomdp.num_actions,
                num_observations=pomdp.num_observations,
                max_nodes=m_val, include_smaller=True,
            )
            cache = precompute_distance_cache(
                pomdp=pomdp, policies=policies, horizon=horizon,
                distance_mode="w1", d_obs=d_obs,
            )

            # Original: find best policy by exhaustive evaluation
            t0 = time.perf_counter()
            orig_values = []
            for p in policies:
                p_m = trajectory_observation_distribution(pomdp=pomdp, policy=p, horizon=horizon)
                v_m = expected_sequence_score(p_m, score_fn=sequence_score)
                orig_values.append(v_m)
            best_orig_idx = int(np.argmax(orig_values))
            best_orig_value = orig_values[best_orig_idx]
            time_original = time.perf_counter() - t0

            for eps in eps_grid:
                partition = compute_partition_from_cache(cache, epsilon=float(eps))

                # Quotient: find best policy on quotient
                t0 = time.perf_counter()
                quot_values = []
                for p in policies:
                    p_q = quotient_observation_sequence_distribution(
                        pomdp=pomdp, partition=partition, policy=p,
                    )
                    v_q = expected_sequence_score(p_q, score_fn=sequence_score)
                    quot_values.append(v_q)
                best_quot_idx = int(np.argmax(quot_values))
                time_quotient = time.perf_counter() - t0

                # Evaluate quotient-best policy on original
                p_m_qbest = trajectory_observation_distribution(
                    pomdp=pomdp, policy=policies[best_quot_idx], horizon=horizon,
                )
                v_quot_on_orig = expected_sequence_score(p_m_qbest, score_fn=sequence_score)

                rows.append({
                    "benchmark": bench_name,
                    "m": float(m_val),
                    "epsilon": float(eps),
                    "num_policies": float(len(policies)),
                    "num_classes": float(partition.num_classes_total),
                    "time_original_s": round(time_original, 6),
                    "time_quotient_s": round(time_quotient, 6),
                    "speedup_ratio": round(time_original / max(time_quotient, 1e-9), 2),
                    "value_original": round(best_orig_value, 8),
                    "value_quotient_on_original": round(v_quot_on_orig, 8),
                    "value_gap": round(best_orig_value - v_quot_on_orig, 8),
                })

    return pd.DataFrame(rows)


def run_all_experiments(
    output_dir: Path,
    profile: str = "quick",
    eps_grid: Sequence[float] | None = None,
    ms: Sequence[int] | None = None,
    horizons: Sequence[int] | None = None,
    include_nonlipschitz: bool = True,
    include_spectral: bool = True,
    include_hierarchical_scaling: bool = False,
    stochastic_samples: int | None = None,
    max_horizon: int = 10,
    segment_horizon: int | None = None,
    skip_plots: bool = False,
    seed: int = 7,
    parallel: bool = False,
    workers: int | None = None,
    worker_fraction: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    cfg = _build_config(
        profile=profile,
        eps_grid=eps_grid or (),
        ms=ms or (),
        horizons=horizons or (),
        include_nonlipschitz=include_nonlipschitz,
        include_hierarchical_scaling=include_hierarchical_scaling,
        stochastic_samples=stochastic_samples,
        seed=seed,
        max_horizon=max_horizon,
        segment_horizon=segment_horizon,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _set_thread_caps_if_unset()

    resolved_workers = _resolve_parallel_workers(
        parallel=parallel,
        workers=workers,
        worker_fraction=worker_fraction,
    )
    spectral_horizon = min(cfg.horizons) if cfg.horizons else 2
    spectral_ks = (1, 3, 5, 10)

    tasks: List[ExperimentTask] = [
        ExperimentTask("tiger_reproduction", tiger_reproduction_sanity, {}),
        ExperimentTask(
            "capacity_sweep_tiger",
            run_capacity_sweep_tiger,
            {"eps_grid": cfg.eps_grid, "ms": cfg.ms},
        ),
        ExperimentTask(
            "capacity_sweep_gridworld",
            run_capacity_sweep_gridworld,
            {"eps_grid": cfg.eps_grid, "ms": cfg.ms, "horizon": 2},
        ),
        ExperimentTask(
            "value_loss_bounds_lipschitz",
            run_lipschitz_value_bounds,
            {"eps_grid": cfg.eps_grid},
        ),
        ExperimentTask(
            "value_loss_nonlipschitz_tiger",
            run_nonlipschitz_value_track,
            {"eps_grid": cfg.eps_grid},
        ),
        ExperimentTask(
            "horizon_gap_tiger",
            run_horizon_gap_tiger,
            {"horizons": cfg.horizons, "epsilon": 0.5},
        ),
        ExperimentTask(
            "metric_sensitivity_gridworld",
            run_gridworld_metric_sensitivity,
            {"eps_grid": cfg.eps_grid},
        ),
        ExperimentTask(
            "stochastic_vs_deterministic_sanity",
            run_stochastic_vs_deterministic_sanity,
            {"stochastic_samples": cfg.stochastic_samples, "seed": cfg.seed},
        ),
        ExperimentTask(
            "observation_noise_sensitivity",
            run_observation_noise_sensitivity,
            {"eps_grid": cfg.eps_grid},
        ),
        ExperimentTask(
            "multi_seed_witness",
            run_multi_seed_witness,
            {"stochastic_samples": min(cfg.stochastic_samples, 200)},
        ),
        ExperimentTask(
            "baseline_comparison",
            run_baseline_comparison,
            {"eps_grid": cfg.eps_grid, "include_gridworld": True},
        ),
        ExperimentTask(
            "ablation_studies",
            run_ablation_studies,
            {"eps_grid": cfg.eps_grid},
        ),
        ExperimentTask(
            "hyperparameter_sensitivity",
            run_hyperparameter_sensitivity,
            {"eps_grid": cfg.eps_grid, "ms": cfg.ms, "horizons": cfg.horizons},
        ),
        ExperimentTask(
            "larger_scale",
            run_larger_scale_experiments,
            {"eps_grid": cfg.eps_grid, "seed": cfg.seed},
        ),
        ExperimentTask(
            "rate_distortion",
            run_rate_distortion_evaluation,
            {"eps_grid": cfg.eps_grid},
        ),
        ExperimentTask(
            "medium_scale",
            run_medium_scale_experiment,
            {"eps_grid": cfg.eps_grid, "seed": cfg.seed},
        ),
        ExperimentTask(
            "data_processing",
            run_data_processing_experiment,
            {"eps_grid": cfg.eps_grid},
        ),
        ExperimentTask(
            "observation_sensitivity",
            run_observation_sensitivity_experiment,
            {"eps_grid": cfg.eps_grid},
        ),
        ExperimentTask(
            "effective_dimension",
            run_effective_dimension,
            {"eps_grid": cfg.eps_grid},
        ),
        ExperimentTask(
            "baseline_sensitivity",
            run_baseline_sensitivity,
            {"eps_grid": cfg.eps_grid},
        ),
        ExperimentTask(
            "clustering_optimality",
            run_clustering_optimality_check,
            {"eps_grid": cfg.eps_grid},
        ),
        ExperimentTask(
            "initial_belief_sensitivity",
            run_initial_belief_sensitivity,
            {"eps_grid": cfg.eps_grid},
        ),
        ExperimentTask(
            "new_benchmarks",
            run_new_benchmark_experiments,
            {"eps_grid": cfg.eps_grid, "ms": cfg.ms},
        ),
        ExperimentTask(
            "planning_speedup",
            run_planning_speedup_experiment,
            {"eps_grid": cfg.eps_grid, "ms": (1,)},
        ),
        ExperimentTask(
            "reward_planning",
            run_reward_planning_experiment,
            {"eps_grid": cfg.eps_grid, "ms": cfg.ms, "horizons": (2, 3)},
        ),
        ExperimentTask(
            "capacity_extended",
            run_extended_capacity_experiment,
            {"eps_grid": cfg.eps_grid, "seed": cfg.seed},
        ),
        ExperimentTask(
            "convergence",
            run_convergence_analysis,
            {"eps_grid": cfg.eps_grid, "seed": cfg.seed},
        ),
        ExperimentTask(
            "m2_medium_scale",
            run_m2_medium_scale_experiment,
            {"eps_grid": cfg.eps_grid, "seed": cfg.seed},
        ),
        ExperimentTask(
            "spectral_rank_at_scale",
            run_spectral_rank_at_scale,
            {"seed": cfg.seed},
        ),
        ExperimentTask(
            "sampling_variance",
            run_sampling_variance_analysis,
            {"eps_grid": cfg.eps_grid, "base_seed": cfg.seed},
        ),
    ]

    if include_spectral:
        tasks.extend(
            [
                ExperimentTask(
                    "spectral_rank_analysis",
                    run_spectral_rank_analysis,
                    {"ms": cfg.ms, "horizon": spectral_horizon},
                ),
                ExperimentTask(
                    "spectral_partition_comparison",
                    run_spectral_partition_comparison,
                    {
                        "eps_grid": cfg.eps_grid,
                        "ms": cfg.ms,
                        "horizon": spectral_horizon,
                        "ks": spectral_ks,
                    },
                ),
            ]
        )

    if cfg.include_hierarchical_scaling:
        tasks.extend(
            [
                ExperimentTask(
                    "hierarchical_t_scaling",
                    run_hierarchical_t_scaling,
                    {
                        "max_horizon": cfg.max_horizon,
                        "segment_horizon": cfg.segment_horizon,
                        "epsilon": 0.5,
                        "m": 1,
                    },
                ),
                ExperimentTask(
                    "layered_bound_validation",
                    run_layered_bound_validation,
                    {
                        "eps_grid": cfg.eps_grid,
                        "max_horizon": cfg.max_horizon,
                        "epsilon": 0.5,
                    },
                ),
                ExperimentTask(
                    "principal_fsc_horizon_scaling",
                    run_principal_fsc_horizon_scaling,
                    {
                        "epsilon": 0.5,
                        "calibration_horizon": min(6, cfg.max_horizon),
                        "horizons_full": tuple(h for h in (4, 5, 6, 7) if h <= cfg.max_horizon),
                        "horizons_subset": tuple(h for h in (7, 8) if h <= cfg.max_horizon),
                        "ari_target": 0.999,
                        "default_k": 10,
                        "max_k_search": 20,
                    },
                ),
            ]
        )

    dfs, timings = _execute_experiment_tasks(tasks, workers=resolved_workers)

    # Persist tables
    dfs["capacity_sweep_tiger"].to_csv(output_dir / "capacity_sweep_tiger.csv", index=False)
    dfs["capacity_sweep_gridworld"].to_csv(output_dir / "capacity_sweep_gridworld.csv", index=False)
    dfs["value_loss_bounds_lipschitz"].to_csv(output_dir / "value_loss_bounds_lipschitz.csv", index=False)
    dfs["value_loss_nonlipschitz_tiger"].to_csv(output_dir / "value_loss_nonlipschitz_tiger.csv", index=False)
    dfs["horizon_gap_tiger"].to_csv(output_dir / "horizon_gap_tiger.csv", index=False)
    dfs["metric_sensitivity_gridworld"].to_csv(output_dir / "metric_sensitivity_gridworld.csv", index=False)
    dfs["stochastic_vs_deterministic_sanity"].to_csv(output_dir / "stochastic_vs_deterministic_sanity.csv", index=False)
    dfs["tiger_reproduction"].to_csv(output_dir / "tiger_reproduction.csv", index=False)
    dfs["observation_noise_sensitivity"].to_csv(output_dir / "observation_noise_sensitivity.csv", index=False)
    dfs["multi_seed_witness"].to_csv(output_dir / "multi_seed_witness.csv", index=False)
    dfs["baseline_comparison"].to_csv(output_dir / "baseline_comparison.csv", index=False)
    dfs["ablation_studies"].to_csv(output_dir / "ablation_studies.csv", index=False)
    dfs["hyperparameter_sensitivity"].to_csv(output_dir / "hyperparameter_sensitivity.csv", index=False)
    dfs["larger_scale"].to_csv(output_dir / "larger_scale.csv", index=False)
    dfs["rate_distortion"].to_csv(output_dir / "rate_distortion.csv", index=False)
    dfs["medium_scale"].to_csv(output_dir / "medium_scale.csv", index=False)
    dfs["data_processing"].to_csv(output_dir / "data_processing.csv", index=False)
    dfs["observation_sensitivity"].to_csv(output_dir / "observation_sensitivity.csv", index=False)
    dfs["effective_dimension"].to_csv(output_dir / "effective_dimension.csv", index=False)
    dfs["baseline_sensitivity"].to_csv(output_dir / "baseline_sensitivity.csv", index=False)
    dfs["clustering_optimality"].to_csv(output_dir / "clustering_optimality.csv", index=False)
    dfs["initial_belief_sensitivity"].to_csv(output_dir / "initial_belief_sensitivity.csv", index=False)
    dfs["new_benchmarks"].to_csv(output_dir / "new_benchmarks.csv", index=False)
    dfs["planning_speedup"].to_csv(output_dir / "planning_speedup.csv", index=False)
    dfs["reward_planning"].to_csv(output_dir / "reward_planning.csv", index=False)
    dfs["capacity_extended"].to_csv(output_dir / "capacity_extended.csv", index=False)
    dfs["convergence"].to_csv(output_dir / "convergence.csv", index=False)

    summary = {
        "profile": cfg.profile,
        "eps_grid": ",".join(str(x) for x in cfg.eps_grid),
        "ms": ",".join(str(m) for m in cfg.ms),
        "horizons": ",".join(str(h) for h in cfg.horizons),
        "include_nonlipschitz": cfg.include_nonlipschitz,
        "include_hierarchical_scaling": cfg.include_hierarchical_scaling,
        "stochastic_samples": cfg.stochastic_samples,
        "seed": cfg.seed,
        "max_horizon": cfg.max_horizon,
        "segment_horizon": cfg.segment_horizon if cfg.segment_horizon is not None else "",
        "parallel": bool(parallel),
        "workers": int(resolved_workers),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "results_summary.csv", index=False)

    if include_spectral:
        dfs["spectral_rank_analysis"].to_csv(output_dir / "spectral_rank_analysis.csv", index=False)
        dfs["spectral_partition_comparison"].to_csv(output_dir / "spectral_partition_comparison.csv", index=False)

    # m=2 medium-scale and spectral-at-scale experiments (W1 revision)
    dfs["m2_medium_scale"].to_csv(output_dir / "m2_medium_scale.csv", index=False)
    dfs["spectral_rank_at_scale"].to_csv(output_dir / "spectral_rank_at_scale.csv", index=False)
    dfs["sampling_variance"].to_csv(output_dir / "sampling_variance.csv", index=False)

    if cfg.include_hierarchical_scaling:
        dfs["hierarchical_t_scaling"].to_csv(output_dir / "hierarchical_t_scaling.csv", index=False)
        dfs["layered_bound_validation"].to_csv(output_dir / "layered_bound_validation.csv", index=False)
        dfs["principal_fsc_horizon_scaling"].to_csv(output_dir / "principal_fsc_horizon_scaling.csv", index=False)

    # Save timing summary
    timing_df = pd.DataFrame(timings)
    timing_df["platform"] = platform.platform()
    timing_df["python_version"] = platform.python_version()
    timing_df.to_csv(output_dir / "timing_summary.csv", index=False)

    if not skip_plots:
        _save_plot_capacity(dfs["capacity_sweep_tiger"], output_dir / "fig_capacity_vs_m.png")
        _save_plot_capacity(dfs["capacity_sweep_gridworld"], output_dir / "fig_capacity_gridworld.png")
        _save_plot_value_bounds(dfs["value_loss_bounds_lipschitz"], output_dir / "fig_value_loss_with_two_bounds.png")
        _save_plot_horizon_gap(dfs["horizon_gap_tiger"], output_dir / "fig_gap_vs_horizon.png")
        _save_plot_metric_sensitivity(dfs["metric_sensitivity_gridworld"], output_dir / "fig_w1_vs_tv_gridworld.png")
        _save_plot_noise_sensitivity(dfs["observation_noise_sensitivity"], output_dir / "fig_noise_sensitivity.png")
        _save_plot_baseline_comparison(dfs["baseline_comparison"], output_dir / "fig_baseline_comparison.png")
        _save_plot_hyperparameter_heatmap(dfs["hyperparameter_sensitivity"], output_dir / "fig_hyperparameter_heatmap.png")
        if include_spectral and "spectral_rank_analysis" in dfs:
            _save_plot_spectral_decay(dfs["spectral_rank_analysis"], output_dir / "fig_spectral_decay.png")
        if include_spectral and "spectral_partition_comparison" in dfs:
            _save_plot_partition_agreement(dfs["spectral_partition_comparison"], output_dir / "fig_partition_agreement.png")
        _save_plot_larger_scale(dfs["larger_scale"], output_dir / "fig_larger_scale_class_counts.png")
        _save_plot_scaling(dfs["larger_scale"], output_dir / "fig_scaling_behavior.png")
        _save_plot_rate_distortion(dfs["rate_distortion"], output_dir / "fig_rate_distortion.png")
        _save_plot_reward_planning(dfs["reward_planning"], output_dir / "fig_reward_planning.png")
        if "m2_medium_scale" in dfs:
            _save_plot_m1_vs_m2_comparison(dfs["m2_medium_scale"], output_dir / "fig_m1_vs_m2_medium_scale.png")
        if cfg.include_hierarchical_scaling and "hierarchical_t_scaling" in dfs:
            rt_df = pd.concat(
                [dfs["hierarchical_t_scaling"], dfs.get("principal_fsc_horizon_scaling", pd.DataFrame())],
                ignore_index=True,
            )
            _save_plot_runtime_vs_horizon_log(rt_df, output_dir / "fig_runtime_vs_horizon_log.png")
        if cfg.include_hierarchical_scaling and "layered_bound_validation" in dfs:
            _save_plot_layered_bound_validation(
                dfs["layered_bound_validation"], output_dir / "fig_layered_bound_vs_empirical.png"
            )

    return dfs
