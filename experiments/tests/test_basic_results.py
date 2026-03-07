from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analysis import (
    run_all_experiments,
    run_baseline_comparison,
    run_capacity_sweep_tiger,
    run_clustering_optimality_check,
    run_data_processing_experiment,
    run_gridworld_metric_sensitivity,
    run_hyperparameter_sensitivity,
    run_initial_belief_sensitivity,
    run_lipschitz_value_bounds,
    run_multi_seed_witness,
    run_new_benchmark_experiments,
    run_nonlipschitz_value_track,
    run_observation_noise_sensitivity,
    run_observation_sensitivity_experiment,
    run_planning_speedup_experiment,
    tiger_reproduction_sanity,
)


def test_tiger_reproduction_sanity() -> None:
    df = tiger_reproduction_sanity()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["w1_L_vs_R"] == pytest_approx(0.49, 1e-6)
    assert int(row["paper_exact_class_count"]) == 7
    assert int(row["paper_eps_0_5_class_count"]) == 3


def test_capacity_and_epsilon_monotonicity() -> None:
    eps = [0.0, 0.2, 0.4, 0.6]
    df = run_capacity_sweep_tiger(eps_grid=eps, ms=[1, 2])

    for e in eps:
        sub = df[df["epsilon"] == e].sort_values("m")
        c1 = float(sub[sub["m"] == 1.0]["class_count"].iloc[0])
        c2 = float(sub[sub["m"] == 2.0]["class_count"].iloc[0])
        assert c2 >= c1

    for m in [1.0, 2.0]:
        sub = df[df["m"] == m].sort_values("epsilon")
        vals = sub["class_count"].to_numpy(dtype=float)
        assert np.all(vals[:-1] >= vals[1:])


def test_dual_bound_columns_and_formula() -> None:
    eps = [0.0, 0.2, 0.5]
    df = run_lipschitz_value_bounds(eps_grid=eps)
    required = {
        "empirical_value_error",
        "theorem_4_4_style_bound",
        "canonical_quotient_bound",
        "L_R",
        "horizon",
        "epsilon",
        "reward_track",
        "reward_range",
        "theorem_bound_vacuous",
        "canonical_bound_vacuous",
    }
    assert required.issubset(set(df.columns))

    # Synthetic track: L_R=1
    syn = df[df["reward_track"] == "synthetic_lipschitz"]
    assert len(syn) == len(eps)
    assert (syn["L_R"] == 1.0).all()
    assert (syn["bound_applicable"] == True).all()

    for _, row in syn.iterrows():
        assert row["empirical_value_error"] >= 0.0
        assert row["theorem_4_4_style_bound"] >= 0.0
        assert row["canonical_quotient_bound"] >= 0.0

        expected = (
            float(row["L_R"])
            * float(row["horizon"])
            * float(row["epsilon"])
            * (1.0 + 2.0 * float(row["horizon"]) * 2.0 * 2.0)
        )
        assert row["canonical_quotient_bound"] == pytest_approx(expected, 1e-12)

    # Standard reward track: L_R=110
    std = df[df["reward_track"] == "standard_reward"]
    assert len(std) == len(eps)
    assert (std["L_R"] == 110.0).all()
    # Canonical bound is vacuous at any eps > 0 with L_R=110
    std_nonzero = std[std["epsilon"] > 0]
    if len(std_nonzero) > 0:
        assert std_nonzero["canonical_bound_vacuous"].all()


def test_nonlipschitz_track_presence() -> None:
    df = run_nonlipschitz_value_track(eps_grid=[0.0, 0.5])
    assert "empirical_value_error" in df.columns
    assert "bound_applicable" in df.columns
    assert not df["bound_applicable"].any()
    assert "L_R_standard" in df.columns
    assert (df["L_R_standard"] == 110.0).all()
    assert "bound_vacuous" in df.columns
    assert "hypothetical_bound_l_r_110" in df.columns


def test_w1_vs_tv_sensitivity() -> None:
    eps = [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    df = run_gridworld_metric_sensitivity(eps_grid=eps)
    assert (df["delta_tv_minus_w1"].abs() > 0).any()


def test_observation_noise_sensitivity() -> None:
    df = run_observation_noise_sensitivity(eps_grid=[0.0, 0.3, 0.5], accuracies=[0.75, 0.85, 0.95])
    assert "accuracy" in df.columns
    assert "class_count" in df.columns
    assert "max_value_error" in df.columns
    assert len(df) == 9  # 3 accuracies * 3 epsilons
    assert (df["max_value_error"] >= 0).all()


def test_multi_seed_witness() -> None:
    df = run_multi_seed_witness(seeds=[7, 42], stochastic_samples=50)
    assert len(df) == 2
    assert "max_distance_deterministic" in df.columns
    assert "max_distance_stochastic" in df.columns
    # No stochastic should exceed deterministic
    assert (df["stochastic_exceeds_det"] == 0.0).all()


def test_baseline_comparison() -> None:
    df = run_baseline_comparison(eps_grid=[0.0, 0.3, 0.5])
    methods = set(df["method"].unique())
    assert {"epsilon_quotient", "truncation_d1", "random_partition", "belief_distance", "bisimulation"} == methods
    assert (df["max_value_error"] >= 0).all()
    assert (df["class_count"] >= 1).all()
    assert "matches_epsilon_quotient" in df.columns
    # epsilon_quotient itself should never match itself
    eq_rows = df[df["method"] == "epsilon_quotient"]
    assert not eq_rows["matches_epsilon_quotient"].any()


def test_hyperparameter_sensitivity() -> None:
    df = run_hyperparameter_sensitivity(eps_grid=[0.0, 0.3], ms=[1], horizons=[2])
    assert len(df) == 2  # 1 m * 1 horizon * 2 eps
    assert "class_count" in df.columns


def test_clustering_optimality() -> None:
    from experiments.clustering import cluster_complete_linkage, cluster_optimal, clustering_optimality_gap

    # Simple test: 4 items, two natural pairs
    d = np.array([
        [0.0, 0.1, 0.9, 0.9],
        [0.1, 0.0, 0.9, 0.9],
        [0.9, 0.9, 0.0, 0.1],
        [0.9, 0.9, 0.1, 0.0],
    ])
    greedy = cluster_complete_linkage(d, 0.5)
    optimal = cluster_optimal(d, 0.5)
    assert len(greedy) == len(optimal) == 2

    gap = clustering_optimality_gap(d, 0.5)
    assert gap["gap"] == 0
    assert gap["greedy_is_optimal"]


def test_clustering_optimality_check_experiment() -> None:
    df = run_clustering_optimality_check(eps_grid=[0.0, 0.3, 0.5])
    assert "greedy_clusters" in df.columns
    assert "optimal_clusters" in df.columns
    assert "gap" in df.columns
    assert "greedy_is_optimal" in df.columns
    # Greedy should never produce fewer clusters than optimal
    assert (df["gap"] >= 0).all()


def test_data_processing_stochastic_channel() -> None:
    df = run_data_processing_experiment(eps_grid=[0.0, 0.3])
    assert "channel_type" in df.columns
    stoch_rows = df[df["channel_type"] == "stochastic"]
    assert len(stoch_rows) > 0
    assert stoch_rows["monotonicity_holds"].all()
    det_rows = df[df["channel_type"] == "deterministic"]
    assert len(det_rows) > 0
    assert det_rows["monotonicity_holds"].all()


def test_observation_sensitivity_monotonicity() -> None:
    df = run_observation_sensitivity_experiment(
        eps_grid=[0.0, 0.3],
        delta_grid=[0.0, 0.5, 1.0],
    )
    assert "delta_o" in df.columns
    assert "monotonicity_holds" in df.columns
    # Distance monotonicity must hold for all rows
    assert df["monotonicity_holds"].all()
    # At delta=0, coarsened should equal original
    zero_rows = df[df["delta_o"] == 0.0]
    for _, row in zero_rows.iterrows():
        assert row["classes_full"] == row["classes_coarsened"]
    # Larger delta should give fewer or equal classes
    for bench in df["benchmark"].unique():
        for eps in df["epsilon"].unique():
            sub = df[(df["benchmark"] == bench) & (df["epsilon"] == eps)].sort_values("delta_o")
            classes = sub["classes_coarsened"].to_numpy(dtype=float)
            assert np.all(classes[:-1] >= classes[1:]), (
                f"Non-monotonic class count for {bench} at eps={eps}"
            )


def test_initial_belief_sensitivity() -> None:
    df = run_initial_belief_sensitivity(
        eps_grid=[0.0, 0.3, 0.5],
        beliefs=[[0.5, 0.5], [0.3, 0.7], [0.7, 0.3]],
    )
    assert "b0_left" in df.columns
    assert "adjusted_rand_index" in df.columns
    assert len(df) == 9  # 3 beliefs * 3 epsilons
    # Uniform belief should have ARI=1.0 with itself
    uniform = df[df["b0_left"] == 0.5]
    assert (uniform["adjusted_rand_index"] == 1.0).all()


def test_new_benchmark_experiments() -> None:
    df = run_new_benchmark_experiments(eps_grid=[0.0, 0.3], ms=[1])
    assert "benchmark" in df.columns
    benchmarks = set(df["benchmark"].unique())
    assert "hallway_10" in benchmarks
    assert "network_4" in benchmarks
    assert "class_count" in df.columns
    assert "compression_ratio" in df.columns
    assert (df["class_count"] >= 1).all()


def test_planning_speedup() -> None:
    df = run_planning_speedup_experiment(eps_grid=[0.0, 0.3], ms=[1])
    assert "benchmark" in df.columns
    assert "time_original_s" in df.columns
    assert "time_quotient_s" in df.columns
    assert "value_gap" in df.columns
    assert (df["value_gap"] >= -1e-9).all()  # quotient-best should not exceed original-best


def test_end_to_end_smoke(tmp_path: Path) -> None:
    out = tmp_path / "basic"
    run_all_experiments(
        output_dir=out,
        profile="quick",
        eps_grid=[0.0, 0.2, 0.4],
        ms=[1, 2],
        horizons=[2, 3],
        include_nonlipschitz=True,
        stochastic_samples=50,
        skip_plots=False,
        seed=11,
    )

    expected = [
        "capacity_sweep_tiger.csv",
        "value_loss_bounds_lipschitz.csv",
        "value_loss_nonlipschitz_tiger.csv",
        "horizon_gap_tiger.csv",
        "metric_sensitivity_gridworld.csv",
        "stochastic_vs_deterministic_sanity.csv",
        "observation_noise_sensitivity.csv",
        "multi_seed_witness.csv",
        "baseline_comparison.csv",
        "observation_sensitivity.csv",
        "ablation_studies.csv",
        "hyperparameter_sensitivity.csv",
        "clustering_optimality.csv",
        "initial_belief_sensitivity.csv",
        "new_benchmarks.csv",
        "planning_speedup.csv",
        "timing_summary.csv",
        "fig_capacity_vs_m.png",
        "fig_value_loss_with_two_bounds.png",
        "fig_gap_vs_horizon.png",
        "fig_w1_vs_tv_gridworld.png",
        "fig_noise_sensitivity.png",
        "fig_baseline_comparison.png",
        "fig_hyperparameter_heatmap.png",
    ]

    for name in expected:
        path = out / name
        assert path.exists(), f"Missing artifact: {name}"
        assert path.stat().st_size > 0, f"Empty artifact: {name}"


def pytest_approx(value: float, tol: float):
    return pytest.approx(value, abs=tol)


import pytest  # noqa: E402  # keep import local to avoid optional dependency issues at import time
