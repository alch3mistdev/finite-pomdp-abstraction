from __future__ import annotations

import numpy as np

from experiments.benchmarks import (
    discrete_observation_metric,
    gridworld_geometric_observation_metric,
    gridworld_pomdp,
    stationary_counterexample_pomdp,
    tiger_discrete_observation_metric,
    tiger_full_actions_pomdp,
)
from experiments.exact_clock_aware import (
    build_future_distribution_cache,
    build_history_partition,
    build_prefix_history_tables,
    compute_belief_equivalence_classes,
    enumerate_clock_aware_open_loop_policies,
    evaluate_observable_objective_original,
    evaluate_observable_objective_quotient,
    planning_summary_for_objective,
    observation_objective_score,
)
from experiments.theory_first_tables import (
    render_latent_planning_table,
    render_observation_planning_table,
    render_probe_family_table,
    run_exact_latent_planning,
    run_exact_observation_planning,
    run_probe_family_comparison,
)


def _partitions_for(pomdp, horizon: int):
    registry, tables = build_prefix_history_tables(pomdp=pomdp, horizon=horizon)
    belief_classes = compute_belief_equivalence_classes(pomdp=pomdp, registry=registry, tables=tables)
    q_clk = build_history_partition(pomdp=pomdp, tables=tables, belief_classes=belief_classes, family_name="clk")
    q_op = build_history_partition(pomdp=pomdp, tables=tables, belief_classes=belief_classes, family_name="op")
    return registry, tables, q_clk, q_op


def test_probe_family_coarsening_on_witness() -> None:
    pomdp = stationary_counterexample_pomdp()
    _, _, q_clk, q_op = _partitions_for(pomdp, horizon=3)
    assert q_clk.num_classes_total > q_op.num_classes_total


def test_probe_family_comparison_small_rows() -> None:
    df = run_probe_family_comparison(tiger_horizons=[2], gridworld_horizons=[2], include_witness=True)
    assert set(df["benchmark"]) == {"Tiger", "GridWorld 3x3", "Stationary witness"}
    assert (df["clk_classes"] >= df["op_classes"]).all()
    witness = df[df["benchmark"] == "Stationary witness"].iloc[0]
    assert int(witness["clk_classes"]) > int(witness["op_classes"])
    tiger = df[df["benchmark"] == "Tiger"].iloc[0]
    assert int(tiger["clk_classes"]) == int(tiger["op_classes"])


def test_observation_accessible_exact_preservation_on_witness() -> None:
    pomdp = stationary_counterexample_pomdp()
    registry, tables, q_clk, _ = _partitions_for(pomdp, horizon=3)
    future_distribution = build_future_distribution_cache(pomdp, registry, 3)
    objective = lambda actions, obs_seq: 1.0 if len(actions) >= 3 and actions[1:] == (0, 1) and obs_seq[-1] == 3 else 0.0
    policies = enumerate_clock_aware_open_loop_policies(pomdp.num_actions, 3)
    summary = planning_summary_for_objective(
        pomdp=pomdp,
        registry=registry,
        tables=tables,
        partition=q_clk,
        policies=policies,
        evaluate_original=lambda policy, fd=future_distribution, obj=objective, init=int(tables[0][0, 0]): evaluate_observable_objective_original(
            policy, fd, obj, init
        ),
        evaluate_quotient=lambda policy, p=pomdp, r=registry, t=tables, q=q_clk, obj=objective: evaluate_observable_objective_quotient(
            p, r, t, q, policy, obj
        ),
    )
    assert summary["regret"] == pytest_approx(0.0, 1e-12)


def test_exact_observation_planning_small_cases_have_zero_regret() -> None:
    df = run_exact_observation_planning(tiger_horizons=[2], gridworld_horizons=[2])
    assert (np.abs(df["regret"].to_numpy(dtype=float)) <= 1e-9).all()


def test_exact_latent_planning_small_cases_are_well_formed() -> None:
    df = run_exact_latent_planning(tiger_horizons=[2], gridworld_horizons=[2])
    assert {"Tiger", "GridWorld 3x3"} == set(df["benchmark"].unique())
    assert (df["regret"] >= -1e-9).all()


def test_required_small_benchmarks_and_horizons_appear() -> None:
    probe = run_probe_family_comparison(tiger_horizons=[2, 4], gridworld_horizons=[2, 3], include_witness=True)
    obs = run_exact_observation_planning(tiger_horizons=[2, 4], gridworld_horizons=[2, 3])
    latent = run_exact_latent_planning(tiger_horizons=[2, 4], gridworld_horizons=[2, 3])

    assert {(row["benchmark"], int(row["horizon"])) for _, row in probe.iterrows()} == {
        ("Tiger", 2),
        ("Tiger", 4),
        ("GridWorld 3x3", 2),
        ("GridWorld 3x3", 3),
        ("Stationary witness", 3),
    }
    assert {(row["benchmark"], int(row["horizon"])) for _, row in obs.iterrows()} == {
        ("Tiger", 2),
        ("Tiger", 4),
        ("GridWorld 3x3", 2),
        ("GridWorld 3x3", 3),
    }
    assert {(row["benchmark"], int(row["horizon"])) for _, row in latent.iterrows()} == {
        ("Tiger", 2),
        ("Tiger", 4),
        ("GridWorld 3x3", 2),
        ("GridWorld 3x3", 3),
    }


def test_renderers_include_expected_labels() -> None:
    probe = run_probe_family_comparison(tiger_horizons=[2], gridworld_horizons=[], include_witness=True)
    obs = run_exact_observation_planning(tiger_horizons=[2], gridworld_horizons=[])
    latent = run_exact_latent_planning(tiger_horizons=[2], gridworld_horizons=[])
    assert "Stationary witness" in render_probe_family_table(probe)
    assert "Obs score" in render_observation_planning_table(obs)
    assert "Tiger reward" in render_latent_planning_table(latent)


def pytest_approx(value: float, tol: float):
    return pytest.approx(value, abs=tol)


import pytest  # noqa: E402
