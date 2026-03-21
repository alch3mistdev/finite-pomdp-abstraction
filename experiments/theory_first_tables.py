"""Theory-first exact table generation for the paper revision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .benchmarks import (
    discrete_observation_metric,
    gridworld_geometric_observation_metric,
    gridworld_pomdp,
    stationary_counterexample_pomdp,
    tiger_discrete_observation_metric,
    tiger_full_actions_pomdp,
)
from .exact_clock_aware import (
    ExactHistoryPartition,
    action_observation_step_rewards,
    build_future_distribution_cache,
    build_history_partition,
    build_prefix_history_tables,
    compute_belief_equivalence_classes,
    enumerate_clock_aware_open_loop_policies,
    enumerate_operational_open_loop_policies,
    evaluate_additive_observable_objective_original,
    evaluate_additive_observable_objective_quotient,
    evaluate_observable_objective_original,
    evaluate_observable_objective_quotient,
    family_distance_gap,
    latent_value_original,
    latent_value_quotient,
    observation_score_step_rewards,
    planning_summary_for_objective,
    total_adjusted_rand_index,
)
from .pomdp_core import FinitePOMDP


def _tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _short_action_name(name: str) -> str:
    parts = [part for part in name.replace("-", "_").split("_") if part]
    if not parts:
        return name[:2].upper()
    return "".join(part[0].upper() for part in parts)


def _policy_signature(pomdp: FinitePOMDP, actions: Tuple[int, ...]) -> str:
    if not actions:
        return "[]"
    return " ".join(_short_action_name(pomdp.action_names[action]) for action in actions)


def _goal_reward_gridworld(size: int = 3) -> FinitePOMDP:
    gw = gridworld_pomdp(size=size)
    goal_rewards = np.zeros_like(gw.rewards)
    goal_rewards[gw.num_states - 1, :] = 1.0
    return FinitePOMDP(
        state_names=gw.state_names,
        action_names=gw.action_names,
        observation_names=gw.observation_names,
        transition=gw.transition,
        observation=gw.observation,
        rewards=goal_rewards,
        initial_belief=gw.initial_belief,
    )


def _witness_observable_objective(
    actions: Tuple[int, ...],
    obs_seq: Tuple[int, ...],
) -> float:
    # Actions are A/B and observations are L/R/U/X/Y in that order.
    return 1.0 if len(actions) >= 3 and actions[1:] == (0, 1) and obs_seq[-1] == 3 else 0.0


def _benchmark_bundle(name: str, pomdp: FinitePOMDP, horizon: int, d_obs: np.ndarray) -> Dict[str, object]:
    registry, tables = build_prefix_history_tables(pomdp=pomdp, horizon=horizon)
    belief_classes = compute_belief_equivalence_classes(pomdp=pomdp, registry=registry, tables=tables)
    q_clk = build_history_partition(pomdp=pomdp, tables=tables, belief_classes=belief_classes, family_name="clk")
    q_op = build_history_partition(pomdp=pomdp, tables=tables, belief_classes=belief_classes, family_name="op")
    future_distribution = build_future_distribution_cache(pomdp, registry, horizon)
    return {
        "name": name,
        "pomdp": pomdp,
        "horizon": horizon,
        "d_obs": d_obs,
        "registry": registry,
        "tables": tables,
        "belief_classes": belief_classes,
        "q_clk": q_clk,
        "q_op": q_op,
        "future_distribution": future_distribution,
        "clk_policies": enumerate_clock_aware_open_loop_policies(pomdp.num_actions, horizon),
        "op_policies": enumerate_operational_open_loop_policies(pomdp.num_actions, horizon),
    }


def build_exact_bundles(
    tiger_horizons: Sequence[int] = (2, 4, 6, 8, 10),
    gridworld_horizons: Sequence[int] = (2, 3),
    include_witness: bool = True,
) -> List[Dict[str, object]]:
    bundles: List[Dict[str, object]] = []
    for horizon in tiger_horizons:
        bundles.append(_benchmark_bundle("Tiger", tiger_full_actions_pomdp(), horizon, tiger_discrete_observation_metric()))
    for horizon in gridworld_horizons:
        bundles.append(
            _benchmark_bundle("GridWorld 3x3", gridworld_pomdp(size=3), horizon, gridworld_geometric_observation_metric())
        )
    if include_witness:
        bundles.append(
            _benchmark_bundle(
                "Stationary witness",
                stationary_counterexample_pomdp(),
                3,
                discrete_observation_metric(5),
            )
        )
    return bundles


def run_probe_family_comparison(
    tiger_horizons: Sequence[int] = (2, 4, 6, 8, 10),
    gridworld_horizons: Sequence[int] = (2, 3),
    include_witness: bool = True,
) -> pd.DataFrame:
    bundles = build_exact_bundles(
        tiger_horizons=tiger_horizons,
        gridworld_horizons=gridworld_horizons,
        include_witness=include_witness,
    )
    return run_probe_family_comparison_from_bundles(bundles)


def run_probe_family_comparison_from_bundles(bundles: Sequence[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for bundle in bundles:
        pomdp = bundle["pomdp"]
        registry = bundle["registry"]
        tables = bundle["tables"]
        q_clk = bundle["q_clk"]
        q_op = bundle["q_op"]
        horizon = int(bundle["horizon"])
        d_obs = bundle["d_obs"]

        if bundle["name"] == "Stationary witness":
            objective = _witness_observable_objective
            planning = planning_summary_for_objective(
                pomdp=pomdp,
                registry=registry,
                tables=tables,
                partition=q_op,
                policies=bundle["clk_policies"],
                evaluate_original=lambda policy, fd=bundle["future_distribution"], obj=objective, init=int(bundle["tables"][0][0, 0]): evaluate_observable_objective_original(
                    policy, fd, obj, init
                ),
                evaluate_quotient=lambda policy, p=pomdp, r=registry, t=tables, q=q_op, obj=objective: evaluate_observable_objective_quotient(
                    p, r, t, q, policy, obj
                ),
            )
        else:
            step_rewards = observation_score_step_rewards(pomdp)
            planning = planning_summary_for_objective(
                pomdp=pomdp,
                registry=registry,
                tables=tables,
                partition=q_op,
                policies=bundle["clk_policies"],
                evaluate_original=lambda policy, p=pomdp, r=registry, init=int(bundle["tables"][0][0, 0]), sr=step_rewards: evaluate_additive_observable_objective_original(
                    p, r, policy, init, sr
                ),
                evaluate_quotient=lambda policy, p=pomdp, r=registry, t=tables, q=q_op, sr=step_rewards: evaluate_additive_observable_objective_quotient(
                    p, r, t, q, policy, sr
                ),
            )

        max_gap = 0.0
        for depth in range(horizon + 1):
            clk_suffixes = tuple(
                policy.actions[depth:] for policy in enumerate_clock_aware_open_loop_policies(pomdp.num_actions, horizon - depth)
            )
            op_suffixes = tuple(
                policy.actions for policy in enumerate_operational_open_loop_policies(pomdp.num_actions, horizon - depth)
            )
            max_gap = max(
                max_gap,
                family_distance_gap(
                    pomdp=pomdp,
                    registry=registry,
                    tables=tables,
                    depth=depth,
                    clock_policy_suffixes=clk_suffixes,
                    op_policy_suffixes=op_suffixes,
                    d_obs=d_obs,
                ),
            )

        rows.append(
            {
                "benchmark": bundle["name"],
                "horizon": horizon,
                "op_classes": q_op.num_classes_total,
                "clk_classes": q_clk.num_classes_total,
                "ari": total_adjusted_rand_index(q_op, q_clk),
                "max_gap": max_gap,
                "obs_value_changed_under_q_op": bool(planning["regret"] > 1e-9),
            }
        )
    return pd.DataFrame(rows)


def run_m2_probe_family_comparison(
    tiger_horizon: int = 2,
) -> pd.DataFrame:
    """Compare clock-aware vs operational probe families at m=2 using general W1.

    Uses the general precompute_distance_cache pipeline with unrolled FSCs,
    since the m=1 belief-table approach doesn't extend to m>=2.
    """
    from .fsc_enum import enumerate_clock_aware_deterministic_fscs, enumerate_deterministic_fscs
    from .quotient import compute_partition_from_cache, precompute_distance_cache

    pomdp = tiger_full_actions_pomdp()
    d_obs = tiger_discrete_observation_metric()
    horizon = tiger_horizon
    m = 2

    clk_policies = enumerate_clock_aware_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        m=m,
        horizon=horizon,
    )
    op_policies = enumerate_deterministic_fscs(
        num_actions=pomdp.num_actions,
        num_observations=pomdp.num_observations,
        max_nodes=m,
        include_smaller=True,
    )

    clk_cache = precompute_distance_cache(
        pomdp=pomdp, policies=clk_policies, horizon=horizon,
        distance_mode="w1", d_obs=d_obs,
    )
    op_cache = precompute_distance_cache(
        pomdp=pomdp, policies=op_policies, horizon=horizon,
        distance_mode="w1", d_obs=d_obs,
    )

    clk_part = compute_partition_from_cache(clk_cache, epsilon=0.0)
    op_part = compute_partition_from_cache(op_cache, epsilon=0.0)

    # Compute ARI between the two partitions
    from .spectral import partition_agreement
    agreement = partition_agreement(clk_part, op_part)

    # Compute max gap: largest absolute difference in pairwise distances
    max_gap = 0.0
    for depth in range(horizon + 1):
        clk_mat = clk_cache.max_distance_matrices[depth]
        op_mat = op_cache.max_distance_matrices[depth]
        max_gap = max(max_gap, float(np.abs(clk_mat - op_mat).max()))

    return pd.DataFrame([{
        "benchmark": "Tiger",
        "horizon": horizon,
        "m": m,
        "op_classes": op_part.num_classes_total,
        "clk_classes": clk_part.num_classes_total,
        "num_clk_policies": len(clk_policies),
        "num_op_policies": len(op_policies),
        "ari": float(agreement["adjusted_rand_index"]),
        "max_gap": max_gap,
    }])


def run_exact_observation_planning(
    tiger_horizons: Sequence[int] = (2, 4, 6, 8, 10),
    gridworld_horizons: Sequence[int] = (2, 3),
) -> pd.DataFrame:
    bundles = build_exact_bundles(tiger_horizons=tiger_horizons, gridworld_horizons=gridworld_horizons, include_witness=False)
    return run_exact_observation_planning_from_bundles(bundles)


def run_exact_observation_planning_from_bundles(bundles: Sequence[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for bundle in bundles:
        name = str(bundle["name"])
        pomdp = bundle["pomdp"]
        horizon = int(bundle["horizon"])
        objectives = [
            ("Obs score", observation_score_step_rewards(pomdp)),
            ("Action+obs score", action_observation_step_rewards(pomdp)),
        ]
        for objective_name, step_rewards in objectives:
            summary = planning_summary_for_objective(
                pomdp=pomdp,
                registry=bundle["registry"],
                tables=bundle["tables"],
                partition=bundle["q_clk"],
                policies=bundle["clk_policies"],
                evaluate_original=lambda policy, p=pomdp, r=bundle["registry"], init=int(bundle["tables"][0][0, 0]), sr=step_rewards: evaluate_additive_observable_objective_original(
                    p, r, policy, init, sr
                ),
                evaluate_quotient=lambda policy, p=pomdp, r=bundle["registry"], t=bundle["tables"], q=bundle["q_clk"], sr=step_rewards: evaluate_additive_observable_objective_quotient(
                    p, r, t, q, policy, sr
                ),
            )
            rows.append(
                {
                    "benchmark": name,
                    "horizon": horizon,
                    "objective": objective_name,
                    "histories": bundle["q_clk"].total_histories,
                    "classes": bundle["q_clk"].num_classes_total,
                    "time_original_s": summary["time_original_s"],
                    "time_quotient_s": summary["time_quotient_s"],
                    "policy_original": _policy_signature(pomdp, summary["best_original_policy"].actions),
                    "policy_quotient": _policy_signature(pomdp, summary["best_quotient_policy"].actions),
                    "value_original_best": summary["value_original_best"],
                    "value_quotient_best_on_original": summary["value_quotient_best_on_original"],
                    "regret": summary["regret"],
                }
            )
    return pd.DataFrame(rows)


def run_exact_latent_planning(
    tiger_horizons: Sequence[int] = (2, 4, 6, 8, 10),
    gridworld_horizons: Sequence[int] = (2, 3),
) -> pd.DataFrame:
    bundles = [
        _benchmark_bundle("Tiger", tiger_full_actions_pomdp(), horizon, tiger_discrete_observation_metric())
        for horizon in tiger_horizons
    ] + [
        _benchmark_bundle("GridWorld 3x3", _goal_reward_gridworld(size=3), horizon, gridworld_geometric_observation_metric())
        for horizon in gridworld_horizons
    ]
    return run_exact_latent_planning_from_bundles(bundles)


def run_exact_latent_planning_from_bundles(bundles: Sequence[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for bundle in bundles:
        name = str(bundle["name"])
        pomdp = bundle["pomdp"]
        horizon = int(bundle["horizon"])
        reward_name = "Tiger reward" if name == "Tiger" else "Goal reward"
        summary = planning_summary_for_objective(
            pomdp=pomdp,
            registry=bundle["registry"],
            tables=bundle["tables"],
            partition=bundle["q_clk"],
            policies=bundle["clk_policies"],
            evaluate_original=lambda policy, p=pomdp: latent_value_original(p, policy),
            evaluate_quotient=lambda policy, p=pomdp, r=bundle["registry"], t=bundle["tables"], q=bundle["q_clk"]: latent_value_quotient(
                p, r, t, q, policy
            ),
        )
        rows.append(
            {
                "benchmark": name,
                "horizon": horizon,
                "reward": reward_name,
                "histories": bundle["q_clk"].total_histories,
                "classes": bundle["q_clk"].num_classes_total,
                "time_original_s": summary["time_original_s"],
                "time_quotient_s": summary["time_quotient_s"],
                "policy_original": _policy_signature(pomdp, summary["best_original_policy"].actions),
                "policy_quotient": _policy_signature(pomdp, summary["best_quotient_policy"].actions),
                "value_original_best": summary["value_original_best"],
                "value_quotient_best_on_original": summary["value_quotient_best_on_original"],
                "regret": summary["regret"],
            }
        )
    return pd.DataFrame(rows)


def _bool_tex(value: object) -> str:
    return "Yes" if bool(value) else "No"


def render_probe_family_table(df: pd.DataFrame) -> str:
    has_m = "m" in df.columns
    if has_m:
        lines = [
            r"\begin{tabular}{lrlrrrrr}",
            r"\toprule",
            r"Benchmark & $T$ & $m$ & $|\Pi^{\mathrm{op}}|$ & $|Q^{\mathrm{op}}|$ & $|Q^{\mathrm{clk}}|$ & ARI & $\max_{h,h'} |d^{\mathrm{clk}}-d^{\mathrm{op}}|$ \\",
            r"\midrule",
        ]
    else:
        lines = [
            r"\begin{tabular}{llrrrrr}",
            r"\toprule",
            r"Benchmark & $T$ & $|Q^{\mathrm{op}}|$ & $|Q^{\mathrm{clk}}|$ & ARI & $\max_{h,h'} |d^{\mathrm{clk}}-d^{\mathrm{op}}|$ & Obs.\ value changed?\\",
            r"\midrule",
        ]
    for row in df.itertuples(index=False):
        if has_m:
            m_val = int(row.m) if hasattr(row, "m") else 1
            n_op = int(row.num_op_policies) if hasattr(row, "num_op_policies") else "---"
            lines.append(
                f"{_tex_escape(row.benchmark)} & {int(row.horizon)} & {m_val} & {n_op} & {int(row.op_classes)} & {int(row.clk_classes)} & "
                f"{float(row.ari):.3f} & {float(row.max_gap):.3f}\\\\"
            )
        else:
            lines.append(
                f"{_tex_escape(row.benchmark)} & {int(row.horizon)} & {int(row.op_classes)} & {int(row.clk_classes)} & "
                f"{float(row.ari):.3f} & {float(row.max_gap):.3f} & {_bool_tex(row.obs_value_changed_under_q_op)}\\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def render_observation_planning_table(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrp{3.2cm}rrr}",
        r"\toprule",
        r"Benchmark & Obj. & $T$ & Hist. & Cls. & $t_{\mathrm{orig}}$ & $t_{Q^{\mathrm{clk}}}$ & Policy & $V_{\mathrm{orig}}$ & $V_{Q^{\mathrm{clk}}}$ & Regret\\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        policy = _tex_escape(row.policy_quotient)
        lines.append(
            f"{_tex_escape(row.benchmark)} & {_tex_escape(row.objective)} & {int(row.horizon)} & {int(row.histories)} & {int(row.classes)} & "
            f"{float(row.time_original_s):.3f} & {float(row.time_quotient_s):.3f} & {policy} & "
            f"{float(row.value_original_best):.3f} & {float(row.value_quotient_best_on_original):.3f} & {float(row.regret):.3f}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def render_latent_planning_table(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrp{3.2cm}rrr}",
        r"\toprule",
        r"Benchmark & Reward & $T$ & Hist. & Cls. & $t_{\mathrm{orig}}$ & $t_{Q^{\mathrm{clk}}}$ & Policy & $V_{\mathrm{orig}}$ & $V_{Q^{\mathrm{clk}}\rightarrow M}$ & Regret\\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{_tex_escape(row.benchmark)} & {_tex_escape(row.reward)} & {int(row.horizon)} & {int(row.histories)} & {int(row.classes)} & "
            f"{float(row.time_original_s):.3f} & {float(row.time_quotient_s):.3f} & {_tex_escape(row.policy_quotient)} & "
            f"{float(row.value_original_best):.3f} & {float(row.value_quotient_best_on_original):.3f} & {float(row.regret):.3f}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def export_theory_first_tables(output_dir: Path, paper_generated_dir: Path) -> Dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_generated_dir.mkdir(parents=True, exist_ok=True)

    shared_bundles = build_exact_bundles()
    probe_df = run_probe_family_comparison_from_bundles(shared_bundles)

    # Add m=2 clock-aware comparison for Tiger T=2
    m2_df = run_m2_probe_family_comparison(tiger_horizon=2)

    obs_df = run_exact_observation_planning_from_bundles([b for b in shared_bundles if b["name"] != "Stationary witness"])
    latent_bundles = [
        _benchmark_bundle("Tiger", tiger_full_actions_pomdp(), int(b["horizon"]), tiger_discrete_observation_metric())
        for b in shared_bundles
        if b["name"] == "Tiger"
    ] + [
        _benchmark_bundle("GridWorld 3x3", _goal_reward_gridworld(size=3), int(b["horizon"]), gridworld_geometric_observation_metric())
        for b in shared_bundles
        if b["name"] == "GridWorld 3x3"
    ]
    latent_df = run_exact_latent_planning_from_bundles(latent_bundles)

    artifacts = {
        "probe_family_comparison": probe_df,
        "m2_probe_family_comparison": m2_df,
        "clock_aware_observation_planning": obs_df,
        "clock_aware_latent_planning": latent_df,
    }
    for name, df in artifacts.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
        (output_dir / f"{name}.json").write_text(df.to_json(orient="records", indent=2))

    (paper_generated_dir / "table_probe_family_comparison.tex").write_text(render_probe_family_table(probe_df))
    (paper_generated_dir / "table_m2_probe_family_comparison.tex").write_text(render_probe_family_table(m2_df))
    (paper_generated_dir / "table_observation_planning.tex").write_text(render_observation_planning_table(obs_df))
    (paper_generated_dir / "table_latent_planning.tex").write_text(render_latent_planning_table(latent_df))
    return artifacts
