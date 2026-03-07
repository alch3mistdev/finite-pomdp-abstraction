"""Tests for hierarchical scalability utilities."""

from __future__ import annotations

import numpy as np

from experiments.analysis import run_principal_fsc_horizon_scaling
from experiments.hierarchical import (
    compose_layered_distortion,
    run_hierarchical_t_scaling,
    run_layered_bound_validation,
)


def test_compose_layered_distortion_single_layer() -> None:
    eps = 0.123
    out = compose_layered_distortion([eps], [0.7])
    assert abs(out - eps) < 1e-12


def test_compose_layered_distortion_multi_layer() -> None:
    # Formula: e1*L2*L3 + e2*L3 + e3
    e1, e2, e3 = 0.2, 0.1, 0.05
    L1, L2, L3 = 0.9, 0.8, 0.6
    expected = e1 * L2 * L3 + e2 * L3 + e3
    out = compose_layered_distortion([e1, e2, e3], [L1, L2, L3])
    assert abs(out - expected) < 1e-12


def test_run_hierarchical_t_scaling_small() -> None:
    df = run_hierarchical_t_scaling(horizons=(4,), epsilon=0.5, m=1, segment_horizon=4)
    assert len(df) == 2  # direct + layered
    assert set(df["method"].unique()) == {"direct", "layered"}
    assert np.isclose(df[df["method"] == "direct"]["horizon"].iloc[0], 4.0)


def test_run_layered_bound_validation_checks_hold() -> None:
    df = run_layered_bound_validation(
        eps_grid=(0.2, 0.5),
        long_horizons=(4,),
        epsilon=0.5,
    )
    assert len(df) > 0
    assert bool(df["bound_holds"].all())
    expected_checks = {
        "single_layer_reduction",
        "wrapper_stack_deterministic",
        "wrapper_stack_stochastic",
        "wrapper_stack_composed",
        "small_horizon_equivalence",
        "long_horizon_theorem_bound",
        "composed_distortion_accounting",
    }
    assert expected_checks.issubset(set(df["check"].unique()))


def test_principal_fsc_horizon_scaling_smoke() -> None:
    df = run_principal_fsc_horizon_scaling(
        epsilon=0.5,
        calibration_horizon=4,
        horizons_full=(4,),
        horizons_subset=(4,),
        ari_target=0.95,
        default_k=5,
        max_k_search=8,
    )
    assert len(df) >= 3
    assert {"calibration", "full_exact", "principal_subset"}.issubset(set(df["method"].unique()))
    subset = df[df["method"] == "principal_subset"].iloc[0]
    assert subset["policy_count"] <= 147
    assert subset["runtime_s"] > 0.0
