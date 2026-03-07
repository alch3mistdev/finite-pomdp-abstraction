#!/usr/bin/env python3
"""CLI entrypoint for the enhanced basic experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

# Support direct script execution via absolute path.
if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.analysis import run_all_experiments


def _parse_csv_floats(raw: str | None) -> List[float]:
    if raw is None or raw.strip() == "":
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_ints(raw: str | None) -> List[int]:
    if raw is None or raw.strip() == "":
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run enhanced finite-POMDP basic experiments")
    parser.add_argument(
        "--profile",
        choices=["quick", "extended"],
        default="quick",
        help="Preset runtime profile (default: quick)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results" / "basic"),
        help="Directory for CSV and PNG artifacts",
    )
    parser.add_argument(
        "--eps-grid",
        default=None,
        help='Optional CSV list of epsilon values, e.g. "0,0.1,0.2"',
    )
    parser.add_argument(
        "--ms",
        default="1,2",
        help='CSV list of memory bounds for capacity sweep, e.g. "1,2"',
    )
    parser.add_argument(
        "--horizons",
        default="2,3,4",
        help='CSV list of horizons for Tiger horizon-gap experiment, e.g. "2,3,4"',
    )
    parser.add_argument(
        "--include-nonlipschitz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable supplementary non-Lipschitz Tiger reward track (default: true)",
    )
    parser.add_argument(
        "--stochastic-samples",
        type=int,
        default=None,
        help="Override stochastic FSC sample count for sanity experiment",
    )
    parser.add_argument(
        "--include-spectral",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable spectral approximation experiments (default: true)",
    )
    parser.add_argument(
        "--include-hierarchical-scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable long-horizon hierarchical scaling and principal-FSC tracks (default: false)",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=10,
        help="Maximum horizon used by hierarchical scaling tracks (default: 10)",
    )
    parser.add_argument(
        "--segment-horizon",
        type=int,
        default=None,
        help="Optional fixed segment horizon tau for layered runs (default: auto 4/5)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG figure generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed (default: 7)",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run independent experiment tracks in parallel (default: true)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional fixed worker count for parallel execution",
    )
    parser.add_argument(
        "--worker-fraction",
        type=float,
        default=0.5,
        help="Auto worker fraction of logical cores when --workers is not set (default: 0.5)",
    )

    args = parser.parse_args()
    if args.worker_fraction <= 0:
        parser.error("--worker-fraction must be > 0")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be >= 1")

    output_dir = Path(args.output_dir).resolve()

    run_all_experiments(
        output_dir=output_dir,
        profile=args.profile,
        eps_grid=_parse_csv_floats(args.eps_grid),
        ms=_parse_csv_ints(args.ms),
        horizons=_parse_csv_ints(args.horizons),
        include_nonlipschitz=bool(args.include_nonlipschitz),
        include_spectral=bool(args.include_spectral),
        include_hierarchical_scaling=bool(args.include_hierarchical_scaling),
        stochastic_samples=args.stochastic_samples,
        max_horizon=int(args.max_horizon),
        segment_horizon=(None if args.segment_horizon is None else int(args.segment_horizon)),
        skip_plots=bool(args.skip_plots),
        seed=int(args.seed),
        parallel=bool(args.parallel),
        workers=args.workers,
        worker_fraction=float(args.worker_fraction),
    )

    print(f"Wrote experiment artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
