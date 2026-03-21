#!/usr/bin/env python3
"""Validate archived Tier III artifact files against the reported paper rows."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_csv(name: str) -> list[dict[str, str]]:
    path = ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_rows(name: str, actual: list[dict[str, str]], expected: list[dict[str, str]]) -> None:
    if actual != expected:
        raise AssertionError(
            f"{name} contents do not match the archived paper rows.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


def main() -> int:
    required_files = [
        "README.md",
        "meaningful_scale.csv",
        "m2_medium_scale.csv",
        "computational_profile.csv",
        "bootstrap_coverage.csv",
        "timing_summary.csv",
    ]
    missing = [name for name in required_files if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Tier III files: {', '.join(missing)}")

    meaningful = read_csv("meaningful_scale.csv")
    expected_meaningful = [
        {
            "track": "Scale-up",
            "benchmark": "Net. Mon. (n=12)",
            "state_count": "4096",
            "observation_count": "3",
            "horizon": "2",
            "method": "Sampling",
            "epsilon_0_classes": "13",
            "epsilon_03_classes": "5",
            "runtime_seconds_reported": "~5",
            "ci_lo": "0.31",
            "ci_hi": "0.43",
        },
        {
            "track": "Large O",
            "benchmark": "Random",
            "state_count": "2000",
            "observation_count": "10",
            "horizon": "2",
            "method": "Sampling",
            "epsilon_0_classes": "111",
            "epsilon_03_classes": "10",
            "runtime_seconds_reported": "~30",
            "ci_lo": "0.10",
            "ci_hi": "0.20",
        },
        {
            "track": "Long T",
            "benchmark": "Net. Mon. (n=10)",
            "state_count": "1024",
            "observation_count": "3",
            "horizon": "20",
            "method": "Layered",
            "epsilon_0_classes": "13",
            "epsilon_03_classes": "5",
            "runtime_seconds_reported": "~25",
            "ci_lo": "0.67",
            "ci_hi": "0.84",
        },
    ]
    assert_rows("meaningful_scale.csv", meaningful, expected_meaningful)

    m2_scaling = read_csv("m2_medium_scale.csv")
    expected_m2_scaling = [
        {
            "benchmark": "RockSample(4,4)",
            "state_count": "257",
            "m": "1",
            "horizon": "2",
            "fscs": "9",
            "epsilon_0_classes": "5",
            "epsilon_01_classes": "5",
            "epsilon_03_classes": "4",
            "epsilon_05_classes": "3",
            "runtime_seconds_reported": "0.5",
            "ci_lo": "0.35",
            "ci_hi": "0.47",
        },
        {
            "benchmark": "RockSample(4,4)",
            "state_count": "257",
            "m": "2",
            "horizon": "2",
            "fscs": "5193",
            "epsilon_0_classes": "5",
            "epsilon_01_classes": "5",
            "epsilon_03_classes": "5",
            "epsilon_05_classes": "5",
            "runtime_seconds_reported": "321",
            "ci_lo": "1.00",
            "ci_hi": "1.00",
        },
        {
            "benchmark": "Net. Mon. (n=4)",
            "state_count": "16",
            "m": "1",
            "horizon": "3",
            "fscs": "5",
            "epsilon_0_classes": "14",
            "epsilon_01_classes": "9",
            "epsilon_03_classes": "6",
            "epsilon_05_classes": "5",
            "runtime_seconds_reported": "1.1",
            "ci_lo": "0.48",
            "ci_hi": "0.66",
        },
        {
            "benchmark": "Net. Mon. (n=4)",
            "state_count": "16",
            "m": "2",
            "horizon": "3",
            "fscs": "1605",
            "epsilon_0_classes": "14",
            "epsilon_01_classes": "14",
            "epsilon_03_classes": "14",
            "epsilon_05_classes": "6",
            "runtime_seconds_reported": "353",
            "ci_lo": "0.67",
            "ci_hi": "0.84",
        },
        {
            "benchmark": "Net. Mon. (n=9)",
            "state_count": "512",
            "m": "1",
            "horizon": "2",
            "fscs": "10",
            "epsilon_0_classes": "5",
            "epsilon_01_classes": "5",
            "epsilon_03_classes": "4",
            "epsilon_05_classes": "3",
            "runtime_seconds_reported": "11",
            "ci_lo": "0.28",
            "ci_hi": "0.40",
        },
        {
            "benchmark": "Net. Mon. (n=9)",
            "state_count": "512",
            "m": "2",
            "horizon": "2",
            "fscs": "6410",
            "epsilon_0_classes": "5",
            "epsilon_01_classes": "5",
            "epsilon_03_classes": "5",
            "epsilon_05_classes": "3",
            "runtime_seconds_reported": "6802",
            "ci_lo": "0.39",
            "ci_hi": "0.49",
        },
    ]
    assert_rows("m2_medium_scale.csv", m2_scaling, expected_m2_scaling)

    timing_summary = read_csv("timing_summary.csv")
    if len(timing_summary) < 9:
        raise AssertionError("timing_summary.csv is unexpectedly incomplete.")

    tier3_timing = {
        (row["artifact"], row["benchmark"]): row["runtime_seconds_reported"]
        for row in timing_summary
        if row["tier"] == "Tier III"
    }
    expected_timing = {
        ("meaningful_scale.csv", "Net. Mon. (n=12)"): "~5",
        ("meaningful_scale.csv", "Random (|O|=10)"): "~30",
        ("meaningful_scale.csv", "Net. Mon. (n=10)"): "~25",
        ("m2_medium_scale.csv", "RockSample(4,4) m=2"): "321",
        ("m2_medium_scale.csv", "Net. Mon. (n=4) m=2"): "353",
        ("m2_medium_scale.csv", "Net. Mon. (n=9) m=2"): "6802",
    }
    for key, expected in expected_timing.items():
        actual = tier3_timing.get(key)
        if actual != expected:
            raise AssertionError(f"Timing crosswalk mismatch for {key}: expected {expected}, got {actual}")

    print("Tier III artifact package verified.")
    print(f"Checked {len(meaningful)} meaningful-scale rows, {len(m2_scaling)} higher-memory rows, and timing crosswalk entries.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(f"Tier III artifact verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
