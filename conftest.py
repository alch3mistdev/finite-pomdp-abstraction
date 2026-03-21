from __future__ import annotations

import math
import os


# Keep BLAS/OpenMP backends from oversubscribing CPUs across xdist workers.
_THREAD_CAP_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

for _var in _THREAD_CAP_VARS:
    os.environ.setdefault(_var, "1")


def _parse_positive_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 1:
        return None
    return value


def _parse_positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value <= 0:
        return default
    return value


def pytest_xdist_auto_num_workers(config) -> int:
    del config
    override_workers = _parse_positive_int("PYTEST_WORKERS")
    if override_workers is not None:
        return override_workers

    fraction = _parse_positive_float("PYTEST_WORKER_FRACTION", default=0.5)
    cpu_count = os.cpu_count() or 1
    workers = max(1, math.floor(cpu_count * fraction))
    return min(cpu_count, workers)
