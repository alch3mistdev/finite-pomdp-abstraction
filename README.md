# Finite-POMDP Abstraction via Agent-Bounded Indistinguishability

Companion code for the paper:

> **Finite-POMDP Abstraction via Agent-Bounded Indistinguishability: A Bounded-Interaction Myhill-Nerode Theorem**
> Anthony T Nixon
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18904640.svg)](https://doi.org/10.5281/zenodo.18904640)

## Overview

This repository implements the quotient POMDP framework for finite POMDPs under bounded-agent indistinguishability. The framework defines a closed-loop pseudometric based on the 1-Wasserstein distance over observation-sequence distributions conditioned on finite-state controller (FSC) policies. Histories that no bounded agent can distinguish are merged into a quotient POMDP, yielding the unique minimal environment abstraction for the given agent class.

The code reproduces all experimental results from the paper, including capacity sweeps, value-loss bound validation, metric sensitivity analysis, spectral approximation, baseline comparisons, and horizon scaling.

## Installation

Requires Python 3.10-3.12.

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Reproducing All Results

**Quick profile** (~2 minutes):

```bash
python -m experiments.run_basic_results --profile quick --seed 7
```

**Extended profile** (finer epsilon grid, more stochastic samples):

```bash
python -m experiments.run_basic_results --profile extended --seed 7
```

Output artifacts (CSV tables and PNG figures) are written to `experiments/results/basic/` by default. Use `--output-dir <path>` to change.

Parallel execution is enabled by default. Use `--no-parallel` for serial runs, `--workers N` for a fixed worker count, or `--worker-fraction F` (default `0.5`) for automatic worker sizing.

### Long-Horizon Scaling

To include the hierarchical horizon-scaling experiments:

```bash
python -m experiments.run_basic_results --profile quick --include-hierarchical-scaling --max-horizon 10 --seed 7
```

## Determinism and Seeds

Core computations (history enumeration, distance computation, partitioning) are **fully deterministic** -- they perform exact enumeration over all histories and FSC policies. No randomness is involved.

The `--seed` flag affects **only** the stochastic FSC sampling used in the witness theorem validation experiment (`stochastic_vs_deterministic_sanity`). The default seed is `7`. All reported results are reproducible across runs and platforms with the same seed.

## Running Tests

```bash
# Default: parallel pytest with xdist (-n auto, 50% of logical cores)
pytest -v experiments/tests/

# Force serial execution
pytest -v -n 0 experiments/tests/

# Override worker count
PYTEST_WORKERS=4 pytest -v experiments/tests/

# Run a single test file
pytest -v experiments/tests/test_basic_results.py
```

## Output Artifacts

| File | Description |
|------|-------------|
| `tiger_reproduction.csv` | Tiger worked-example sanity check |
| `capacity_sweep_tiger.csv` | Class count vs memory bound and epsilon |
| `value_loss_bounds_lipschitz.csv` | Empirical value error vs theoretical bounds |
| `value_loss_nonlipschitz_tiger.csv` | Value error under non-Lipschitz reward |
| `horizon_gap_tiger.csv` | Bound gap growth with horizon |
| `metric_sensitivity_gridworld.csv` | W1 vs TV class counts on GridWorld |
| `stochastic_vs_deterministic_sanity.csv` | Witness theorem validation |
| `spectral_rank_analysis.csv` | Effective rank of FSC distance tensor |
| `spectral_partition_comparison.csv` | Exact vs k-FSC approximate partitions |
| `observation_noise_sensitivity.csv` | Sensitivity to observation accuracy |
| `multi_seed_witness.csv` | Multi-seed witness theorem stability |
| `baseline_comparison.csv` | Epsilon-quotient vs baseline methods |
| `ablation_studies.csv` | Ablation: W1 vs TV, m=1 vs m=2, spectral vs full |
| `hyperparameter_sensitivity.csv` | (epsilon, m, T) sweep heatmap data |
| `timing_summary.csv` | Wall-clock time per experiment |
| `fig_*.png` | Corresponding figures |

## CLI Options

```
--profile {quick,extended}    Runtime profile (default: quick)
--output-dir PATH             Output directory
--eps-grid "0,0.1,0.2,..."    Custom epsilon grid (CSV)
--ms "1,2"                    Memory bounds for capacity sweep
--horizons "2,3,4"            Horizons for gap experiment
--include-nonlipschitz        Include non-Lipschitz reward track (default: yes)
--include-spectral            Include spectral experiments (default: yes)
--include-hierarchical-scaling  Include long-horizon scaling tracks (default: no)
--stochastic-samples N        Override stochastic FSC sample count
--parallel / --no-parallel    Enable/disable parallel experiment tracks (default: parallel)
--workers N                   Fixed worker count (overrides auto fraction)
--worker-fraction F           Auto worker fraction of logical cores (default: 0.5)
--skip-plots                  Skip PNG generation
--seed N                      Random seed (default: 7)
```

## Architecture

```
pomdp_core.py          Core POMDP and FSC data structures
    |
metrics.py             TV and Wasserstein distance computation
benchmarks.py          Tiger, GridWorld, network monitoring POMDPs
fsc_enum.py            Deterministic/stochastic FSC enumeration
sampling.py            Monte Carlo trajectory sampling
    |
quotient.py            History partitioning and quotient construction
clustering.py          Complete-linkage and optimal partitioning
spectral.py            SVD-based spectral approximation
baselines.py           Truncation, random, belief-distance baselines
    |
hierarchical.py        Long-horizon layered scaling
analysis.py            Experiment orchestration and plotting
    |
run_basic_results.py   CLI entrypoint
```

### Key Abstractions

- **`FinitePOMDP`**: Frozen dataclass defining a POMDP with transition, observation, and reward matrices.
- **`DeterministicFSC` / `StochasticFSC`**: Finite-state controllers that define agent policies. Deterministic uses `(action, next_node)` mappings; stochastic uses `(alpha, beta)` probability tensors.
- **`DistanceCache`**: Pre-computed max-distance matrices across all policies for a given horizon and memory bound.
- **`PartitionResult`**: Output of clustering -- maps histories to equivalence classes.

### Core Algorithm Pipeline

1. Enumerate all observation histories of length T
2. For each history pair and each FSC policy, compute conditional observation-sequence distributions
3. Measure distances (TV or W1 via `scipy.optimize.linprog`) and take max over policies
4. Cluster histories via complete-linkage with threshold epsilon
5. Construct quotient POMDP and measure value loss against theoretical bounds

## Reference Results

The `reference_results/` directory contains the exact CSV data and PNG figures used in the paper, generated with `--profile extended --seed 7`. You can compare your generated results against these to verify reproducibility.

## Citation

```bibtex
@article{nixon2026finite,
  title={Finite-{POMDP} Abstraction via Agent-Bounded Indistinguishability:
         A Bounded-Interaction {M}yhill--{N}erode Theorem},
  author={Nixon, Anthony},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
