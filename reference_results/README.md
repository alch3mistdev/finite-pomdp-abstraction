# Reference Results

This directory contains the exact experimental results used in the paper, generated with:

```bash
python -m experiments.run_basic_results --profile extended --seed 7
```

These are provided so you can verify your own reproduced results against the paper's data. CSV files contain the raw numerical data; PNG files are the corresponding figures.

To regenerate these results yourself, run the command above from the repository root.

## New in v0.1.1

| File | Description |
|------|-------------|
| `stationary_counterexample.csv` | Stationary-policy counterexample data |
| `value_bound_tightness_real_reward.csv` | Value-bound tightness under real reward |
| `fig_value_bound_tightness_real_reward.png` | Corresponding figure |
