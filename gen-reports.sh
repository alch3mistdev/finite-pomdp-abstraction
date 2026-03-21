#!/bin/bash

# Run the basic results experiment and write outputs to reference_results/
python -m experiments.run_basic_results \
  --profile quick \
  --seed 7 \
  --include-hierarchical-scaling \
  --output-dir ./reference_results
