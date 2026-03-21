# Tier 3 Artifacts

This directory is a reproducibility package for the paper's heavy operational stress tracks.

Provenance:
- Values are transcribed from the current manuscript tables in `paper/Nixon2026_FinitePOMDPAbstraction_v0.1.1.tex`.
- No new experiments were run to create these files.
- No paper text or claim files were edited as part of this package.

Files:
- `meaningful_scale.csv` maps to `tab:meaningful-scale`.
- `m2_medium_scale.csv` maps to `tab:m2-scaling`.
- `computational_profile.csv` maps to `tab:computational-profile`.
- `bootstrap_coverage.csv` maps to `tab:bootstrap-coverage`.
- `timing_summary.csv` is a convenience index compiled from the runtime values reported in the manuscript tables above.
- `verify_tier3_artifacts.py` checks the archived stress-track rows and timing crosswalk without rerunning the heavy experiments.

Scope:
- `meaningful_scale.csv` contains the Tier III large-state and long-horizon rows discussed in the paper.
- `m2_medium_scale.csv` contains the Tier III probe-family scaling rows.
- `computational_profile.csv` contains the appendix bottleneck breakdown for the m=1 benchmarks.
- `bootstrap_coverage.csv` contains the appendix bootstrap CI coverage check.
- `timing_summary.csv` cross-references the runtime values cited in the paper for the heavy tracks.

Interpretation:
- These are inspection artifacts, not extra claims.
- The paper remains the canonical source for the scientific interpretation.
- If a value here differs from the manuscript, the manuscript should be treated as authoritative.

Quick check:
- Run `python artifacts/tier3/verify_tier3_artifacts.py` from the repository root.
