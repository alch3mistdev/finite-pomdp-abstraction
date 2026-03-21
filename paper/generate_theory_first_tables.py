#!/usr/bin/env python3
"""Generate the theory-first paper tables from repo experiment code."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis import export_paper_theory_first_tables


def main() -> None:
    output_dir = REPO_ROOT / "paper" / "generated" / "data"
    paper_generated_dir = REPO_ROOT / "paper" / "generated"
    export_paper_theory_first_tables(output_dir=output_dir, paper_generated_dir=paper_generated_dir)
    print(f"Wrote theory-first tables to {paper_generated_dir}")


if __name__ == "__main__":
    main()
