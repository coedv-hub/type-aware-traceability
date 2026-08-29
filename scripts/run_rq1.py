#!/usr/bin/env python3
"""Generate RQ1 benchmark statistics and the paper Table 3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DatasetLoader
from src.evaluation.rq1_statistics import build_statistics, export_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "datasets.yaml"),
        help="Dataset configuration file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loader = DatasetLoader(args.config)
    table = build_statistics(loader)
    csv_path, latex_path = export_statistics(table, PROJECT_ROOT / "tables")
    print(table.to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"LaTeX: {latex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
