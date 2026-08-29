#!/usr/bin/env python3
"""Validate one or all curated datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DatasetLoader
from src.data.validator import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", help="Dataset name or 'all'.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "datasets.yaml"),
        help="Dataset configuration file.",
    )
    return parser.parse_args()


def select_datasets(loader: DatasetLoader, requested: str) -> list[str]:
    if requested.casefold() == "all":
        return loader.dataset_names
    lookup = {name.casefold(): name for name in loader.dataset_names}
    if requested.casefold() not in lookup:
        raise SystemExit(
            f"Unknown dataset {requested!r}; choose from {loader.dataset_names} or all."
        )
    return [lookup[requested.casefold()]]


def main() -> int:
    args = parse_args()
    loader = DatasetLoader(args.config)
    summaries = []
    for name in select_datasets(loader, args.dataset):
        summary = validate_dataset(loader.load(name))
        summaries.append(summary)
        status = "PASS" if summary.valid else "FAIL"
        print(
            f"[{status}] {name}: req={summary.requirements}, "
            f"code={summary.code_files}, links={summary.positive_links}, "
            f"no-code={summary.no_code_requirements}, "
            f"candidate-pairs={summary.candidate_pairs}, "
            f"errors={summary.errors}, warnings={summary.warnings}"
        )
        for issue in summary.issues:
            print(f"  - {issue.level.upper()} {issue.check}: {issue.message}")

    print("\nValidation summary (JSON)")
    print(json.dumps([summary.to_dict() for summary in summaries], indent=2))
    return 0 if all(summary.valid for summary in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
