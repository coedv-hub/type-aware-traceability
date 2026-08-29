#!/usr/bin/env python3
"""Run TF-IDF, BM25, and LSI on one or all datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib")
)

import numpy as np
import pandas as pd

sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.ir import create_baseline
from src.data.loader import DatasetLoader, TraceDataset
from src.evaluation.metrics import evaluate_rankings
from src.utils import ensure_directory, load_config


METHODS = ("tfidf", "bm25", "lsi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", help="Dataset name or 'all'.")
    parser.add_argument(
        "--method",
        default="all",
        choices=[*METHODS, "all"],
        help="IR baseline to run.",
    )
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


def rank_scores(
    requirement_ids: list[str], code_ids: list[str], scores: np.ndarray
) -> dict[str, list[tuple[str, float]]]:
    rankings: dict[str, list[tuple[str, float]]] = {}
    for row_index, req_id in enumerate(requirement_ids):
        pairs = [
            (code_id, float(scores[row_index, code_index]))
            for code_index, code_id in enumerate(code_ids)
        ]
        rankings[req_id] = sorted(pairs, key=lambda item: (-item[1], item[0]))
    return rankings


def output_frames(
    dataset: TraceDataset,
    requirement_ids: list[str],
    code_ids: list[str],
    scores: np.ndarray,
    rankings: dict[str, list[tuple[str, float]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positives = dataset.positive_links
    raw_rows = []
    for req_index, req_id in enumerate(requirement_ids):
        req_type = dataset.requirements[req_id].req_type
        for code_index, code_id in enumerate(code_ids):
            raw_rows.append(
                {
                    "Req_ID": req_id,
                    "Type": req_type,
                    "Code_File": code_id,
                    "Score": float(scores[req_index, code_index]),
                    "Is_Positive": (req_id, code_id) in positives,
                }
            )

    ranked_rows = []
    for req_id in requirement_ids:
        req_type = dataset.requirements[req_id].req_type
        for rank, (code_id, score) in enumerate(rankings[req_id], start=1):
            ranked_rows.append(
                {
                    "Req_ID": req_id,
                    "Type": req_type,
                    "Rank": rank,
                    "Code_File": code_id,
                    "Score": score,
                    "Is_Positive": (req_id, code_id) in positives,
                }
            )
    return pd.DataFrame(raw_rows), pd.DataFrame(ranked_rows)


def run_method(
    dataset: TraceDataset, method: str, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    requirements = dataset.evaluation_requirements
    requirement_ids = list(requirements)
    code_ids = list(dataset.code_files)
    requirement_texts = [requirements[req_id].text for req_id in requirement_ids]
    code_texts = [dataset.code_files[code_id].text for code_id in code_ids]

    baseline = create_baseline(method, config)
    started = time.perf_counter()
    baseline.fit(code_ids, code_texts)
    scores = baseline.score_all(requirement_texts)
    elapsed = time.perf_counter() - started
    if scores.shape != (len(requirement_ids), len(code_ids)):
        raise RuntimeError(
            f"Unexpected score shape {scores.shape}; expected "
            f"{(len(requirement_ids), len(code_ids))}"
        )

    rankings = rank_scores(requirement_ids, code_ids, scores)
    evaluation = config["evaluation"]
    metrics = evaluate_rankings(
        ranked=rankings,
        gold=dataset.ground_truth_by_requirement,
        requirement_types={
            req_id: requirement.req_type
            for req_id, requirement in requirements.items()
        },
        binary_top_k=int(evaluation["binary_top_k"]),
        cutoffs=tuple(int(value) for value in evaluation["ranking_cutoffs"]),
    )
    metrics.insert(0, "Method", method.upper())
    metrics.insert(0, "Dataset", dataset.name)
    metrics["BinaryTopK"] = int(evaluation["binary_top_k"])
    metrics["BinaryProtocol"] = str(evaluation.get("binary_protocol", "top_k"))
    metrics["BinaryThreshold"] = float("nan")
    metrics["RuntimeSeconds"] = elapsed
    raw, ranked = output_frames(
        dataset, requirement_ids, code_ids, scores, rankings
    )
    metric_columns = [
        "Precision",
        "Recall",
        "F1",
        "MRR",
        *[
            metric
            for k in evaluation["ranking_cutoffs"]
            for metric in (f"P@{int(k)}", f"R@{int(k)}")
        ],
    ]
    metric_values = metrics[metric_columns].to_numpy(dtype=float)
    populated_metric_values = metrics.loc[
        metrics["Requirements"].gt(0), metric_columns
    ].to_numpy(dtype=float)
    undefined_groups = metrics.loc[
        metrics["Requirements"].eq(0), "Group"
    ].tolist()
    ranking_lengths = [len(items) for items in rankings.values()]
    diagnostics = {
        "success": True,
        "runtime_seconds": elapsed,
        "score_shape": list(scores.shape),
        "nan_scores": int(np.isnan(scores).sum()),
        "nonfinite_scores": int((~np.isfinite(scores)).sum()),
        "empty_rankings": sum(length == 0 for length in ranking_lengths),
        "incomplete_rankings": sum(
            length != len(code_ids) for length in ranking_lengths
        ),
        "empty_positive_links": sum(
            not dataset.ground_truth_by_requirement[req_id]
            for req_id in requirement_ids
        ),
        "nan_metrics": int(np.isnan(metric_values).sum()),
        "undefined_metric_groups": undefined_groups,
        "unexpected_nan_metrics": int(
            np.isnan(populated_metric_values).sum()
        ),
        "out_of_range_metrics": int(
            ((metric_values < 0.0) | (metric_values > 1.0)).sum()
        ),
    }
    diagnostics["has_anomaly"] = any(
        diagnostics[key]
        for key in (
            "nan_scores",
            "nonfinite_scores",
            "empty_rankings",
            "incomplete_rankings",
            "empty_positive_links",
            "unexpected_nan_metrics",
            "out_of_range_metrics",
        )
    )
    return raw, ranked, metrics, diagnostics


def update_summary_table(
    new_rows: pd.DataFrame, path: Path, dataset_order: list[str]
) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_csv(path)
        keys = set(
            zip(
                new_rows["Dataset"],
                new_rows["Method"],
                new_rows["Group"],
            )
        )
        existing = existing[
            ~existing.apply(
                lambda row: (row["Dataset"], row["Method"], row["Group"]) in keys,
                axis=1,
            )
        ]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()
    combined["_dataset_order"] = pd.Categorical(
        combined["Dataset"], categories=dataset_order, ordered=True
    )
    combined["_method_order"] = pd.Categorical(
        combined["Method"],
        categories=[method.upper() for method in METHODS],
        ordered=True,
    )
    combined["_group_order"] = pd.Categorical(
        combined["Group"],
        categories=["Overall", "FR", "NFR", "Mixed"],
        ordered=True,
    )
    combined = (
        combined.sort_values(["_dataset_order", "_method_order", "_group_order"])
        .drop(columns=["_dataset_order", "_method_order", "_group_order"])
        .reset_index(drop=True)
    )
    combined.to_csv(path, index=False)
    return combined


def write_table(frame: pd.DataFrame, csv_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    csv_path.with_suffix(".tex").write_text(
        frame.to_latex(index=False, float_format="%.4f", na_rep="--"),
        encoding="utf-8",
    )


def dataset_summary(dataset: TraceDataset) -> dict[str, Any]:
    requirements = dataset.evaluation_requirements
    gold = dataset.ground_truth_by_requirement
    return {
        "dataset": dataset.name,
        "total_requirements": len(dataset.requirements),
        "evaluated_requirements": len(requirements),
        "excluded_no_code_requirements": (
            len(dataset.requirements) - len(requirements)
        ),
        "code_files": len(dataset.code_files),
        "candidate_pairs": len(requirements) * len(dataset.code_files),
        "positive_links": sum(len(items) for items in gold.values()),
        "empty_positive_link_requirements": sum(
            not gold[req_id] for req_id in requirements
        ),
        "methods": {},
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    loader = DatasetLoader(args.config)
    datasets = select_datasets(loader, args.dataset)
    methods = METHODS if args.method == "all" else (args.method,)
    results_root = ensure_directory(PROJECT_ROOT / config["output"]["results_dir"])
    tables_root = ensure_directory(PROJECT_ROOT / config["output"]["tables_dir"])

    metric_frames: list[pd.DataFrame] = []
    run_summaries: list[dict[str, Any]] = []
    failed = False
    for dataset_name in datasets:
        dataset = loader.load(dataset_name)
        summary = dataset_summary(dataset)
        for method in methods:
            print(f"Running {method.upper()} on {dataset_name}...")
            try:
                raw, ranked, metrics, diagnostics = run_method(
                    dataset, method, config
                )
                output_dir = ensure_directory(results_root / dataset_name)
                raw_path = output_dir / f"{method}_raw_scores.csv"
                ranked_path = output_dir / f"{method}_ranked_results.csv"
                overall_path = output_dir / f"{method}_overall_metrics.csv"
                type_path = output_dir / f"{method}_type_wise_metrics.csv"
                raw.to_csv(raw_path, index=False)
                ranked.to_csv(ranked_path, index=False)
                metrics.loc[metrics["Group"].eq("Overall")].to_csv(
                    overall_path, index=False
                )
                metrics.loc[~metrics["Group"].eq("Overall")].to_csv(
                    type_path, index=False
                )
                metric_frames.append(metrics)
                summary["methods"][method.upper()] = diagnostics
                print(metrics.to_string(index=False))
                print(f"  raw scores:     {raw_path}")
                print(f"  rankings:       {ranked_path}")
                print(f"  overall metrics:{overall_path}")
                print(f"  type metrics:   {type_path}")
            except Exception as exc:
                failed = True
                summary["methods"][method.upper()] = {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "has_anomaly": True,
                }
                print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        run_summaries.append(summary)

    if not metric_frames:
        raise RuntimeError("No IR baseline completed successfully.")

    metrics = pd.concat(metric_frames, ignore_index=True)
    summary_path = tables_root / "ir_baseline_metrics.csv"
    combined = update_summary_table(metrics, summary_path, loader.dataset_names)
    write_table(combined, summary_path)
    overall_table = combined.loc[combined["Group"].eq("Overall")].reset_index(
        drop=True
    )
    type_table = combined.loc[~combined["Group"].eq("Overall")].reset_index(
        drop=True
    )
    overall_path = tables_root / "baseline_overall_metrics.csv"
    type_path = tables_root / "baseline_type_wise_metrics.csv"
    write_table(overall_table, overall_path)
    write_table(type_table, type_path)

    protocol = {
        "binary_metrics": "micro pair-level Precision/Recall/F1",
        "binary_protocol": config["evaluation"]["binary_protocol"],
        "binary_top_k": int(config["evaluation"]["binary_top_k"]),
        "ranking_metrics": "macro mean over requirement-level rankings",
        "ranking_cutoffs": [
            int(value) for value in config["evaluation"]["ranking_cutoffs"]
        ],
        "no_code_requirements_excluded": True,
        "type_wise_scope": "only requirements whose canonical Type equals the group",
    }
    run_summary = {
        "protocol": protocol,
        "datasets": run_summaries,
        "all_requested_methods_succeeded": not failed,
        "any_anomaly": any(
            method_summary.get("has_anomaly", False)
            for dataset_summary_item in run_summaries
            for method_summary in dataset_summary_item["methods"].values()
        ),
        "any_nan_metrics": any(
            method_summary.get("nan_metrics", 0) > 0
            for dataset_summary_item in run_summaries
            for method_summary in dataset_summary_item["methods"].values()
        ),
    }
    run_summary_path = results_root / "run_summary.json"
    run_summary_path.write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    flat_summary_rows = []
    for item in run_summaries:
        for method, method_summary in item["methods"].items():
            flat_summary_rows.append(
                {
                    key: value
                    for key, value in {
                        **{k: v for k, v in item.items() if k != "methods"},
                        "method": method,
                        **method_summary,
                    }.items()
                    if not isinstance(value, (list, dict))
                }
            )
    pd.DataFrame(flat_summary_rows).to_csv(
        results_root / "run_summary.csv", index=False
    )
    print(f"\nMetrics table: {summary_path}")
    print(f"Overall table: {overall_path}")
    print(f"Type table:    {type_path}")
    print(f"Run summary:   {run_summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
