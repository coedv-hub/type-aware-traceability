#!/usr/bin/env python3
"""Run Sentence-BERT and CodeBERT on one or all datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.neural import create_neural_baseline
from src.data.loader import DatasetLoader, TraceDataset
from src.evaluation.metrics import evaluate_rankings
from src.utils import ensure_directory, load_config


METHODS = ("sentencebert", "codebert")
METHOD_LABELS = {
    "sentencebert": "Sentence-BERT",
    "codebert": "CodeBERT",
}
GROUPS = ("Overall", "FR", "NFR", "Mixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", help="Dataset name or 'all'.")
    parser.add_argument(
        "--method",
        default="all",
        choices=[*METHODS, "all"],
        help="Neural baseline to run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device; auto prefers CUDA/MPS and falls back to CPU.",
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
    return {
        req_id: sorted(
            [
                (code_id, float(scores[req_index, code_index]))
                for code_index, code_id in enumerate(code_ids)
            ],
            key=lambda item: (-item[1], item[0]),
        )
        for req_index, req_id in enumerate(requirement_ids)
    }


def output_frames(
    dataset: TraceDataset,
    requirement_ids: list[str],
    code_ids: list[str],
    scores: np.ndarray,
    rankings: dict[str, list[tuple[str, float]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positives = dataset.positive_links
    raw = pd.DataFrame(
        [
            {
                "Req_ID": req_id,
                "Type": dataset.requirements[req_id].req_type,
                "Code_File": code_id,
                "Score": float(scores[req_index, code_index]),
                "Is_Positive": (req_id, code_id) in positives,
            }
            for req_index, req_id in enumerate(requirement_ids)
            for code_index, code_id in enumerate(code_ids)
        ]
    )
    ranked = pd.DataFrame(
        [
            {
                "Req_ID": req_id,
                "Type": dataset.requirements[req_id].req_type,
                "Rank": rank,
                "Code_File": code_id,
                "Score": score,
                "Is_Positive": (req_id, code_id) in positives,
            }
            for req_id in requirement_ids
            for rank, (code_id, score) in enumerate(rankings[req_id], start=1)
        ]
    )
    return raw, ranked


def run_method(
    dataset: TraceDataset,
    method: str,
    config: dict[str, Any],
    cache_dir: Path,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    requirements = dataset.evaluation_requirements
    requirement_ids = list(requirements)
    code_ids = list(dataset.code_files)
    requirement_texts = [requirements[item].text for item in requirement_ids]
    code_texts = [dataset.code_files[item].text for item in code_ids]
    baseline = create_neural_baseline(
        method, config, dataset.name, cache_dir, device
    )

    started = time.perf_counter()
    baseline.fit(code_ids, code_texts)
    scores = baseline.score_all(requirement_ids, requirement_texts)
    elapsed = time.perf_counter() - started
    expected_shape = (len(requirement_ids), len(code_ids))
    if scores.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected score shape {scores.shape}; expected {expected_shape}"
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
    metrics.insert(0, "Method", METHOD_LABELS[method])
    metrics.insert(0, "Dataset", dataset.name)
    metrics["RuntimeSeconds"] = elapsed
    metrics["BinaryTopK"] = int(evaluation["binary_top_k"])
    metrics["BinaryProtocol"] = str(evaluation.get("binary_protocol", "top_k"))
    metrics["BinaryThreshold"] = float("nan")

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
            for cutoff in evaluation["ranking_cutoffs"]
            for metric in (f"P@{int(cutoff)}", f"R@{int(cutoff)}")
        ],
    ]
    populated = metrics.loc[metrics["Requirements"].gt(0), metric_columns]
    ranking_lengths = [len(items) for items in rankings.values()]
    empty_embeddings = 0
    if baseline.code_embeddings is not None:
        empty_embeddings += int(
            (np.linalg.norm(baseline.code_embeddings, axis=1) <= 0.0).sum()
        )
    cache_misses = sum(
        stats.embeddings for stats in baseline.cache_stats.values() if not stats.hit
    )
    verification_misses = sum(
        stats.verification_misses for stats in baseline.cache_stats.values()
    )
    diagnostics = {
        "success": True,
        "device": baseline.device,
        "model": baseline.model_name,
        "model_revision": baseline.model_revision,
        "runtime_seconds": elapsed,
        "score_shape": list(scores.shape),
        "nan_scores": int(np.isnan(scores).sum()),
        "nonfinite_scores": int((~np.isfinite(scores)).sum()),
        "abnormal_similarity": int(((scores < -1.0) | (scores > 1.0)).sum()),
        "empty_rankings": sum(length == 0 for length in ranking_lengths),
        "incomplete_rankings": sum(
            length != len(code_ids) for length in ranking_lengths
        ),
        "empty_embeddings": empty_embeddings,
        "cache_misses": cache_misses,
        "cache_verification_misses": verification_misses,
        "cache": {
            role: {
                "hit": stats.hit,
                "embeddings": stats.embeddings,
                "path": stats.path,
            }
            for role, stats in baseline.cache_stats.items()
        },
        "undefined_metric_groups": metrics.loc[
            metrics["Requirements"].eq(0), "Group"
        ].tolist(),
        "nan_metrics": int(metrics[metric_columns].isna().sum().sum()),
        "unexpected_nan_metrics": int(populated.isna().sum().sum()),
    }
    diagnostics["has_anomaly"] = any(
        diagnostics[key]
        for key in (
            "nan_scores",
            "nonfinite_scores",
            "abnormal_similarity",
            "empty_rankings",
            "incomplete_rankings",
            "empty_embeddings",
            "cache_verification_misses",
            "unexpected_nan_metrics",
        )
    )
    return raw, ranked, metrics, diagnostics


def update_table(
    rows: pd.DataFrame, path: Path, dataset_order: list[str]
) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_csv(path)
        keys = set(zip(rows["Dataset"], rows["Method"], rows["Group"]))
        existing = existing[
            ~existing.apply(
                lambda row: (row["Dataset"], row["Method"], row["Group"]) in keys,
                axis=1,
            )
        ]
        rows = pd.concat([existing, rows], ignore_index=True)
    rows["_dataset_order"] = pd.Categorical(
        rows["Dataset"], categories=dataset_order, ordered=True
    )
    rows["_method_order"] = pd.Categorical(
        rows["Method"], categories=list(METHOD_LABELS.values()), ordered=True
    )
    rows["_group_order"] = pd.Categorical(
        rows["Group"], categories=list(GROUPS), ordered=True
    )
    rows = (
        rows.sort_values(["_dataset_order", "_method_order", "_group_order"])
        .drop(columns=["_dataset_order", "_method_order", "_group_order"])
        .reset_index(drop=True)
    )
    rows.to_csv(path, index=False)
    return rows


def dataset_summary(dataset: TraceDataset) -> dict[str, Any]:
    requirements = dataset.evaluation_requirements
    return {
        "dataset": dataset.name,
        "evaluated_requirements": len(requirements),
        "candidate_pairs": len(requirements) * len(dataset.code_files),
        "methods": {},
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    loader = DatasetLoader(args.config)
    datasets = select_datasets(loader, args.dataset)
    methods = METHODS if args.method == "all" else (args.method,)
    results_root = ensure_directory(
        PROJECT_ROOT / config["output"]["neural_results_dir"]
    )
    tables_root = ensure_directory(PROJECT_ROOT / config["output"]["tables_dir"])
    cache_dir = ensure_directory(PROJECT_ROOT / config["neural"]["cache_dir"])

    metric_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    failed = False
    for dataset_name in datasets:
        dataset = loader.load(dataset_name)
        summary = dataset_summary(dataset)
        for method in methods:
            print(f"Running {METHOD_LABELS[method]} on {dataset_name}...")
            try:
                raw, ranked, metrics, diagnostics = run_method(
                    dataset, method, config, cache_dir, args.device
                )
                output_dir = ensure_directory(results_root / dataset_name)
                raw.to_csv(output_dir / f"{method}_raw_scores.csv", index=False)
                ranked.to_csv(
                    output_dir / f"{method}_ranked_results.csv", index=False
                )
                metrics.loc[metrics["Group"].eq("Overall")].to_csv(
                    output_dir / f"{method}_overall_metrics.csv", index=False
                )
                metrics.loc[~metrics["Group"].eq("Overall")].to_csv(
                    output_dir / f"{method}_type_wise_metrics.csv", index=False
                )
                metric_frames.append(metrics)
                summary["methods"][METHOD_LABELS[method]] = diagnostics
                print(metrics.to_string(index=False))
            except Exception as exc:
                failed = True
                summary["methods"][METHOD_LABELS[method]] = {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "has_anomaly": True,
                }
                print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        summaries.append(summary)

    if metric_frames:
        all_metrics = pd.concat(metric_frames, ignore_index=True)
        update_table(
            all_metrics.loc[all_metrics["Group"].eq("Overall")].copy(),
            tables_root / "neural_baseline_overall_metrics.csv",
            loader.dataset_names,
        )
        update_table(
            all_metrics.loc[~all_metrics["Group"].eq("Overall")].copy(),
            tables_root / "neural_baseline_type_wise_metrics.csv",
            loader.dataset_names,
        )

    run_summary = {
        "protocol": {
            "binary_metrics": "micro pair-level Precision/Recall/F1",
            "binary_protocol": config["evaluation"]["binary_protocol"],
            "binary_top_k": int(config["evaluation"]["binary_top_k"]),
            "ranking_metrics": "macro mean over requirement-level rankings",
            "ranking_cutoffs": [
                int(value) for value in config["evaluation"]["ranking_cutoffs"]
            ],
        },
        "datasets": summaries,
        "all_requested_methods_succeeded": not failed,
        "any_anomaly": any(
            method.get("has_anomaly", False)
            for dataset in summaries
            for method in dataset["methods"].values()
        ),
    }
    (results_root / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    flat_rows = []
    for dataset in summaries:
        for method, diagnostics in dataset["methods"].items():
            flat_rows.append(
                {
                    key: value
                    for key, value in {
                        "dataset": dataset["dataset"],
                        "evaluated_requirements": dataset[
                            "evaluated_requirements"
                        ],
                        "candidate_pairs": dataset["candidate_pairs"],
                        "method": method,
                        **diagnostics,
                    }.items()
                    if not isinstance(value, (list, dict))
                }
            )
    pd.DataFrame(flat_rows).to_csv(results_root / "run_summary.csv", index=False)
    print(f"\nRun summary: {results_root / 'run_summary.json'}")
    if not metric_frames:
        print("No neural baseline completed successfully.", file=sys.stderr)
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
