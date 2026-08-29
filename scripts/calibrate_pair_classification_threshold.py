#!/usr/bin/env python3
"""Calibrate one global per-method threshold for the pair-classification metric.

Reads each method's predictions.csv (from run_pair_classification.py),
restricts to the validation_requirement_ids in configs/validation_split.json
pooled across all 4 active datasets, and picks the threshold that maximizes
pooled F1. Per configs/pair_classification_protocol.yaml: never uses
eANCI's full 39-requirement result or any Assumption/ value for this
selection. Writes results to audit/pair_classification_validation/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_directory  # noqa: E402
from scripts.run_llm_baselines import safe_name  # noqa: E402
from scripts.run_pair_classification import (  # noqa: E402
    ABLATION_LABELS,
    PROPOSED_PAIR_CLASSIFIER_LABELS,
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "pair_classification"
AUDIT_DIR = PROJECT_ROOT / "audit" / "pair_classification_validation"
VALIDATION_SPLIT = PROJECT_ROOT / "configs" / "validation_split.json"
FROZEN_THRESHOLDS_PATH = PROJECT_ROOT / "configs" / "pair_classification_thresholds.yaml"
# The four-dataset CALIBRATION scope. Deliberately hardcoded, not derived
# from DatasetLoader.dataset_names (which now also lists "Industrial" --
# re-enabled 2026-07-24 as a fully external held-out dataset for this
# track, see configs/datasets.yaml and
# scripts/generate_pair_classification_manifest.py). Industrial has no
# validation_requirement_ids entry in configs/validation_split.json and
# must NEVER be added to this tuple: it is only ever scored once, on
# heldout, using the threshold this script freezes from these 4 datasets
# alone -- see scripts/aggregate_pair_classification_tables.py's separate
# 5-dataset REPORTING_DATASETS for where Industrial's result is reported.
DATASETS = ("eTour", "eANCI", "iTrust", "LibEST")
CALIBRATION_METHODS = (
    "TFIDF", "BM25", "LSI", "Sentence-BERT", "CodeBERT",
    "Direct Prompting", "Generic RAG-LLM",
) + PROPOSED_PAIR_CLASSIFIER_LABELS + ABLATION_LABELS


def candidate_thresholds(scores: pd.Series) -> list[float]:
    """Every observed score is a genuine partition point under score>=t, so
    an exhaustive search only needs the sorted unique values themselves
    (plus one point below the minimum, to allow the "predict everything
    positive" cutoff) -- a fixed 0.05-step grid can miss the true optimum
    if scores cluster off-grid."""
    unique_sorted = sorted(scores.unique())
    if not unique_sorted:
        return [0.0]
    below_min = unique_sorted[0] - 1e-9
    return [below_min] + unique_sorted


def load_validation_requirement_ids() -> dict[str, set[str]]:
    payload = json.loads(VALIDATION_SPLIT.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise SystemExit(f"Validation split is not frozen: {VALIDATION_SPLIT}")
    return {
        dataset_name: set(info["validation_requirement_ids"])
        for dataset_name, info in payload["datasets"].items()
        if dataset_name in DATASETS
    }


def precision_recall_f1(labels: pd.Series, predicted_positive: pd.Series) -> tuple[float, float, float]:
    tp = int(((labels == 1) & predicted_positive).sum())
    fp = int(((labels == 0) & predicted_positive).sum())
    fn = int(((labels == 1) & ~predicted_positive).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def calibrate_method(method: str, validation_ids: dict[str, set[str]]) -> dict | None:
    frames = []
    for dataset_name in DATASETS:
        full_path = RESULTS_ROOT / dataset_name / safe_name(method) / "predictions.csv"
        validation_only_path = (
            RESULTS_ROOT / dataset_name / safe_name(method) / "_split_validation" / "predictions.csv"
        )
        if full_path.is_file():
            predictions = pd.read_csv(full_path)
            by_split = predictions[predictions["Split"] == "validation"]
        elif validation_only_path.is_file():
            # Run via `--split validation`: the file already contains only
            # validation rows, but still verify that -- never silently
            # trust a filename convention over the data itself.
            predictions = pd.read_csv(validation_only_path)
            if not (predictions["Split"] == "validation").all():
                raise SystemExit(
                    f"{method}/{dataset_name}: {validation_only_path} contains "
                    "non-validation rows despite being the validation-only "
                    "output path."
                )
            by_split = predictions
        else:
            return None

        by_ids_reference = predictions if full_path.is_file() else by_split
        by_ids = by_ids_reference[
            by_ids_reference["Req_ID"].isin(validation_ids[dataset_name])
        ]
        if set(by_split["Req_ID"]) != set(by_ids["Req_ID"]):
            raise SystemExit(
                f"{method}/{dataset_name}: manifest Split column disagrees "
                "with a fresh load of configs/validation_split.json -- "
                "regenerate the manifest."
            )
        frames.append(by_split)
    pooled = pd.concat(frames, ignore_index=True)
    if pooled.empty:
        return None

    best = None
    for threshold in candidate_thresholds(pooled["Score"]):
        predicted_positive = pooled["Score"] >= threshold
        precision, recall, f1 = precision_recall_f1(pooled["Label"], predicted_positive)
        row = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pooled_pairs": len(pooled),
        }
        if best is None or f1 > best["f1"]:
            best = row
    return best


def main() -> int:
    validation_ids = load_validation_requirement_ids()
    ensure_directory(AUDIT_DIR)
    rows = []
    for method in CALIBRATION_METHODS:
        result = calibrate_method(method, validation_ids)
        if result is None:
            print(f"{method}: no predictions.csv for all 4 datasets yet, skipped.")
            continue
        result["method"] = method
        rows.append(result)
        print(
            f"{method}: threshold={result['threshold']:.2f} "
            f"F1={result['f1']:.4f} P={result['precision']:.4f} "
            f"R={result['recall']:.4f} (n={result['pooled_pairs']} pooled "
            "validation pairs, 4 datasets)"
        )

    if rows:
        frame = pd.DataFrame(rows)[
            ["method", "threshold", "precision", "recall", "f1", "pooled_pairs"]
        ]
        frame.to_csv(AUDIT_DIR / "threshold_calibration.csv", index=False)
        print(f"\nWrote {AUDIT_DIR / 'threshold_calibration.csv'}")
        write_frozen_thresholds_yaml(rows)
        print(f"Wrote {FROZEN_THRESHOLDS_PATH}")
    else:
        print("No methods had complete predictions yet; nothing written.")
    return 0


def write_frozen_thresholds_yaml(rows: list[dict]) -> None:
    lines = [
        "# Frozen per-method pair-classification thresholds.",
        "# Regenerated by scripts/calibrate_pair_classification_threshold.py --",
        "# do not hand-edit. Selected by maximizing F1 on the pooled",
        "# validation_requirement_ids from configs/validation_split.json across",
        "# all 4 active datasets (global, not per-dataset). Search space is",
        "# every unique observed validation score (exhaustive, not a step grid).",
        "# See configs/pair_classification_protocol.yaml for the full protocol.",
        "",
        "status: frozen",
        f"calibrated_on_pooled_validation_pairs: {rows[0]['pooled_pairs']}",
        "",
        "methods:",
    ]
    for row in rows:
        lines.append(f"  {row['method']!r}:")
        lines.append(f"    threshold: {row['threshold']}")
        lines.append(f"    validation_precision: {row['precision']}")
        lines.append(f"    validation_recall: {row['recall']}")
        lines.append(f"    validation_f1: {row['f1']}")
    FROZEN_THRESHOLDS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
