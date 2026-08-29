#!/usr/bin/env python3
"""Apply the frozen per-method threshold to a dataset's manifest.

Writes two families of output per dataset/method, both under
results/pair_classification/{dataset}/{method}/:

- metrics.csv / type_group_metrics.csv / density_group_metrics.csv:
  computed over the FULL manifest (validation + heldout mixed). This is
  DESCRIPTIVE ONLY -- it includes the same requirements used to select the
  threshold, so it is not a valid measure of generalization. Kept
  unmodified for continuity; do not treat it as a held-out test result.
- heldout_metrics.csv / heldout_type_group_metrics.csv /
  heldout_density_group_metrics.csv: computed over ONLY the Split=heldout
  rows -- genuinely disjoint from the validation requirements the
  threshold was calibrated on. This is the real generalization estimate.

Also computes an "Always-Positive" trivial baseline (predicts every pair
positive, no model/score involved) on the heldout split only, written to
results/pair_classification/{dataset}/Always-Positive/heldout_*.csv, as a
sanity floor: since the manifest is exactly 1:1 balanced per requirement
and Split is assigned per requirement, any complete-pairs subset (heldout,
any type, any density group) is itself exactly balanced, so Always-Positive
should read P=0.5, R=1.0, F1=0.667 on every non-empty group -- computed
from the actual manifest here, not hardcoded, so a violation of that
invariant would show up rather than being silently assumed.

Reads the threshold from
audit/pair_classification_validation/threshold_calibration.csv (frozen by
scripts/calibrate_pair_classification_threshold.py on validation pairs
only). Refuses to run for a method with no frozen threshold yet. Groups
with zero eligible requirements (e.g. eANCI's NFR) get an explicit
N/A (NaN) row rather than being omitted, matching the existing top-k
tables' convention in src/evaluation/metrics.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import VALID_TYPES, ensure_directory  # noqa: E402
from scripts.run_pair_classification import (  # noqa: E402
    ABLATION_LABELS,
    MANIFEST_DIR,
    PROPOSED_PAIR_CLASSIFIER_LABELS,
    output_dir_for,
)

AUDIT_DIR = PROJECT_ROOT / "audit" / "pair_classification_validation"
DENSITY_GROUPS = ("1-2", "3-10", ">10")
IR_NEURAL_METHODS = ("TFIDF", "BM25", "LSI", "Sentence-BERT", "CodeBERT")
LLM_METHODS = (
    ("Direct Prompting", "Generic RAG-LLM")
    + PROPOSED_PAIR_CLASSIFIER_LABELS
    + ABLATION_LABELS
)
ALWAYS_POSITIVE_LABEL = "Always-Positive"
DATASETS = ("eTour", "eANCI", "iTrust", "LibEST")


def load_frozen_threshold(method: str) -> float:
    path = AUDIT_DIR / "threshold_calibration.csv"
    if not path.is_file():
        raise SystemExit(
            f"No frozen threshold file at {path}; run "
            "scripts/calibrate_pair_classification_threshold.py first."
        )
    frame = pd.read_csv(path)
    row = frame.loc[frame["method"] == method]
    if row.empty:
        raise SystemExit(
            f"No frozen threshold for method={method!r} in {path}; "
            "run calibration for this method first."
        )
    return float(row.iloc[0]["threshold"])


def precision_recall_f1(labels: pd.Series, predicted_positive: pd.Series) -> tuple[float, float, float]:
    tp = int(((labels == 1) & predicted_positive).sum())
    fp = int(((labels == 0) & predicted_positive).sum())
    fn = int(((labels == 1) & ~predicted_positive).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def group_row(
    dataset_name: str,
    method: str,
    group_label: str,
    subset: pd.DataFrame,
    threshold: float,
    predicted_positive: pd.Series | None = None,
) -> dict[str, object]:
    """predicted_positive, if given, overrides Score>=threshold (used for
    the Always-Positive baseline, which has no Score column at all)."""
    requirements = subset["Req_ID"].nunique()
    if requirements == 0:
        return {
            "Dataset": dataset_name,
            "Method": method,
            "Group": group_label,
            "Requirements": 0,
            "Pairs": 0,
            "Threshold": threshold,
            "Precision": float("nan"),
            "Recall": float("nan"),
            "F1": float("nan"),
        }
    if predicted_positive is None:
        predicted_positive = subset["Score"] >= threshold
    precision, recall, f1 = precision_recall_f1(subset["Label"], predicted_positive)
    return {
        "Dataset": dataset_name,
        "Method": method,
        "Group": group_label,
        "Requirements": int(requirements),
        "Pairs": int(len(subset)),
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def group_rows_for(
    dataset_name: str,
    method: str,
    predictions: pd.DataFrame,
    threshold: float,
    predicted_positive: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (overall, type_wise, density_wise) row-sets for one
    predictions frame (already filtered to whatever scope -- full or
    heldout -- the caller wants)."""
    pp_overall = predicted_positive
    overall = pd.DataFrame(
        [group_row(dataset_name, method, "Overall", predictions, threshold, pp_overall)]
    )

    type_rows = []
    for type_label in VALID_TYPES:
        mask = predictions["Type"] == type_label
        subset = predictions[mask]
        pp = predicted_positive[mask] if predicted_positive is not None else None
        type_rows.append(group_row(dataset_name, method, type_label, subset, threshold, pp))
    type_wise = pd.DataFrame(type_rows)

    density_rows = []
    for density_label in DENSITY_GROUPS:
        mask = predictions["DensityGroup"] == density_label
        subset = predictions[mask]
        pp = predicted_positive[mask] if predicted_positive is not None else None
        density_rows.append(group_row(dataset_name, method, density_label, subset, threshold, pp))
    density_wise = pd.DataFrame(density_rows)

    return overall, type_wise, density_wise


def evaluate(dataset_name: str, method: str) -> None:
    threshold = load_frozen_threshold(method)
    output_dir = output_dir_for(dataset_name, method)
    predictions_path = output_dir / "predictions.csv"
    if not predictions_path.is_file():
        raise SystemExit(
            f"Missing predictions: {predictions_path}. Run "
            "scripts/run_pair_classification.py first."
        )
    predictions = pd.read_csv(predictions_path)

    # Full scope (validation + heldout mixed) -- descriptive only.
    overall, type_wise, density_wise = group_rows_for(dataset_name, method, predictions, threshold)
    overall.to_csv(output_dir / "metrics.csv", index=False)
    type_wise.to_csv(output_dir / "type_group_metrics.csv", index=False)
    density_wise.to_csv(output_dir / "density_group_metrics.csv", index=False)

    # Heldout-only scope -- the genuine generalization estimate.
    heldout = predictions[predictions["Split"] == "heldout"]
    h_overall, h_type_wise, h_density_wise = group_rows_for(dataset_name, method, heldout, threshold)
    h_overall.to_csv(output_dir / "heldout_metrics.csv", index=False)
    h_type_wise.to_csv(output_dir / "heldout_type_group_metrics.csv", index=False)
    h_density_wise.to_csv(output_dir / "heldout_density_group_metrics.csv", index=False)

    print(
        f"{dataset_name}/{method}: threshold={threshold:.4f} "
        f"full F1={overall.iloc[0]['F1']:.4f} (descriptive) "
        f"heldout F1={h_overall.iloc[0]['F1']:.4f} (generalization) -> {output_dir}"
    )


def evaluate_llm_heldout_only(dataset_name: str, method: str) -> None:
    """Evaluate LLM predictions produced with --split heldout.

    Reads predictions from the isolated _split_heldout subfolder and
    checks that every row is held out and each PairGroupID has exactly
    one positive and one negative, including with-replacement sampling.
    Writes heldout_metrics.csv, heldout_type_group_metrics.csv, and
    heldout_density_group_metrics.csv to the method's top-level output
    directory for aggregation. No full-manifest metrics are produced.
    """
    threshold = load_frozen_threshold(method)
    heldout_dir = output_dir_for(dataset_name, method, split="heldout")
    predictions_path = heldout_dir / "predictions.csv"
    if not predictions_path.is_file():
        raise SystemExit(
            f"Missing heldout predictions: {predictions_path}. Run "
            f'scripts/run_pair_classification.py --dataset {dataset_name} '
            f'--method "{method}" --split heldout --live-api first.'
        )
    predictions = pd.read_csv(predictions_path)

    non_heldout = predictions[predictions["Split"] != "heldout"]
    if not non_heldout.empty:
        raise SystemExit(
            f"{predictions_path}: {len(non_heldout)} row(s) are not "
            "Split=heldout (should be impossible for a _split_heldout output)."
        )

    # groupby(...).size() only enumerates combinations that actually
    # appear -- a PairGroupID with just one row (e.g. its negative is
    # missing entirely) would pass a "count per (PairGroupID, Label) == 1"
    # check trivially, since the missing label simply never shows up.
    # Require exactly 2 rows per group AND that the two rows' labels are
    # the set {0, 1} -- catches a missing row, an extra row, and a
    # duplicated label (two positives or two negatives) alike.
    per_group = predictions.groupby("PairGroupID")
    group_sizes = per_group.size()
    wrong_size = group_sizes[group_sizes != 2]
    if not wrong_size.empty:
        raise SystemExit(
            f"{predictions_path}: PairGroupID(s) without exactly 2 rows "
            f"(should be impossible): {wrong_size.index.tolist()[:5]}"
        )
    label_sets = per_group["Label"].apply(lambda labels: frozenset(int(x) for x in labels))
    wrong_labels = label_sets[label_sets != frozenset({0, 1})]
    if not wrong_labels.empty:
        raise SystemExit(
            f"{predictions_path}: PairGroupID(s) without exactly one "
            "positive and one negative row (should be impossible): "
            f"{wrong_labels.index.tolist()[:5]}"
        )

    output_dir = ensure_directory(output_dir_for(dataset_name, method))
    overall, type_wise, density_wise = group_rows_for(dataset_name, method, predictions, threshold)
    overall.to_csv(output_dir / "heldout_metrics.csv", index=False)
    type_wise.to_csv(output_dir / "heldout_type_group_metrics.csv", index=False)
    density_wise.to_csv(output_dir / "heldout_density_group_metrics.csv", index=False)
    print(
        f"{dataset_name}/{method}: threshold={threshold:.4f} "
        f"heldout F1={overall.iloc[0]['F1']:.4f} (generalization) -> {output_dir}"
    )


def evaluate_always_positive_heldout(dataset_name: str) -> None:
    """Trivial baseline, no score/threshold involved: every pair is
    predicted positive. Uses the manifest directly (not any method's
    predictions.csv), heldout rows only."""
    manifest_path = MANIFEST_DIR / f"{dataset_name}.csv"
    manifest = pd.read_csv(manifest_path)
    heldout = manifest[manifest["Split"] == "heldout"]
    predicted_positive = pd.Series(True, index=heldout.index)

    overall, type_wise, density_wise = group_rows_for(
        dataset_name, ALWAYS_POSITIVE_LABEL, heldout, float("nan"), predicted_positive
    )
    output_dir = ensure_directory(output_dir_for(dataset_name, ALWAYS_POSITIVE_LABEL))
    overall.to_csv(output_dir / "heldout_metrics.csv", index=False)
    type_wise.to_csv(output_dir / "heldout_type_group_metrics.csv", index=False)
    density_wise.to_csv(output_dir / "heldout_density_group_metrics.csv", index=False)
    print(
        f"{dataset_name}/{ALWAYS_POSITIVE_LABEL}: heldout "
        f"P={overall.iloc[0]['Precision']:.4f} "
        f"R={overall.iloc[0]['Recall']:.4f} "
        f"F1={overall.iloc[0]['F1']:.4f} -> {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--method", default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    methods = IR_NEURAL_METHODS + LLM_METHODS if args.method == "all" else (args.method,)
    for dataset_name in datasets:
        for method in methods:
            if method in LLM_METHODS:
                evaluate_llm_heldout_only(dataset_name, method)
            else:
                evaluate(dataset_name, method)
        evaluate_always_positive_heldout(dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
