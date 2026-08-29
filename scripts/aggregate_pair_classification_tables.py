#!/usr/bin/env python3
"""Aggregate per-dataset pair-classification metrics into tables/pair_classification/.

Reads results/pair_classification/{dataset}/{method}/*.csv (written by
scripts/evaluate_pair_classification.py) and concatenates across all 4
active datasets. Writes two families of table:

- overall_metrics.csv / type_wise_metrics.csv / density_group_metrics.csv:
  the FULL-manifest (validation+heldout mixed) numbers. DESCRIPTIVE ONLY --
  see the module docstring in evaluate_pair_classification.py. Existing
  files from prior runs are only ever overwritten with a re-derivation of
  the same descriptive scope, never replaced by heldout numbers.
- heldout_overall_metrics.csv / heldout_type_wise_metrics.csv /
  heldout_density_group_metrics.csv: the genuine held-out generalization
  numbers, IR/neural methods plus the Always-Positive trivial baseline.

Does not touch the existing tables/*.csv (top-k) files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_directory  # noqa: E402
from scripts.run_pair_classification import (  # noqa: E402
    ABLATION_LABELS,
    ABLATION_WO_BI_ALIGNMENT_LABEL,
    ABLATION_WO_DEPENDENCY_LABEL,
    ABLATION_WO_TYPE_LABEL,
    ABLATION_WO_VERIFICATION_LABEL,
    PROPOSED_PAIR_CLASSIFIER_V3_LABEL,
    output_dir_for,
)
from scripts.evaluate_pair_classification import (  # noqa: E402
    ALWAYS_POSITIVE_LABEL,
    LLM_METHODS,
    PROPOSED_PAIR_CLASSIFIER_LABELS,
    load_frozen_threshold,
)

OUTPUT_DIR = PROJECT_ROOT / "tables" / "pair_classification"
# Four-dataset CALIBRATION scope -- the descriptive (full-manifest) tables
# stay scoped to exactly these 4, since "descriptive" specifically means
# "includes the same requirements the threshold was calibrated on", which
# is meaningless for a dataset with no validation split at all.
DATASETS = ("eTour", "eANCI", "iTrust", "LibEST")
# Industrial (see scripts/run_pair_classification.py's
# HELDOUT_ONLY_DATASETS) is never part of calibration, but its heldout
# generalization result IS worth reporting once it exists -- this is the
# five-dataset REPORTING scope, used only for the heldout_*.csv tables
# below, and only via optional_datasets so a missing Industrial evaluation
# never blocks the other 4 datasets' tables.
HELDOUT_ONLY_DATASETS = ("Industrial",)
IR_NEURAL_METHODS = ("TFIDF", "BM25", "LSI", "Sentence-BERT", "CodeBERT")


def collect(
    filename: str,
    methods: tuple[str, ...],
    *,
    optional_methods: tuple[str, ...] = (),
    datasets: tuple[str, ...] | None = None,
    optional_datasets: tuple[str, ...] = (),
) -> pd.DataFrame:
    """optional_methods (e.g. the two Proposed-PairClassifier prompt
    variants) are included only if their file actually exists -- v1 and v2
    are alternatives, not both required, so the aggregate must not force a
    dataset to have run BOTH before it can produce a table. Every other
    method remains strictly required, as before: missing data for those
    still raises immediately.

    optional_datasets (Industrial) works the same way at the dataset
    level: its file is included if present, silently skipped if not --
    running the 4 calibration datasets must never be blocked on Industrial
    having been evaluated yet, and vice versa.

    datasets defaults to None (resolved to the module-level DATASETS
    inside the function body, not bound as a literal default value) so
    that tests patching aggregate_pair_classification_tables.DATASETS
    still take effect on calls that omit this argument."""
    if datasets is None:
        datasets = DATASETS
    frames = []
    for dataset_name in datasets:
        for method in methods:
            path = output_dir_for(dataset_name, method) / filename
            if not path.is_file():
                if method in optional_methods or dataset_name in optional_datasets:
                    continue
                raise SystemExit(
                    f"Missing {path}; run scripts/evaluate_pair_classification.py "
                    f"for {dataset_name}/{method} first."
                )
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


TABLE6_CONFIGURATIONS = (
    ("Full (GPT-4.1)", PROPOSED_PAIR_CLASSIFIER_V3_LABEL),
    ("w/o Type", ABLATION_WO_TYPE_LABEL),
    ("w/o Dependency", ABLATION_WO_DEPENDENCY_LABEL),
    ("w/o Bi-Align.", ABLATION_WO_BI_ALIGNMENT_LABEL),
    ("w/o Verification", ABLATION_WO_VERIFICATION_LABEL),
)


def _pooled_heldout_predictions(method: str, datasets: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for dataset_name in datasets:
        path = output_dir_for(dataset_name, method, split="heldout") / "predictions.csv"
        if not path.is_file():
            continue
        predictions = pd.read_csv(path)
        frames.append(predictions[predictions["Split"] == "heldout"])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pooled_type_wise_f1(
    method: str, datasets: tuple[str, ...]
) -> dict[str, float | None]:
    """Genuine pooled-pairs F1 per requirement Type (FR/NFR/Mixed): pool
    heldout predictions across `datasets` BEFORE computing metrics, rather
    than averaging each dataset's own F1 -- those are not equivalent when
    heldout sizes differ across datasets. Mirrors
    calibrate_pair_classification_threshold.py's own pooling methodology
    (concatenate raw rows, then compute P/R/F1 once), matching Table 4's
    "pooled held-out pairs" convention. Returns None per type with no
    eligible pairs pooled (e.g. eANCI contributes no NFR rows) rather than
    a misleading 0.0."""
    pooled = _pooled_heldout_predictions(method, datasets)
    result: dict[str, float | None] = {"FR": None, "NFR": None, "Mixed": None}
    if pooled.empty:
        return result
    try:
        threshold = load_frozen_threshold(method)
    except SystemExit:
        # Not calibrated yet (e.g. an ablation with no
        # threshold_calibration.csv row) -- leave every type None rather
        # than crashing the whole aggregation run.
        return result
    predicted_positive = pooled["Score"] >= threshold
    for type_label in result:
        mask = pooled["Type"] == type_label
        if not mask.any():
            continue
        labels = pooled.loc[mask, "Label"]
        pp = predicted_positive[mask]
        tp = int(((labels == 1) & pp).sum())
        fp = int(((labels == 0) & pp).sum())
        fn = int(((labels == 1) & ~pp).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        result[type_label] = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
    return result


def avg_f1_across_datasets(method: str, datasets: tuple[str, ...]) -> float | None:
    """Simple unweighted mean of each dataset's own heldout Overall F1 --
    matches Table 5's rightmost 'Avg. F1' column convention (average of
    per-dataset F1s), used for Table 6's 'Avg.' column. Returns None if no
    dataset has been evaluated for this method yet."""
    scores = []
    for dataset_name in datasets:
        path = output_dir_for(dataset_name, method) / "heldout_metrics.csv"
        if not path.is_file():
            continue
        overall = pd.read_csv(path)
        row = overall.loc[overall["Group"] == "Overall"]
        if row.empty or pd.isna(row.iloc[0]["F1"]):
            continue
        scores.append(float(row.iloc[0]["F1"]))
    return sum(scores) / len(scores) if scores else None


def build_ablation_summary_table(
    configurations: tuple[tuple[str, str], ...] = TABLE6_CONFIGURATIONS,
    datasets: tuple[str, ...] = (),
    baseline_label: str = "Full (GPT-4.1)",
) -> pd.DataFrame:
    """Table 6 shape: Configuration | Avg. | Delta | FR | NFR | Mixed.
    `configurations` is (display_name, method_label) pairs; `baseline_label`
    identifies which row's Avg. anchors every row's Delta = Avg. -
    baseline Avg. `datasets` defaults to the 5-dataset reporting scope
    (DATASETS + HELDOUT_ONLY_DATASETS) resolved at call time, so tests
    patching those module-level constants still take effect."""
    if not datasets:
        datasets = DATASETS + HELDOUT_ONLY_DATASETS
    per_method_avg: dict[str, float | None] = {
        method: avg_f1_across_datasets(method, datasets) for _, method in configurations
    }
    baseline_avg = next(
        (per_method_avg[method] for name, method in configurations if name == baseline_label),
        None,
    )
    rows = []
    for display_name, method in configurations:
        avg = per_method_avg[method]
        delta = None if avg is None or baseline_avg is None else round(avg - baseline_avg, 4)
        type_f1 = pooled_type_wise_f1(method, datasets)
        rows.append(
            {
                "Configuration": display_name,
                "Avg.": round(avg, 4) if avg is not None else None,
                "Delta": delta,
                "FR": round(type_f1["FR"], 4) if type_f1["FR"] is not None else None,
                "NFR": round(type_f1["NFR"], 4) if type_f1["NFR"] is not None else None,
                "Mixed": round(type_f1["Mixed"], 4) if type_f1["Mixed"] is not None else None,
            }
        )
    return pd.DataFrame(rows, columns=["Configuration", "Avg.", "Delta", "FR", "NFR", "Mixed"])


def main() -> int:
    ensure_directory(OUTPUT_DIR)

    # Full scope -- descriptive only (see module docstring).
    overall = collect("metrics.csv", IR_NEURAL_METHODS)
    overall_path = OUTPUT_DIR / "overall_metrics.csv"
    overall.to_csv(overall_path, index=False)
    print(f"Wrote {overall_path} ({len(overall)} rows) [descriptive, full manifest]")

    type_wise = collect("type_group_metrics.csv", IR_NEURAL_METHODS)
    type_path = OUTPUT_DIR / "type_wise_metrics.csv"
    type_wise.to_csv(type_path, index=False)
    print(f"Wrote {type_path} ({len(type_wise)} rows) [descriptive, full manifest]")

    density = collect("density_group_metrics.csv", IR_NEURAL_METHODS)
    density_path = OUTPUT_DIR / "density_group_metrics.csv"
    density.to_csv(density_path, index=False)
    print(f"Wrote {density_path} ({len(density)} rows) [descriptive, full manifest]")

    # Heldout scope -- genuine generalization estimate, incl. Always-Positive
    # and the LLM-based methods (which only ever have heldout-scoped
    # predictions -- see evaluate_pair_classification.evaluate_llm_heldout_only).
    # The two Proposed-PairClassifier prompt variants (v1/v2) are
    # alternatives -- only whichever has actually been run is included,
    # never both required.
    heldout_methods = IR_NEURAL_METHODS + LLM_METHODS + (ALWAYS_POSITIVE_LABEL,)
    heldout_overall = collect(
        "heldout_metrics.csv", heldout_methods,
        optional_methods=PROPOSED_PAIR_CLASSIFIER_LABELS + ABLATION_LABELS,
        datasets=DATASETS + HELDOUT_ONLY_DATASETS, optional_datasets=HELDOUT_ONLY_DATASETS,
    )
    heldout_overall_path = OUTPUT_DIR / "heldout_overall_metrics.csv"
    heldout_overall.to_csv(heldout_overall_path, index=False)
    print(f"Wrote {heldout_overall_path} ({len(heldout_overall)} rows) [heldout, generalization]")

    heldout_type_wise = collect(
        "heldout_type_group_metrics.csv", heldout_methods, optional_methods=PROPOSED_PAIR_CLASSIFIER_LABELS + ABLATION_LABELS,
        datasets=DATASETS + HELDOUT_ONLY_DATASETS, optional_datasets=HELDOUT_ONLY_DATASETS,
    )
    heldout_type_path = OUTPUT_DIR / "heldout_type_wise_metrics.csv"
    heldout_type_wise.to_csv(heldout_type_path, index=False)
    print(f"Wrote {heldout_type_path} ({len(heldout_type_wise)} rows) [heldout, generalization]")

    heldout_density = collect(
        "heldout_density_group_metrics.csv", heldout_methods, optional_methods=PROPOSED_PAIR_CLASSIFIER_LABELS + ABLATION_LABELS,
        datasets=DATASETS + HELDOUT_ONLY_DATASETS, optional_datasets=HELDOUT_ONLY_DATASETS,
    )
    heldout_density_path = OUTPUT_DIR / "heldout_density_group_metrics.csv"
    heldout_density.to_csv(heldout_density_path, index=False)
    print(f"Wrote {heldout_density_path} ({len(heldout_density)} rows) [heldout, generalization]")

    # Table 6 shape (Full + 4 ablations): 5-dataset Avg. F1, Delta vs Full,
    # and pooled FR/NFR/Mixed F1. Rows with no evaluated predictions yet
    # are written with None/NaN cells rather than being omitted, so the
    # table's shape is stable before every ablation has been run.
    ablation_summary = build_ablation_summary_table()
    ablation_summary_path = OUTPUT_DIR / "ablation_summary_table6.csv"
    ablation_summary.to_csv(ablation_summary_path, index=False)
    print(f"Wrote {ablation_summary_path} ({len(ablation_summary)} rows) [Table 6 shape]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
