"""Pair-level and ranking metrics for traceability recovery."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


RankedResults = dict[str, list[tuple[str, float]]]
GroundTruth = dict[str, set[str]]


def precision_recall_f1(
    predicted: set[tuple[str, str]], gold: set[tuple[str, str]]
) -> tuple[float, float, float]:
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _eligible_requirements(
    ranked: RankedResults, gold: GroundTruth, requirement_ids: Iterable[str]
) -> list[str]:
    return [
        req_id
        for req_id in requirement_ids
        if req_id in ranked and req_id in gold and bool(gold[req_id])
    ]


def mean_reciprocal_rank(
    ranked: RankedResults, gold: GroundTruth, requirement_ids: Iterable[str]
) -> float:
    reciprocal_ranks: list[float] = []
    for req_id in _eligible_requirements(ranked, gold, requirement_ids):
        relevant = gold[req_id]
        rank = next(
            (
                index
                for index, (code_id, _) in enumerate(ranked[req_id], start=1)
                if code_id in relevant
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def precision_at_k(
    ranked: RankedResults,
    gold: GroundTruth,
    requirement_ids: Iterable[str],
    k: int,
) -> float:
    values: list[float] = []
    for req_id in _eligible_requirements(ranked, gold, requirement_ids):
        retrieved = [code_id for code_id, _ in ranked[req_id][:k]]
        values.append(len(set(retrieved) & gold[req_id]) / k)
    return sum(values) / len(values) if values else 0.0


def recall_at_k(
    ranked: RankedResults,
    gold: GroundTruth,
    requirement_ids: Iterable[str],
    k: int,
) -> float:
    values: list[float] = []
    for req_id in _eligible_requirements(ranked, gold, requirement_ids):
        retrieved = {code_id for code_id, _ in ranked[req_id][:k]}
        values.append(len(retrieved & gold[req_id]) / len(gold[req_id]))
    return sum(values) / len(values) if values else 0.0


def evaluate_rankings(
    ranked: RankedResults,
    gold: GroundTruth,
    requirement_types: dict[str, str],
    binary_top_k: int = 10,
    cutoffs: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """Compute overall and FR/NFR/Mixed metrics using a top-k binary protocol."""
    rows: list[dict[str, object]] = []
    groups = {
        "Overall": list(requirement_types),
        "FR": [
            req_id for req_id, req_type in requirement_types.items() if req_type == "FR"
        ],
        "NFR": [
            req_id
            for req_id, req_type in requirement_types.items()
            if req_type == "NFR"
        ],
        "Mixed": [
            req_id
            for req_id, req_type in requirement_types.items()
            if req_type == "Mixed"
        ],
    }

    for group, group_ids in groups.items():
        eligible = _eligible_requirements(ranked, gold, group_ids)
        if not eligible:
            row: dict[str, object] = {
                "Group": group,
                "Requirements": 0,
                "Precision": float("nan"),
                "Recall": float("nan"),
                "F1": float("nan"),
                "MRR": float("nan"),
            }
            for k in cutoffs:
                row[f"P@{k}"] = float("nan")
                row[f"R@{k}"] = float("nan")
            rows.append(row)
            continue
        predicted = {
            (req_id, code_id)
            for req_id in eligible
            for code_id, _ in ranked[req_id][:binary_top_k]
        }
        gold_pairs = {
            (req_id, code_id) for req_id in eligible for code_id in gold[req_id]
        }
        precision, recall, f1 = precision_recall_f1(predicted, gold_pairs)
        row: dict[str, object] = {
            "Group": group,
            "Requirements": len(eligible),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MRR": mean_reciprocal_rank(ranked, gold, eligible),
        }
        for k in cutoffs:
            row[f"P@{k}"] = precision_at_k(ranked, gold, eligible, k)
            row[f"R@{k}"] = recall_at_k(ranked, gold, eligible, k)
        rows.append(row)
    return pd.DataFrame(rows)
