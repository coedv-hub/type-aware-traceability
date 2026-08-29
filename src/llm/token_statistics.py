"""Token and cost accounting for estimated and actual LLM usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TokenRecord:
    dataset: str
    baseline: str
    provider: str
    model: str
    requirement_id: str
    code_file: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    token_source: str
    cache_hit: bool


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_per_million_usd: float,
    output_per_million_usd: float,
) -> float:
    return (
        prompt_tokens * input_per_million_usd
        + completion_tokens * output_per_million_usd
    ) / 1_000_000


def records_frame(records: list[TokenRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.__dict__ for record in records])


def summarize_records(records: list[TokenRecord]) -> dict[str, Any]:
    if not records:
        return {
            "pairs": 0,
            "requirements": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "average_tokens_per_requirement": 0.0,
            "average_tokens_per_pair": 0.0,
        }
    # Code summarisation is cached under the synthetic ``code_summary``
    # namespace.  It is an implementation detail, not an evaluated
    # requirement, so it must not inflate the experiment-level requirement
    # count or make the per-requirement token figure misleading.
    requirement_count = len(
        {
            (record.dataset, record.requirement_id)
            for record in records
            if record.requirement_id and record.requirement_id != "code_summary"
        }
    )
    total_tokens = sum(record.total_tokens for record in records)
    return {
        "pairs": len(records),
        "requirements": requirement_count,
        "prompt_tokens": sum(record.prompt_tokens for record in records),
        "completion_tokens": sum(record.completion_tokens for record in records),
        "total_tokens": total_tokens,
        "estimated_cost_usd": sum(
            record.estimated_cost_usd for record in records
        ),
        "average_tokens_per_requirement": (
            total_tokens / requirement_count if requirement_count else 0.0
        ),
        "average_tokens_per_pair": total_tokens / len(records),
    }
