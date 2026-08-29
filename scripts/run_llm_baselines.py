#!/usr/bin/env python3
"""Run or dry-run the Direct Prompting and Generic RAG-LLM baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.llm import create_llm_baseline
from src.data.loader import DatasetLoader, TraceDataset
from src.llm import DeepSeekClient, LLMClient, OpenAIClient, RateLimiter
from src.llm.cache import CacheIdentity, LLMCache
from src.llm.token_statistics import (
    TokenRecord,
    estimate_cost,
    records_frame,
    summarize_records,
)
from src.evaluation.metrics import evaluate_rankings
from src.utils import ensure_directory, load_config, write_json


METHODS = ("direct", "rag")
METHOD_LABELS = {"direct": "Direct Prompting", "rag": "Generic RAG-LLM"}
RETRIEVERS = ("tfidf", "bm25", "lsi", "sentencebert", "codebert", "hybrid", "hybrid3")
IR_RETRIEVERS = {"tfidf", "bm25", "lsi"}
HYBRID_RETRIEVERS = {
    "hybrid": ("sentencebert", "lsi"),
    # Three-way reciprocal rank fusion of S-BERT, LSI, and BM25.
    "hybrid3": ("sentencebert", "lsi", "bm25"),
}
RRF_CONSTANT = 60
DEFAULT_STAGE4B_SAMPLE_SIZE = 1
DEFAULT_LLM_TOP_K = 10
GROUPS = ("Overall", "FR", "NFR", "Mixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="eTour")
    parser.add_argument(
        "--method",
        choices=[*METHODS, "all"],
        default="direct",
        help="Run one baseline (default: direct); use all only when intentional.",
    )
    parser.add_argument(
        "--provider", choices=["openai", "deepseek"], default="openai"
    )
    parser.add_argument(
        "--retriever",
        choices=RETRIEVERS,
        default="sentencebert",
        help="Existing Stage 2/3 ranking used to select LLM candidates.",
    )
    parser.add_argument(
        "--llm-top-k",
        type=int,
        default=DEFAULT_LLM_TOP_K,
        help="Evaluate only the top-k code files from the retriever ranking.",
    )
    parser.add_argument(
        "--candidate-pool-k",
        type=int,
        default=None,
        help=(
            "Retriever candidate-pool size shared with Proposed Framework. "
            "Defaults to --llm-top-k for backward compatibility."
        ),
    )
    parser.add_argument("--model", help="Override the provider's default model.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Sample evaluation requirements. In paper mode, omitting "
            "this option evaluates the full dataset; otherwise the default is 1."
        ),
    )
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="Write rankings, metrics, raw predictions, token/cost data, and paper tables.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Explicitly request resume mode. Resume is also automatic: every "
            "completed pair is read from the persistent cache."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate prompts, exercise cache, and estimate cost without API calls.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "datasets.yaml"),
    )
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--rpm-limit", type=int, default=10)
    parser.add_argument("--tpm-limit", type=int, default=20_000)
    parser.add_argument("--min-call-interval", type=float, default=3.0)
    return parser.parse_args()


def select_dataset(loader: DatasetLoader, requested: str) -> str:
    lookup = {name.casefold(): name for name in loader.dataset_names}
    if requested.casefold() not in lookup:
        raise SystemExit(
            f"Unknown dataset {requested!r}; choose from {loader.dataset_names}."
        )
    return lookup[requested.casefold()]


def resolve_candidate_pool_k(args: argparse.Namespace) -> int:
    candidate_pool_k = (
        args.llm_top_k if args.candidate_pool_k is None else args.candidate_pool_k
    )
    if candidate_pool_k < 1:
        raise SystemExit("--candidate-pool-k must be at least 1.")
    return candidate_pool_k


def llm_output_dir_for(
    results_root: Path,
    dataset: str,
    provider: str,
    model: str,
    retriever: str,
    llm_top_k: int,
    candidate_pool_k: int,
) -> Path:
    base = (
        results_root
        / dataset
        / provider
        / safe_name(model)
        / "retriever_top_k"
        / retriever
    )
    if candidate_pool_k == llm_top_k:
        return base / f"top_{llm_top_k}"
    return base / f"pool_{candidate_pool_k}"


def sample_requirements(
    dataset: TraceDataset,
    sample_size: int | None,
    seed: int,
    *,
    paper_mode: bool = False,
) -> list[str]:
    eligible = sorted(dataset.evaluation_requirements)
    if sample_size is None:
        if paper_mode:
            return eligible
        sample_size = DEFAULT_STAGE4B_SAMPLE_SIZE
    if sample_size < 1:
        raise SystemExit("--sample-size must be at least 1.")
    if not paper_mode and sample_size not in (1, 2, 5):
        raise SystemExit("Stage 4b supports --sample-size 1, 2, or 5.")
    if sample_size > len(eligible):
        raise SystemExit(
            f"{dataset.name} has only {len(eligible)} evaluation requirements."
        )
    return random.Random(seed).sample(eligible, sample_size)


def create_client(
    provider: str,
    model: str | None,
    config: dict[str, Any],
    *,
    max_retries: int,
    rpm_limit: int,
    tpm_limit: int,
    min_call_interval: float,
) -> LLMClient:
    llm_config = config["llm"]
    item = llm_config["providers"][provider]
    client_class = OpenAIClient if provider == "openai" else DeepSeekClient
    return client_class(
        model=model or str(item["default_model"]),
        api_key_env=str(item["api_key_env"]),
        base_url=str(item["base_url"]),
        max_completion_tokens=int(llm_config["max_completion_tokens"]),
        chars_per_token=float(
            llm_config["token_estimation_chars_per_token"]
        ),
        max_retries=max_retries,
        rate_limiter=RateLimiter(
            requests_per_minute=rpm_limit,
            tokens_per_minute=tpm_limit,
            min_interval_seconds=min_call_interval,
        ),
    )


def pricing_for(model: str, config: dict[str, Any]) -> tuple[float, float]:
    try:
        item = config["llm"]["pricing"][model]
    except KeyError as exc:
        raise SystemExit(
            f"No pricing configured for model {model!r}; add it under llm.pricing."
        ) from exc
    return (
        float(item["input_per_million_usd"]),
        float(item["output_per_million_usd"]),
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_retriever_candidates(
    project_root: Path,
    config: dict[str, Any],
    dataset: TraceDataset,
    requirement_ids: list[str],
    retriever: str,
    top_k: int,
) -> tuple[dict[str, dict[str, tuple[int, float]]], Path]:
    if top_k < 1:
        raise SystemExit("--llm-top-k must be at least 1.")
    if retriever in HYBRID_RETRIEVERS:
        return load_hybrid_retriever_candidates(
            project_root,
            config,
            dataset,
            requirement_ids,
            retriever,
            top_k,
        )
    result_key = (
        "results_dir" if retriever in IR_RETRIEVERS else "neural_results_dir"
    )
    ranking_path = (
        project_root
        / config["output"][result_key]
        / dataset.name
        / f"{retriever}_ranked_results.csv"
    )
    if not ranking_path.is_file():
        stage = "Stage 2 IR" if retriever in IR_RETRIEVERS else "Stage 3 neural"
        raise SystemExit(
            f"Retriever ranking not found: {ranking_path}\n"
            f"Run the corresponding {stage} baseline first; Stage 4 will not "
            "recompute retriever rankings."
        )
    ranking = pd.read_csv(ranking_path)
    required_columns = {"Req_ID", "Rank", "Code_File", "Score"}
    missing = required_columns - set(ranking.columns)
    if missing:
        raise SystemExit(
            f"Retriever ranking {ranking_path} is missing columns: "
            f"{sorted(missing)}."
        )
    candidates: dict[str, dict[str, tuple[int, float]]] = {}
    limit = min(top_k, len(dataset.code_files))
    for requirement_id in requirement_ids:
        rows = (
            ranking.loc[ranking["Req_ID"].astype(str).eq(requirement_id)]
            .sort_values(["Rank", "Code_File"])
            .head(limit)
        )
        if len(rows) < limit:
            raise SystemExit(
                f"Retriever ranking {ranking_path} has only {len(rows)} "
                f"candidate(s) for {requirement_id}; expected {limit}. "
                "Run the corresponding baseline first."
            )
        code_ids = rows["Code_File"].astype(str).tolist()
        unknown = sorted(set(code_ids) - set(dataset.code_files))
        if unknown:
            raise SystemExit(
                f"Retriever ranking {ranking_path} contains unknown code files "
                f"for {requirement_id}: {unknown[:5]}."
            )
        if len(set(code_ids)) != len(code_ids):
            raise SystemExit(
                f"Retriever ranking {ranking_path} contains duplicate candidates "
                f"for {requirement_id}."
            )
        candidates[requirement_id] = {
            str(row.Code_File): (int(row.Rank), float(row.Score))
            for row in rows.itertuples(index=False)
        }
    return candidates, ranking_path


def load_hybrid_retriever_candidates(
    project_root: Path,
    config: dict[str, Any],
    dataset: TraceDataset,
    requirement_ids: list[str],
    retriever: str,
    top_k: int,
) -> tuple[dict[str, dict[str, tuple[int, float]]], Path]:
    """Build a global RRF candidate pool from precomputed retriever rankings."""
    components = HYBRID_RETRIEVERS[retriever]
    rankings = {
        component: load_ranking_frame(project_root, config, dataset, component)
        for component in components
    }
    candidates: dict[str, dict[str, tuple[int, float]]] = {}
    limit = min(top_k, len(dataset.code_files))
    for requirement_id in requirement_ids:
        fused_scores: dict[str, float] = {}
        best_ranks: dict[str, int] = {}
        for frame in rankings.values():
            rows = frame.loc[frame["Req_ID"].astype(str).eq(requirement_id)]
            if rows.empty:
                raise SystemExit(
                    f"Retriever rankings for {retriever} have no candidates "
                    f"for {requirement_id}."
                )
            for row in rows.itertuples(index=False):
                code_id = str(row.Code_File)
                rank = int(row.Rank)
                fused_scores[code_id] = fused_scores.get(code_id, 0.0) + (
                    1.0 / (RRF_CONSTANT + rank)
                )
                best_ranks[code_id] = min(best_ranks.get(code_id, rank), rank)
        ordered = sorted(
            fused_scores.items(),
            key=lambda item: (-item[1], best_ranks[item[0]], item[0]),
        )[:limit]
        if len(ordered) < limit:
            raise SystemExit(
                f"Hybrid retriever {retriever} has only {len(ordered)} "
                f"candidate(s) for {requirement_id}; expected {limit}."
            )
        candidates[requirement_id] = {
            code_id: (rank, float(score))
            for rank, (code_id, score) in enumerate(ordered, start=1)
        }
    source = Path(f"hybrid:{'+'.join(components)}")
    return candidates, source


def load_ranking_frame(
    project_root: Path,
    config: dict[str, Any],
    dataset: TraceDataset,
    retriever: str,
) -> pd.DataFrame:
    result_key = (
        "results_dir" if retriever in IR_RETRIEVERS else "neural_results_dir"
    )
    ranking_path = (
        project_root
        / config["output"][result_key]
        / dataset.name
        / f"{retriever}_ranked_results.csv"
    )
    if not ranking_path.is_file():
        stage = "Stage 2 IR" if retriever in IR_RETRIEVERS else "Stage 3 neural"
        raise SystemExit(
            f"Retriever ranking not found: {ranking_path}\n"
            f"Run the corresponding {stage} baseline first; candidate-pool "
            "fusion will not recompute retriever rankings."
        )
    ranking = pd.read_csv(ranking_path)
    required_columns = {"Req_ID", "Rank", "Code_File", "Score"}
    missing = required_columns - set(ranking.columns)
    if missing:
        raise SystemExit(
            f"Retriever ranking {ranking_path} is missing columns: "
            f"{sorted(missing)}."
        )
    unknown = sorted(set(ranking["Code_File"].astype(str)) - set(dataset.code_files))
    if unknown:
        raise SystemExit(
            f"Retriever ranking {ranking_path} contains unknown code files: "
            f"{unknown[:5]}."
        )
    return ranking


@dataclass(frozen=True)
class RunBudget:
    api_calls: int
    cache_hits: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    miss_cost_by_digest: dict[str, float]


@dataclass
class RunProgress:
    total_pairs: int
    remaining_calls: int
    remaining_cost_usd: float
    started_at: float
    completed_pairs: int = 0
    cache_hits: int = 0

    def record(self, cache_hit: bool, estimated_miss_cost: float) -> None:
        self.completed_pairs += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.remaining_calls = max(0, self.remaining_calls - 1)
            self.remaining_cost_usd = max(
                0.0, self.remaining_cost_usd - estimated_miss_cost
            )

    def estimated_remaining_seconds(self) -> float | None:
        if self.completed_pairs == 0:
            return None
        elapsed = time.monotonic() - self.started_at
        return elapsed / self.completed_pairs * (
            self.total_pairs - self.completed_pairs
        )


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "calculating..."
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:d}m {seconds:02d}s"


def estimate_run_budget(
    dataset: TraceDataset,
    requirement_ids: list[str],
    methods: tuple[str, ...],
    client: LLMClient,
    cache: LLMCache,
    input_price: float,
    output_price: float,
    candidates: dict[str, dict[str, tuple[int, float]]],
    allow_estimated_cache: bool = False,
) -> RunBudget:
    prompt_tokens = 0
    completion_tokens = 0
    cache_hits = 0
    miss_cost_by_digest: dict[str, float] = {}
    for method in methods:
        baseline = create_llm_baseline(
            method,
            client,
            cache,
            dataset,
            estimate_only=False,
            candidates=candidates,
        )
        for requirement_id in requirement_ids:
            for code_file, prompt in baseline.build_prompts(
                requirement_id
            ).items():
                identity = CacheIdentity(
                    model=client.model,
                    prompt=prompt,
                    dataset=dataset.name,
                    requirement_id=requirement_id,
                    code_file=code_file,
                    provider=client.provider,
                )
                if cache.get(
                    identity, allow_estimated=allow_estimated_cache
                ) is not None:
                    cache_hits += 1
                    continue
                pair_prompt_tokens = client.estimate_tokens(prompt)
                pair_completion_tokens = client.max_completion_tokens
                prompt_tokens += pair_prompt_tokens
                completion_tokens += pair_completion_tokens
                miss_cost_by_digest[identity.digest()] = estimate_cost(
                    pair_prompt_tokens,
                    pair_completion_tokens,
                    input_price,
                    output_price,
                )
    return RunBudget(
        api_calls=len(miss_cost_by_digest),
        cache_hits=cache_hits,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated_cost_usd=sum(miss_cost_by_digest.values()),
        miss_cost_by_digest=miss_cost_by_digest,
    )


def print_budget_and_confirm(
    budget: RunBudget,
    *,
    dataset: str,
    methods: tuple[str, ...],
    retriever: str,
    llm_top_k: int,
    requirements: int,
    selected_pairs: int,
) -> bool:
    print("\nPreflight API budget (cache misses only)")
    print(f"  Dataset: {dataset}")
    print(f"  Method: {', '.join(METHOD_LABELS[item] for item in methods)}")
    print(f"  Retriever: {retriever}")
    print(f"  LLM top-k: {llm_top_k}")
    print(f"  Requirements: {requirements}")
    print(f"  Selected pairs: {selected_pairs}")
    print(f"  Cache hits: {budget.cache_hits}")
    print(f"  Estimated API calls: {budget.api_calls}")
    print(f"  Estimated prompt tokens: {budget.prompt_tokens}")
    print(f"  Estimated completion tokens: {budget.completion_tokens}")
    print(f"  Estimated total tokens: {budget.total_tokens}")
    print(f"  Estimated cost: ${budget.estimated_cost_usd:.6f}")
    if budget.api_calls == 0:
        print("All pairs are cached; no API confirmation is needed.")
        return True
    try:
        return input("Continue? [y/N] ").strip() == "y"
    except EOFError:
        return False


def run_method(
    dataset: TraceDataset,
    requirement_ids: list[str],
    method: str,
    client: LLMClient,
    cache: LLMCache,
    estimate_only: bool,
    input_price: float,
    output_price: float,
    progress: RunProgress,
    miss_cost_by_digest: dict[str, float],
    candidates: dict[str, dict[str, tuple[int, float]]],
    retriever: str,
    llm_top_k: int,
    candidate_pool_k: int,
) -> tuple[pd.DataFrame, list[TokenRecord], dict[str, Any]]:
    method_started_at = time.monotonic()
    baseline = create_llm_baseline(
        method, client, cache, dataset, estimate_only, candidates=candidates
    )
    rows: list[dict[str, Any]] = []
    token_records: list[TokenRecord] = []
    prompt_failures = 0
    empty_responses = 0
    cache_verification_failures = 0
    for req_index, requirement_id in enumerate(requirement_ids, start=1):
        prompts = baseline.build_prompts(requirement_id)
        if set(prompts) != set(candidates[requirement_id]):
            raise RuntimeError(
                f"{method}: prompts do not match retriever-selected candidates."
            )
        print(
            f"  {METHOD_LABELS[method]} — Requirement {req_index} / "
            f"{len(requirement_ids)} ({requirement_id})"
        )
        for candidate_index, (code_file, prompt) in enumerate(
            prompts.items(), start=1
        ):
            print(
                f"    Candidate {candidate_index} / {len(prompts)}: "
                f"{code_file}"
            )
            if not prompt.strip():
                prompt_failures += 1
            result = baseline.run_pair(requirement_id, code_file, prompt)
            empty_responses += int(not result.response.text.strip())
            identity = CacheIdentity(
                model=client.model,
                prompt=prompt,
                dataset=dataset.name,
                requirement_id=requirement_id,
                code_file=code_file,
                provider=client.provider,
            )
            miss_cost = miss_cost_by_digest.get(identity.digest(), 0.0)
            progress.record(result.cache_hit, miss_cost)
            print(
                f"    Overall Progress: {progress.completed_pairs} / "
                f"{progress.total_pairs} "
                f"({100 * progress.completed_pairs / progress.total_pairs:.1f}%); "
                f"Estimated Remaining Time: "
                f"{format_duration(progress.estimated_remaining_seconds())}"
            )
            print(
                f"    Cache hits: {progress.cache_hits}; "
                f"API retries: {client.statistics.retries}; "
                f"429 count: {client.statistics.http_429_count}; "
                f"estimated remaining calls: {progress.remaining_calls}; "
                "estimated remaining cost: "
                f"${progress.remaining_cost_usd:.6f}"
            )
            cache_verification_failures += int(
                not result.cache_write_verified
            )
            cost = estimate_cost(
                result.response.prompt_tokens,
                result.response.completion_tokens,
                input_price,
                output_price,
            )
            rows.append(
                {
                    "Dataset": dataset.name,
                    "Method": METHOD_LABELS[method],
                    "Provider": client.provider,
                    "Model": client.model,
                    "Candidate_Selection": "retriever_top_k",
                    "Retriever": retriever,
                    "CandidatePoolK": candidate_pool_k,
                    "LLM_Top_K": llm_top_k,
                    "Retriever_Rank": candidates[requirement_id][code_file][0],
                    "Retriever_Score": candidates[requirement_id][code_file][1],
                    "Req_ID": requirement_id,
                    "Type": dataset.requirements[requirement_id].req_type,
                    "Code_File": code_file,
                    "Prediction": result.prediction.prediction,
                    "Confidence": result.prediction.confidence,
                    "Explanation": result.prediction.explanation,
                    "Raw_Response": result.response.text,
                    "Parse_Error": result.prediction.parse_error,
                    "Is_Positive": (
                        requirement_id, code_file
                    ) in dataset.positive_links,
                    "Is_Estimated": result.response.estimated,
                    "Cache_Hit": result.cache_hit,
                    "Prompt_SHA256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "Prompt_Characters": len(prompt),
                }
            )
            token_records.append(
                TokenRecord(
                    dataset=dataset.name,
                    baseline=METHOD_LABELS[method],
                    provider=client.provider,
                    model=client.model,
                    requirement_id=requirement_id,
                    code_file=code_file,
                    prompt_tokens=result.response.prompt_tokens,
                    completion_tokens=result.response.completion_tokens,
                    total_tokens=result.response.total_tokens,
                    estimated_cost_usd=cost,
                    token_source=(
                        "estimated"
                        if result.response.estimated
                        else "provider_usage"
                    ),
                    cache_hit=result.cache_hit,
                )
            )
    return (
        pd.DataFrame(rows),
        token_records,
        {
            "prompts_generated": len(rows),
            "prompt_failures": prompt_failures,
            "empty_responses": empty_responses,
            "parse_errors": sum(bool(row["Parse_Error"]) for row in rows),
            "cache_hits": sum(row["Cache_Hit"] for row in rows),
            "cache_misses": sum(not row["Cache_Hit"] for row in rows),
            "cache_write_verification_failures": cache_verification_failures,
            "all_candidates_covered": len(rows)
            == sum(len(candidates[req_id]) for req_id in requirement_ids),
            "runtime_seconds": time.monotonic() - method_started_at,
        },
    )


def prediction_score(prediction: str, confidence: Any) -> float:
    """Convert the pair decision to a link-likelihood score for ranking."""
    has_confidence = confidence is not None and not pd.isna(confidence)
    value = float(confidence) if has_confidence else None
    if prediction == "Yes":
        return value if value is not None else 1.0
    if prediction == "No":
        return 1.0 - value if value is not None else 0.0
    return 0.5


def evaluate_predictions(
    dataset: TraceDataset,
    requirement_ids: list[str],
    method: str,
    predictions: pd.DataFrame,
    config: dict[str, Any],
    runtime_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = predictions.copy()
    raw["Score"] = [
        prediction_score(prediction, confidence)
        for prediction, confidence in zip(raw["Prediction"], raw["Confidence"])
    ]
    rankings: dict[str, list[tuple[str, float]]] = {}
    ranked_rows: list[dict[str, Any]] = []
    for requirement_id in requirement_ids:
        requirement_rows = raw.loc[raw["Req_ID"].eq(requirement_id)]
        pairs = sorted(
            zip(requirement_rows["Code_File"], requirement_rows["Score"]),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
        rankings[requirement_id] = [
            (str(code_file), float(score)) for code_file, score in pairs
        ]
        lookup = requirement_rows.set_index("Code_File")
        for rank, (code_file, score) in enumerate(
            rankings[requirement_id], start=1
        ):
            ranked_rows.append(
                {
                    "Req_ID": requirement_id,
                    "Type": dataset.requirements[requirement_id].req_type,
                    "Rank": rank,
                    "Code_File": code_file,
                    "Score": score,
                    "Prediction": lookup.at[code_file, "Prediction"],
                    "Confidence": lookup.at[code_file, "Confidence"],
                    "Is_Positive": bool(lookup.at[code_file, "Is_Positive"]),
                }
            )
    evaluation = config["evaluation"]
    metrics = evaluate_rankings(
        ranked=rankings,
        gold=dataset.ground_truth_by_requirement,
        requirement_types={
            req_id: dataset.requirements[req_id].req_type
            for req_id in requirement_ids
        },
        binary_top_k=int(evaluation["binary_top_k"]),
        cutoffs=tuple(int(value) for value in evaluation["ranking_cutoffs"]),
    )
    metrics.insert(0, "Method", METHOD_LABELS[method])
    metrics.insert(0, "Dataset", dataset.name)
    metrics["RuntimeSeconds"] = runtime_seconds
    metrics["BinaryTopK"] = int(evaluation["binary_top_k"])
    metrics["BinaryProtocol"] = str(evaluation.get("binary_protocol", "top_k"))
    metrics["BinaryThreshold"] = float("nan")
    return pd.DataFrame(ranked_rows), metrics


def update_paper_table(
    rows: pd.DataFrame, path: Path, dataset_order: list[str]
) -> pd.DataFrame:
    if path.is_file():
        existing = pd.read_csv(path)
        selection_columns = [
            column
            for column in (
                "CandidateSelection",
                "Retriever",
                "CandidatePoolK",
                "LLMTopK",
            )
            if column in rows.columns
        ]
        for column in selection_columns:
            if column not in existing.columns:
                existing[column] = ""
        key_columns = ["Dataset", "Method", "Group", *selection_columns]
        keys = set(rows[key_columns].itertuples(index=False, name=None))
        existing = existing[
            ~existing.apply(
                lambda row: tuple(row[column] for column in key_columns) in keys,
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


def update_cost_summary(row: pd.DataFrame, path: Path) -> None:
    if path.is_file():
        existing = pd.read_csv(path)
        selection_columns = [
            column
            for column in (
                "candidate selection",
                "retriever",
                "candidate pool k",
                "llm top k",
            )
            if column in row.columns
        ]
        for column in selection_columns:
            if column not in existing.columns:
                existing[column] = ""
        key_columns = ["dataset", "provider", "model", *selection_columns]
        key = tuple(row.iloc[0][column] for column in key_columns)
        existing = existing[
            ~existing.apply(
                lambda item: tuple(item[column] for column in key_columns) == key,
                axis=1,
            )
        ]
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(path, index=False)


def main() -> int:
    started_at = time.monotonic()
    args = parse_args()
    if args.paper_mode and args.dry_run:
        raise SystemExit(
            "--paper-mode requires real or cached provider responses and cannot "
            "be combined with --dry-run."
        )
    config = load_config(args.config)
    llm_config = config["llm"]
    loader = DatasetLoader(args.config)
    dataset_name = select_dataset(loader, args.dataset)
    dataset = loader.load(dataset_name)
    project_root = config["_project_root"]
    seed = args.seed if args.seed is not None else int(llm_config["random_seed"])
    candidate_pool_k = resolve_candidate_pool_k(args)
    requirement_ids = sample_requirements(
        dataset, args.sample_size, seed, paper_mode=args.paper_mode
    )
    sample_size = len(requirement_ids)
    methods = METHODS if args.method == "all" else (args.method,)
    candidates, retriever_ranking_path = load_retriever_candidates(
        project_root=project_root,
        config=config,
        dataset=dataset,
        requirement_ids=requirement_ids,
        retriever=args.retriever,
        top_k=candidate_pool_k,
    )
    selected_pairs_per_method = sum(
        len(candidates[requirement_id]) for requirement_id in requirement_ids
    )
    selected_pairs = selected_pairs_per_method * len(methods)
    try:
        client = create_client(
            args.provider,
            args.model,
            config,
            max_retries=args.max_retries,
            rpm_limit=args.rpm_limit,
            tpm_limit=args.tpm_limit,
            min_call_interval=args.min_call_interval,
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid Stage 4b API setting: {exc}") from exc
    # Paper mode may run entirely from real cached responses without an API key.
    # It must never silently substitute dry-run estimates for missing paper data.
    estimate_only = args.dry_run or (
        not client.is_configured and not args.paper_mode
    )
    mode = (
        "dry_run"
        if args.dry_run
        else "paper"
        if args.paper_mode
        else "missing_api_key"
        if not client.is_configured
        else "live_sample"
    )

    if not estimate_only and not args.paper_mode and sample_size > int(
        llm_config["max_live_sample_size"]
    ):
        raise SystemExit(
            "Live Stage 4a runs are limited to "
            f"{llm_config['max_live_sample_size']} requirements. "
            "Use --dry-run for larger planning runs."
        )
    if mode == "missing_api_key":
        print(
            f"{client.api_key_env} is not configured. No API call will be made; "
            "prompts and cost estimates will still be generated."
        )
    elif mode == "dry_run":
        print("Dry-run enabled: no API call will be made.")
    elif args.paper_mode:
        print(
            f"Paper mode enabled: {selected_pairs} retriever candidate-pool pairs before "
            f"cache hits ({sample_size} requirements, {len(methods)} method(s))."
        )
        print(
            f"Rate limits: {args.rpm_limit} RPM, {args.tpm_limit} TPM, "
            f"{args.min_call_interval:.1f}s minimum interval, "
            f"{args.max_retries} retries."
        )
    else:
        print(
            f"Live sample enabled: at most {selected_pairs} candidate-pool calls before cache hits "
            f"({sample_size} requirements, {len(methods)} method(s))."
        )
        print(
            f"Rate limits: {args.rpm_limit} RPM, {args.tpm_limit} TPM, "
            f"{args.min_call_interval:.1f}s minimum interval, "
            f"{args.max_retries} retries."
        )

    results_root = ensure_directory(
        project_root / config["output"]["llm_results_dir"]
    )
    cache_root = ensure_directory(
        project_root / config["output"]["llm_cache_dir"]
    )
    statistics_root = ensure_directory(
        project_root / config["output"]["token_statistics_dir"]
    )
    tables_root = ensure_directory(
        project_root / config["output"]["tables_dir"]
    )
    cache = LLMCache(cache_root)
    input_price, output_price = pricing_for(client.model, config)
    budget = estimate_run_budget(
        dataset=dataset,
        requirement_ids=requirement_ids,
        methods=methods,
        client=client,
        cache=cache,
        input_price=input_price,
        output_price=output_price,
        candidates=candidates,
        allow_estimated_cache=estimate_only,
    )
    if args.paper_mode and not client.is_configured and budget.api_calls:
        raise SystemExit(
            f"{client.api_key_env} is not configured and {budget.api_calls} "
            "paper-mode pairs are not cached. Configure the key or restore the "
            "cache; no API call was made."
        )
    if not estimate_only and not print_budget_and_confirm(
        budget,
        dataset=dataset.name,
        methods=methods,
        retriever=args.retriever,
        llm_top_k=candidate_pool_k,
        requirements=sample_size,
        selected_pairs=selected_pairs,
    ):
        print("Aborted: no API calls were made.")
        return 0
    progress = RunProgress(
        total_pairs=selected_pairs,
        remaining_calls=budget.api_calls,
        remaining_cost_usd=budget.estimated_cost_usd,
        started_at=time.monotonic(),
    )
    if args.resume or budget.cache_hits:
        print(
            f"Resume enabled: {budget.cache_hits} completed pair(s) are already "
            "available in cache."
        )

    all_records: list[TokenRecord] = []
    metric_frames: list[pd.DataFrame] = []
    method_summaries: dict[str, Any] = {}
    output_dir = ensure_directory(
        llm_output_dir_for(
            results_root,
            dataset.name,
            client.provider,
            client.model,
            args.retriever,
            args.llm_top_k,
            candidate_pool_k,
        )
    )
    for method in methods:
        predictions, records, diagnostics = run_method(
            dataset=dataset,
            requirement_ids=requirement_ids,
            method=method,
            client=client,
            cache=cache,
            estimate_only=estimate_only,
            input_price=input_price,
            output_price=output_price,
            progress=progress,
            miss_cost_by_digest=budget.miss_cost_by_digest,
            candidates=candidates,
            retriever=args.retriever,
            llm_top_k=candidate_pool_k,
            candidate_pool_k=candidate_pool_k,
        )
        predictions.to_csv(
            output_dir / f"{method}_predictions.csv", index=False
        )
        if args.paper_mode:
            ranked, metrics = evaluate_predictions(
                dataset=dataset,
                requirement_ids=requirement_ids,
                method=method,
                predictions=predictions,
                config=config,
                runtime_seconds=float(diagnostics["runtime_seconds"]),
            )
            ranked = ranked.assign(
                Candidate_Selection="retriever_top_k",
                Retriever=args.retriever,
                CandidatePoolK=candidate_pool_k,
                LLM_Top_K=candidate_pool_k,
            )
            metrics = metrics.assign(
                CandidateSelection="retriever_top_k",
                Retriever=args.retriever,
                CandidatePoolK=candidate_pool_k,
                LLMTopK=candidate_pool_k,
            )
            predictions.to_csv(
                output_dir / f"{method}_raw_predictions.csv", index=False
            )
            ranked.to_csv(
                output_dir / f"{method}_ranked_results.csv", index=False
            )
            metrics.loc[metrics["Group"].eq("Overall")].to_csv(
                output_dir / f"{method}_overall_metrics.csv", index=False
            )
            metrics.loc[~metrics["Group"].eq("Overall")].to_csv(
                output_dir / f"{method}_type_wise_metrics.csv", index=False
            )
            method_token_frame = records_frame(records)
            method_token_frame.to_csv(
                output_dir / f"{method}_token_statistics.csv", index=False
            )
            method_token_summary = summarize_records(records)
            pd.DataFrame(
                [
                    {
                        "Dataset": dataset.name,
                        "Method": METHOD_LABELS[method],
                        "Provider": client.provider,
                        "Model": client.model,
                        "CandidateSelection": "retriever_top_k",
                        "Retriever": args.retriever,
                        "CandidatePoolK": candidate_pool_k,
                        "LLMTopK": candidate_pool_k,
                        "PromptTokens": method_token_summary["prompt_tokens"],
                        "CompletionTokens": method_token_summary[
                            "completion_tokens"
                        ],
                        "TotalTokens": method_token_summary["total_tokens"],
                        "ActualCostUSD": method_token_summary[
                            "estimated_cost_usd"
                        ],
                        "ElapsedSeconds": diagnostics["runtime_seconds"],
                    }
                ]
            ).to_csv(
                output_dir / f"{method}_cost_statistics.csv", index=False
            )
            metric_frames.append(metrics)
            print(metrics.to_string(index=False))
        all_records.extend(records)
        method_summaries[METHOD_LABELS[method]] = {
            **diagnostics,
            **summarize_records(records),
        }

    run_stem = (
        f"{dataset.name}_{client.provider}_{safe_name(client.model)}"
        f"_retriever_top_k_{args.retriever}_top_{candidate_pool_k}"
    )
    run_stem_with_mode = f"{run_stem}_{mode}"
    token_frame = records_frame(all_records)
    token_frame.to_csv(
        statistics_root / f"{run_stem_with_mode}_pair_tokens.csv", index=False
    )
    overall_tokens = summarize_records(all_records)
    new_api_records = [
        record
        for record in all_records
        if not record.cache_hit and record.token_source == "provider_usage"
    ]
    new_api_tokens = summarize_records(new_api_records)
    elapsed_seconds = time.monotonic() - started_at
    runtime_statistics = {
        **client.statistics.to_dict(),
        "cache_hits": progress.cache_hits,
        "prompt_tokens": new_api_tokens["prompt_tokens"],
        "completion_tokens": new_api_tokens["completion_tokens"],
        "total_tokens": new_api_tokens["total_tokens"],
        "actual_cost_usd": new_api_tokens["estimated_cost_usd"],
        "preflight_estimated_cost_usd": budget.estimated_cost_usd,
        "elapsed_seconds": elapsed_seconds,
    }
    summary = {
        "stage": "4c" if args.paper_mode else "4b",
        "mode": mode,
        "api_called": client.statistics.api_calls > 0,
        "dataset": dataset.name,
        "provider": client.provider,
        "model": client.model,
        "api_key_environment_variable": client.api_key_env,
        "sample_size": sample_size,
        "full_dataset": args.paper_mode and args.sample_size is None,
        "paper_mode": args.paper_mode,
        "resume": True,
        "seed": seed,
        "sampled_requirement_ids": requirement_ids,
        "candidate_code_files": len(dataset.code_files),
        "candidate_selection": "retriever_top_k",
        "retriever": args.retriever,
        "retriever_ranking_path": str(retriever_ranking_path),
        "candidate_pool_k": candidate_pool_k,
        "llm_top_k": candidate_pool_k,
        "selected_pairs": selected_pairs,
        "methods": method_summaries,
        "token_statistics": overall_tokens,
        "runtime_statistics": runtime_statistics,
        "preflight_budget": {
            "estimated_api_calls": budget.api_calls,
            "selected_pairs": selected_pairs,
            "cache_hits": budget.cache_hits,
            "estimated_prompt_tokens": budget.prompt_tokens,
            "estimated_completion_tokens": budget.completion_tokens,
            "estimated_total_tokens": budget.total_tokens,
            "estimated_cost_usd": budget.estimated_cost_usd,
        },
        "rate_limit_settings": {
            "requests_per_minute": args.rpm_limit,
            "tokens_per_minute": args.tpm_limit,
            "min_call_interval_seconds": args.min_call_interval,
            "max_retries": args.max_retries,
        },
        "pricing_usd_per_million_tokens": {
            "input": input_price,
            "output": output_price,
        },
        "outputs": {
            "predictions": str(output_dir),
            "cache": str(cache_root),
            "token_statistics": str(statistics_root),
            "paper_tables": str(tables_root) if args.paper_mode else None,
        },
    }
    write_json(
        statistics_root / f"{run_stem_with_mode}_summary.json", summary
    )
    write_json(results_root / "run_summary.json", summary)

    table_row = pd.DataFrame(
        [
            {
                "Dataset": dataset.name,
                "Provider": client.provider,
                "Model": client.model,
                "Mode": mode,
                "CandidateSelection": "retriever_top_k",
                "Retriever": args.retriever,
                "CandidatePoolK": candidate_pool_k,
                "LLMTopK": candidate_pool_k,
                "Requirements": sample_size,
                "Pairs": overall_tokens["pairs"],
                "PromptTokens": overall_tokens["prompt_tokens"],
                "CompletionTokens": overall_tokens["completion_tokens"],
                "TotalTokens": overall_tokens["total_tokens"],
                "EstimatedCostUSD": overall_tokens["estimated_cost_usd"],
                "AverageTokensPerRequirement": overall_tokens[
                    "average_tokens_per_requirement"
                ],
                "AverageTokensPerPair": overall_tokens[
                    "average_tokens_per_pair"
                ],
            }
        ]
    )
    table_path = tables_root / "llm_cost_estimates.csv"
    if table_path.is_file():
        existing = pd.read_csv(table_path)
        key = (
            dataset.name,
            client.provider,
            client.model,
            mode,
            "retriever_top_k",
            args.retriever,
            candidate_pool_k,
            candidate_pool_k,
        )
        existing = existing[
            existing.apply(
                lambda row: (
                    row["Dataset"],
                    row["Provider"],
                    row["Model"],
                    row["Mode"],
                    row.get("CandidateSelection", ""),
                    row.get("Retriever", ""),
                    row.get("CandidatePoolK", ""),
                    row.get("LLMTopK", ""),
                )
                != key,
                axis=1,
            )
        ]
        table_row = pd.concat([existing, table_row], ignore_index=True)
    table_row.to_csv(table_path, index=False)
    if args.paper_mode and metric_frames:
        all_metrics = pd.concat(metric_frames, ignore_index=True)
        update_paper_table(
            all_metrics.loc[all_metrics["Group"].eq("Overall")].copy(),
            tables_root / "llm_baseline_overall_metrics.csv",
            loader.dataset_names,
        )
        update_paper_table(
            all_metrics.copy(),
            tables_root / "llm_baseline_type_wise_metrics.csv",
            loader.dataset_names,
        )
        update_cost_summary(
            pd.DataFrame(
                [
                    {
                        "dataset": dataset.name,
                        "provider": client.provider,
                        "model": client.model,
                        "candidate selection": "retriever_top_k",
                        "retriever": args.retriever,
                        "candidate pool k": candidate_pool_k,
                        "llm top k": candidate_pool_k,
                        "prompt tokens": overall_tokens["prompt_tokens"],
                        "completion tokens": overall_tokens[
                            "completion_tokens"
                        ],
                        "total tokens": overall_tokens["total_tokens"],
                        "actual cost": overall_tokens["estimated_cost_usd"],
                        "elapsed time": elapsed_seconds,
                    }
                ]
            ),
            tables_root / "llm_cost_summary.csv",
        )

    print(f"\nStage {'4c' if args.paper_mode else '4b'} final runtime statistics")
    print(f"  API calls: {runtime_statistics['api_calls']}")
    print(f"  HTTP attempts: {runtime_statistics['http_attempts']}")
    print(f"  Cache hits: {runtime_statistics['cache_hits']}")
    print(f"  Retries: {runtime_statistics['retries']}")
    print(f"  HTTP 429 count: {runtime_statistics['http_429_count']}")
    print(f"  Failures: {runtime_statistics['failures']}")
    print(f"  Prompt tokens: {runtime_statistics['prompt_tokens']}")
    print(f"  Completion tokens: {runtime_statistics['completion_tokens']}")
    print(f"  Total tokens: {runtime_statistics['total_tokens']}")
    print(f"  Actual cost: ${runtime_statistics['actual_cost_usd']:.6f}")
    print(
        "  Preflight estimated cost: "
        f"${runtime_statistics['preflight_estimated_cost_usd']:.6f}"
    )
    print(f"  Elapsed time: {elapsed_seconds:.1f}s")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
