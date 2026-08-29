#!/usr/bin/env python3
"""Run the supplementary full-corpus retrieval experiment for the proposed framework."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DatasetLoader, TraceDataset  # noqa: E402
from src.framework.alignment import (  # noqa: E402
    ALIGNMENT_BATCH_SIZE,
    BidirectionalSemanticAligner,
)
from src.framework.dependency_context import (  # noqa: E402
    build_dependency_graph,
    dependency_coverage_rate,
    dependency_neighbors_to_summarize,
    manifest_dependency_coverage_rate,
)
from src.framework.dependency_context_prompts import (  # noqa: E402
    DependencyContextAlignmentPromptBuilder,
    DependencyContextVerificationPromptBuilder,
)
from src.framework.llm_support import FrameworkLLMService  # noqa: E402
from src.framework.verification import (  # noqa: E402
    VERIFICATION_BATCH_SIZE,
    SelfReflectiveVerifier,
)
from src.llm.cache import LLMCache  # noqa: E402
from src.llm.token_statistics import records_frame, summarize_records  # noqa: E402
from src.utils import ensure_directory, load_config, write_json  # noqa: E402
from scripts.run_llm_baselines import (  # noqa: E402
    create_client,
    load_retriever_candidates,
    pricing_for,
    sample_requirements,
    select_dataset,
)
from scripts.run_pair_classification import (  # noqa: E402
    code_summaries_for_dependency_context,
    estimate_dependency_summary_cost,
    manifest_candidate_code_ids,
)
from scripts.run_proposed_framework import (  # noqa: E402
    apply_scoring_variant,
    atomic_csv,
    completed_pairs,
    ensure_resume_scope,
    estimate_framework_budget,
    evaluate_framework_predictions,
    load_final_framework_config,
    output_dir_for,
    print_preflight_and_confirm,
    run_framework,
)

EXPERIMENT_ROOT = "full_corpus_retrieval_v1"
ALLOWED_DATASETS = ("eTour", "eANCI", "iTrust", "LibEST")
SELECTED_VARIANT_LABEL = (
    "Proposed Framework (selected: evidence_anchored + dependency_context, "
    "frozen 2026-07-20)"
)
DEPENDENCY_CONTEXT_MAX_SHOWN = 3
MAIN_METRIC_COLUMNS = ("Dataset", "Method", "Group", "Requirements", "P@10", "R@10", "MRR")
# Core implementation files this experiment's correctness depends on --
# their SHA-256 is recorded per run so a silent mid-experiment code change
# (e.g. eTour run under one version of alignment.py, eANCI under another)
# is detectable by diffing config_fingerprint.json across datasets.
CORE_SOURCE_FILES = (
    "scripts/run_full_corpus_retrieval.py",
    "scripts/run_proposed_framework.py",
    "scripts/run_pair_classification.py",
    "src/framework/alignment.py",
    "src/framework/verification.py",
    "src/framework/llm_support.py",
    "src/framework/pipeline.py",
    "src/framework/dependency_context.py",
    "src/framework/dependency_context_prompts.py",
    "configs/final_framework_config.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=ALLOWED_DATASETS,
        help="Required unless --aggregate-only.",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Regenerate tables/full_corpus_retrieval_v1/overall_metrics.csv "
        "from whatever per-dataset results/full_corpus_retrieval_v1/.../"
        "metrics.csv files already exist on disk. No API calls, no dataset "
        "required, does not run anything.",
    )
    parser.add_argument("--provider", choices=["openai"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--live-api", action="store_true",
        help="Required to make any real API call. Omit to only preflight.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Smoke-test only: score a random sample instead of the full "
        "dataset. A sampled run is never formal-result-eligible.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-api-calls", type=int, default=None,
        help="Hard cap on real (non-cached) API calls for this run; the "
        "call that would exceed it is refused before it happens and the "
        "run fails with no predictions written.",
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "datasets.yaml"),
    )
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--rpm-limit", type=int, default=10)
    parser.add_argument("--tpm-limit", type=int, default=20_000)
    parser.add_argument("--min-call-interval", type=float, default=3.0)
    return parser.parse_args()


def git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True,
            text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return None


def git_dirty(project_root: Path) -> bool | None:
    """True if the working tree has uncommitted changes, False if clean,
    None if git status could not be determined (e.g. git unavailable) --
    never silently reported as clean in that case."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, capture_output=True,
            text=True, timeout=5, check=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_file_hashes(project_root: Path) -> dict[str, str | None]:
    return {rel: sha256_file(project_root / rel) for rel in CORE_SOURCE_FILES}


def frozen_config_hash(project_root: Path) -> str | None:
    return sha256_file(project_root / "configs" / "final_framework_config.yaml")


def json_safe(value: Any) -> Any:
    """Recursively convert datetime.date/datetime.datetime values to
    ISO-8601 strings so a YAML-parsed dict -- configs/final_framework_config.yaml
    has several bare (unquoted) dates, e.g. `date: 2026-07-17`, which
    PyYAML's safe_load turns into real datetime.date objects -- can be
    passed to write_json()/json.dumps() without crashing.

    Deliberately does NOT use json.dumps(..., default=str): that would
    silently stringify ANY non-serializable type, including a genuine
    future bug (e.g. an accidentally-embedded DataFrame or custom
    object), masking it as if it were fine. Only date/datetime are
    special-cased here; anything else still fails loudly and specifically
    at json.dumps() time, exactly as before this fix."""
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main_metrics_only(metrics: pd.DataFrame) -> pd.DataFrame:
    """Strip a top_k-protocol metrics frame down to exactly Group,
    Requirements, P@10, R@10, MRR -- this experiment's only reportable
    metrics. Drops P@5/R@5 (wrong cutoff) and the generic Precision/
    Recall/F1 columns (a same-protocol-but-differently-named quantity
    that invites confusion with the separate pair-classification F1/P/R
    track) entirely, rather than keep and rename them, per the frozen
    decision to report only P@10/R@10/MRR for this experiment."""
    missing = [c for c in MAIN_METRIC_COLUMNS if c not in metrics.columns]
    if missing:
        raise ValueError(
            f"metrics frame is missing expected column(s) {missing}; cannot "
            "produce the P@10/R@10/MRR-only main table."
        )
    return metrics.loc[:, list(MAIN_METRIC_COLUMNS)].copy()


def build_v3_components(
    dataset: TraceDataset,
    dataset_name: str,
    candidates: dict[str, dict[str, tuple[int, float]]],
    client: Any,
    cache: LLMCache,
    config: dict[str, Any],
    *,
    live_api: bool,
    llm_service: FrameworkLLMService | None = None,
) -> tuple[Any, Any, dict[str, set[str]], dict[str, Any]]:
    """(alignment_prompt_builder, verification_prompt_builder,
    dependency_graph, code_summaries) for the selected v3 variant, scoped
    to exactly the dependency neighbors this run's own retrieved candidates
    will display (never the whole dataset corpus) -- mirrors
    run_pair_classification.py's run_proposed_pair_classifier_live() v3
    path exactly, so the formal pipeline gets the identical mechanism the
    frozen pair-classification selection was validated with.

    `llm_service`, when given (the real-run path), is passed straight
    through to code_summaries_for_dependency_context() so dependency-
    summary calls share the SAME FrameworkLLMService -- and therefore the
    same max_api_calls cap, real_api_calls counter, and token-statistics
    records -- as the alignment/verification calls made later with this
    same instance. Without it (the preflight path, live_api=False), no
    real calls happen anyway."""
    dependency_graph = build_dependency_graph(
        {code_id: artifact.text for code_id, artifact in dataset.code_files.items()}
    )
    candidate_code_ids = manifest_candidate_code_ids(candidates)
    neighbor_ids = dependency_neighbors_to_summarize(
        dependency_graph, candidate_code_ids, DEPENDENCY_CONTEXT_MAX_SHOWN
    )
    code_summaries = code_summaries_for_dependency_context(
        dataset, neighbor_ids, client, cache, config, live_api=live_api,
        llm_service=llm_service,
    )
    alignment_prompt_builder = DependencyContextAlignmentPromptBuilder(
        dependency_graph, code_summaries, DEPENDENCY_CONTEXT_MAX_SHOWN
    )
    verification_prompt_builder = DependencyContextVerificationPromptBuilder(
        dependency_graph, code_summaries, DEPENDENCY_CONTEXT_MAX_SHOWN
    )
    return alignment_prompt_builder, verification_prompt_builder, dependency_graph, code_summaries


def estimate_full_corpus_budget(
    *,
    dataset: TraceDataset,
    dataset_name: str,
    requirement_ids: list[str],
    candidates: dict[str, dict[str, tuple[int, float]]],
    resume_pairs: set[tuple[str, str]],
    client: Any,
    cache: LLMCache,
    config: dict[str, Any],
    input_price: float,
    output_price: float,
    retriever_name: str,
    llm_top_k: int,
    candidate_pool_k: int,
    dcar_selection_strategy: str,
) -> dict[str, Any]:
    """Real preflight for the formal v3 full-corpus run: the base
    understanding/summarization/alignment/verification estimate from
    estimate_framework_budget(), PLUS the scoped dependency-neighbor
    summarization cost (which that function does not know about), folded
    into cache_hits/cache_misses so --live-api's confirmation gate sees the
    true total. Mirrors estimate_proposed_pair_classifier_cost()'s
    folding exactly."""
    candidate_code_ids = manifest_candidate_code_ids(candidates)
    dependency_graph = build_dependency_graph(
        {code_id: artifact.text for code_id, artifact in dataset.code_files.items()}
    )
    neighbor_ids = dependency_neighbors_to_summarize(
        dependency_graph, candidate_code_ids, DEPENDENCY_CONTEXT_MAX_SHOWN
    )
    dependency_summary_estimate = estimate_dependency_summary_cost(
        dataset, dataset_name, neighbor_ids, client, cache, input_price, output_price,
    )
    # Preflight code_summaries lookup never risks a real call (live_api=False).
    code_summaries = code_summaries_for_dependency_context(
        dataset, neighbor_ids, client, cache, config, live_api=False,
    )
    alignment_prompt_builder = DependencyContextAlignmentPromptBuilder(
        dependency_graph, code_summaries, DEPENDENCY_CONTEXT_MAX_SHOWN
    )
    verification_prompt_builder = DependencyContextVerificationPromptBuilder(
        dependency_graph, code_summaries, DEPENDENCY_CONTEXT_MAX_SHOWN
    )
    budget = estimate_framework_budget(
        dataset=dataset,
        requirement_ids=requirement_ids,
        candidates=candidates,
        resume_pairs=resume_pairs,
        client=client,
        cache=cache,
        input_price=input_price,
        output_price=output_price,
        retriever_name=retriever_name,
        llm_top_k=llm_top_k,
        candidate_pool_k=candidate_pool_k,
        dcar_selection_strategy=dcar_selection_strategy,
        alignment_prompt_builder=alignment_prompt_builder,
        verification_prompt_builder=verification_prompt_builder,
    )
    total_cost = budget["estimated_cost_usd"] + dependency_summary_estimate[
        "dependency_summary_estimated_cost_usd"
    ]
    total_prompt_tokens = budget["estimated_prompt_tokens"] + dependency_summary_estimate[
        "dependency_summary_estimated_prompt_tokens"
    ]
    total_cache_hits = budget["cache_hits"] + dependency_summary_estimate[
        "dependency_summary_cache_hits"
    ]
    total_cache_misses = budget["expected_api_calls"] + dependency_summary_estimate[
        "dependency_summary_cache_misses"
    ]
    return {
        "dataset": dataset_name,
        "variant": SELECTED_VARIANT_LABEL,
        "selected_requirements": budget["selected_requirements"],
        "pending_requirements": budget["pending_requirements"],
        "candidate_pool_pairs": budget["candidate_pool_pairs"],
        "pending_pairs": budget["pending_pairs"],
        "call_breakdown_by_category": budget["call_counts"],
        "cache_hits_by_category": {
            name: stats["cache_hits"] for name, stats in budget["category_stats"].items()
        },
        "cache_misses_by_category": {
            name: stats["cache_misses"] for name, stats in budget["category_stats"].items()
        },
        "manifest_dependency_context_coverage_rate": round(
            manifest_dependency_coverage_rate(dependency_graph, candidate_code_ids), 4
        ),
        "full_dataset_dependency_graph_coverage_rate": round(
            dependency_coverage_rate(dependency_graph), 4
        ),
        **dependency_summary_estimate,
        "cache_hits": total_cache_hits,
        "cache_misses": total_cache_misses,
        "estimated_new_prompt_tokens": total_prompt_tokens,
        "estimated_cost_usd": round(total_cost, 6),
    }


def config_fingerprint(
    project_root: Path,
    frozen_config: dict[str, Any],
    dataset_name: str,
    model: str,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ROOT,
        "variant": SELECTED_VARIANT_LABEL,
        "dataset": dataset_name,
        "allowed_datasets": list(ALLOWED_DATASETS),
        "model": model,
        "git_commit": git_commit(project_root),
        "git_dirty": git_dirty(project_root),
        "frozen_config_path": str(project_root / "configs" / "final_framework_config.yaml"),
        "frozen_config": json_safe(frozen_config),
        "frozen_config_sha256": frozen_config_hash(project_root),
        "core_source_file_sha256": source_file_hashes(project_root),
        "alignment_batch_size": ALIGNMENT_BATCH_SIZE,
        "verification_batch_size": VERIFICATION_BATCH_SIZE,
        "batching_mechanism": (
            "deterministic chunked batching, strict per-field schema "
            "validation (align_requirement_batch/verify_batch strict=True "
            "parsing), incomplete/invalid responses retried and never "
            "cached, unrecoverable chunk hard-fails the whole run -- no "
            "local-fallback score can enter these predictions"
        ),
        "main_binary_protocol": "top_k (P@10/R@10/MRR only; the stale "
        "'threshold' supplementary protocol, P@5/R@5, and pair-"
        "classification F1/P/R are never computed or written by this "
        "script)",
        "dependency_context_max_shown": DEPENDENCY_CONTEXT_MAX_SHOWN,
    }


def aggregate_main_experiment_tables(project_root: Path) -> Path:
    """Rebuild tables/full_corpus_retrieval_v1/overall_metrics.csv from
    every metrics.csv found anywhere under results/full_corpus_retrieval_v1/
    (Proposed's nested provider/model/retriever/pool/dcar_top_k path AND
    every baseline's flat {dataset}/{method}/ path -- a plain recursive
    scan needs no knowledge of either shape). Each metrics.csv already
    carries its own Dataset/Method columns (written by
    evaluate_framework_predictions() / evaluate_rankings()-based baseline
    evaluators alike), so no path-parsing is needed to recover them.
    Whatever hasn't been run yet is simply absent -- this never blocks on
    every dataset/method being done."""
    experiment_root_dir = project_root / "results" / EXPERIMENT_ROOT
    rows = []
    if experiment_root_dir.is_dir():
        for metrics_path in sorted(experiment_root_dir.rglob("metrics.csv")):
            metrics = pd.read_csv(metrics_path)
            overall = metrics.loc[metrics["Group"] == "Overall"]
            if overall.empty:
                continue
            rows.append(overall.iloc[0].to_dict())

    table = pd.DataFrame(rows, columns=list(MAIN_METRIC_COLUMNS))
    tables_dir = ensure_directory(project_root / "tables" / EXPERIMENT_ROOT)
    table_path = tables_dir / "overall_metrics.csv"
    table.to_csv(table_path, index=False)
    return table_path


def append_audit_ledger(
    project_root: Path,
    dataset_name: str,
    run_summary: dict[str, Any],
    output_dir_path: Path,
) -> Path:
    """Append one row to this experiment's cross-dataset audit ledger.
    Never overwrites prior rows -- re-running a dataset adds a new dated
    row rather than losing the record of a previous attempt. Shared by
    both the Proposed experiment (run_summary["variant"]) and the
    baselines experiment (run_summary["method"]) -- either key labels
    what was run."""
    audit_dir = ensure_directory(project_root / "audit" / EXPERIMENT_ROOT)
    ledger_path = audit_dir / "run_ledger.csv"
    row = {
        "Timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "Dataset": dataset_name,
        "Variant": run_summary.get("variant", run_summary.get("method")),
        "Requirements": run_summary["requirements"],
        "PredictionsWritten": run_summary["predictions_written"],
        "RealAPICalls": run_summary["real_api_calls"],
        "MaxAPICalls": run_summary["max_api_calls"],
        "FallbackPredictions": run_summary["fallback_predictions"],
        "ParseErrors": run_summary["parse_errors"],
        "EmptyResponses": run_summary["empty_responses"],
        "EstimatedCostUSD": run_summary["estimated_cost_usd"],
        "RuntimeSeconds": run_summary["runtime_seconds"],
        "OutputDir": str(output_dir_path),
    }
    frame = pd.DataFrame([row])
    if ledger_path.is_file():
        existing = pd.read_csv(ledger_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(ledger_path, index=False)
    return ledger_path


def main() -> int:
    started = time.monotonic()
    args = parse_args()
    if args.aggregate_only:
        config = load_config(args.config)
        table_path = aggregate_main_experiment_tables(config["_project_root"])
        print(f"Wrote {table_path}")
        return 0
    if args.dataset is None:
        raise SystemExit("--dataset is required (unless --aggregate-only).")
    if args.dataset not in ALLOWED_DATASETS:
        raise SystemExit(
            f"{args.dataset!r} is not one of the 4 public datasets this "
            f"experiment covers: {ALLOWED_DATASETS}."
        )
    config = load_config(args.config)
    project_root = config["_project_root"]
    # This experiment always uses the one frozen configuration -- there is
    # no --retriever/--candidate-pool-k/etc. override, unlike
    # run_proposed_framework.py's more general CLI.
    frozen_config = load_final_framework_config(project_root)

    retriever = frozen_config["retrieval"]["retriever"]
    candidate_pool_k = int(frozen_config["retrieval"]["candidate_pool_k"])
    llm_top_k = int(frozen_config["retrieval"]["dcar_selected_k"])
    dcar_selection_strategy = frozen_config["retrieval"]["dcar_selection_strategy"]
    scoring_variant = str(frozen_config["scoring"]["selected_variant"])
    threshold = float(frozen_config["scoring"]["threshold"])

    loader = DatasetLoader(args.config)
    dataset_name = select_dataset(loader, args.dataset)
    dataset = loader.load(dataset_name)
    seed = int(config["llm"]["random_seed"])
    requirement_ids = sample_requirements(
        dataset, args.sample_size, seed, paper_mode=(args.sample_size is None),
    )

    model = args.model or str(config["llm"]["providers"][args.provider]["default_model"])
    api_key_env = str(config["llm"]["providers"][args.provider]["api_key_env"])
    has_api_key = bool(os.environ.get(api_key_env, "").strip())

    client = create_client(
        args.provider, model, config, max_retries=args.max_retries,
        rpm_limit=args.rpm_limit, tpm_limit=args.tpm_limit,
        min_call_interval=args.min_call_interval,
    )
    cache = LLMCache(ensure_directory(project_root / config["output"]["llm_cache_dir"]))
    input_price, output_price = pricing_for(client.model, config)

    candidates, retriever_ranking_path = load_retriever_candidates(
        project_root=project_root, config=config, dataset=dataset,
        requirement_ids=requirement_ids, retriever=retriever, top_k=candidate_pool_k,
    )

    output_dir_path = output_dir_for(
        project_root, dataset.name, args.provider, model, retriever, llm_top_k,
        candidate_pool_k, dcar_selection_strategy, args.sample_size,
        results_subdir=EXPERIMENT_ROOT,
    )
    predictions_path = output_dir_path / "predictions.csv"
    ensure_resume_scope(output_dir_path, args.sample_size, args.resume)
    mode = "live_api" if args.live_api else "no_api_local"
    existing_predictions, resume_pairs = completed_pairs(predictions_path, args.resume, mode)

    budget = estimate_full_corpus_budget(
        dataset=dataset, dataset_name=dataset_name, requirement_ids=requirement_ids,
        candidates=candidates, resume_pairs=resume_pairs, client=client, cache=cache,
        config=config, input_price=input_price, output_price=output_price,
        retriever_name=retriever, llm_top_k=llm_top_k, candidate_pool_k=candidate_pool_k,
        dcar_selection_strategy=dcar_selection_strategy,
    )
    fingerprint = config_fingerprint(project_root, frozen_config, dataset_name, model)

    print(f"\n=== {EXPERIMENT_ROOT} preflight: {dataset_name} ===")
    print(f"Variant: {SELECTED_VARIANT_LABEL}")
    print(f"Model/provider: {model} / {args.provider}  (API key configured: {has_api_key})")
    print(f"Output dir: {output_dir_path}")
    print(f"Retriever: {retriever}  candidate_pool_k={candidate_pool_k}  "
          f"dcar_selected_k={llm_top_k}  dcar_selection_strategy={dcar_selection_strategy}")
    print(f"Requirements: {budget['selected_requirements']} "
          f"(pending after resume: {budget['pending_requirements']})")
    print(f"Candidate-pool pairs: {budget['candidate_pool_pairs']} "
          f"(pending: {budget['pending_pairs']})")
    print("Call breakdown by category:")
    for name, count in budget["call_breakdown_by_category"].items():
        hits = budget["cache_hits_by_category"][name]
        misses = budget["cache_misses_by_category"][name]
        print(f"  {name}: calls={count} cache_hits={hits} cache_misses={misses}")
    print(f"Dependency summary: code_ids_in_scope="
          f"{budget['dependency_summary_code_ids_in_scope']} "
          f"cache_hits={budget['dependency_summary_cache_hits']} "
          f"cache_misses={budget['dependency_summary_cache_misses']}")
    print(f"manifest_dependency_context_coverage_rate: "
          f"{budget['manifest_dependency_context_coverage_rate']}")
    print(f"full_dataset_dependency_graph_coverage_rate: "
          f"{budget['full_dataset_dependency_graph_coverage_rate']}")
    print(f"Total cache_hits: {budget['cache_hits']}  cache_misses (real API calls "
          f"if run now): {budget['cache_misses']}")
    print(f"Estimated cost: ${budget['estimated_cost_usd']:.4f}")
    print(f"Config fingerprint: git_commit={fingerprint['git_commit']}  "
          f"main_binary_protocol={fingerprint['main_binary_protocol']}")

    if not args.live_api:
        print("\n--live-api not passed: no API calls made, no predictions written.")
        return 0

    if not has_api_key:
        print(f"\n{api_key_env} is not configured. Aborting: cannot run --live-api.")
        return 1

    if budget["cache_misses"] > 0:
        if not print_preflight_and_confirm(dataset=dataset_name, budget={
            "selected_requirements": budget["selected_requirements"],
            "pending_requirements": budget["pending_requirements"],
            "candidate_pool_pairs": budget["candidate_pool_pairs"],
            "pending_pairs": budget["pending_pairs"],
            "candidate_pool_k": candidate_pool_k,
            "dcar_selected_k": llm_top_k,
            "dcar_selection_strategy": dcar_selection_strategy,
            "category_stats": {
                name: {
                    "estimated_calls": budget["call_breakdown_by_category"][name],
                    "cache_hits": budget["cache_hits_by_category"][name],
                    "cache_misses": budget["cache_misses_by_category"][name],
                    "expected_api_calls": budget["cache_misses_by_category"][name],
                }
                for name in budget["call_breakdown_by_category"]
            },
            "summary_unique_code_files": budget["call_breakdown_by_category"].get(
                "code_summarization", 0
            ),
            "summary_cache_hits": budget["cache_hits_by_category"].get(
                "code_summarization", 0
            ),
            "summary_cache_misses": budget["cache_misses_by_category"].get(
                "code_summarization", 0
            ),
            "summary_api_calls": budget["cache_misses_by_category"].get(
                "code_summarization", 0
            ),
            "summary_reused": budget["cache_hits_by_category"].get(
                "code_summarization", 0
            ),
            "estimated_framework_llm_calls": sum(budget["call_breakdown_by_category"].values()),
            "cache_hits": budget["cache_hits"],
            "expected_api_calls": budget["cache_misses"],
            "estimated_prompt_tokens": budget["estimated_new_prompt_tokens"],
            "estimated_completion_tokens": 0,
            "estimated_total_tokens": budget["estimated_new_prompt_tokens"],
            "estimated_cost_usd": budget["estimated_cost_usd"],
        }):
            print("Aborted: no API calls were made and no formal results were written.")
            return 0

    # One shared FrameworkLLMService for EVERY real call this run makes --
    # dependency-summary fetches (build_v3_components), understanding,
    # code summarization, alignment, and verification -- so max_api_calls,
    # real_api_calls, and token-statistics records are never split across
    # disconnected instances (a separate internal service for dependency
    # summaries would silently bypass this run's budget cap and cost
    # accounting for that whole call category).
    llm_service = FrameworkLLMService(
        client, cache, dataset.name, live_api=True, input_price=input_price,
        output_price=output_price, max_api_calls=args.max_api_calls,
    )
    alignment_prompt_builder, verification_prompt_builder, _graph, _summaries = build_v3_components(
        dataset, dataset_name, candidates, client, cache, config, live_api=True,
        llm_service=llm_service,
    )
    aligner = BidirectionalSemanticAligner(
        llm_service=llm_service, prompt_builder=alignment_prompt_builder
    )
    verifier = SelfReflectiveVerifier(
        llm_service=llm_service, prompt_builder=verification_prompt_builder
    )

    output_dir = ensure_directory(output_dir_path)
    predictions, token_records, diagnostics = run_framework(
        dataset=dataset, requirement_ids=requirement_ids, candidates=candidates,
        provider=args.provider, model=model, retriever=retriever, llm_top_k=llm_top_k,
        candidate_pool_k=candidate_pool_k, threshold=threshold, resume_pairs=resume_pairs,
        llm_service=llm_service, dcar_selection_strategy=dcar_selection_strategy,
        aligner=aligner, verifier=verifier,
    )
    if not existing_predictions.empty:
        predictions = pd.concat([existing_predictions, predictions], ignore_index=True)
        predictions = predictions.drop_duplicates(subset=["Req_ID", "Code_File"], keep="last")
    predictions = apply_scoring_variant(predictions, scoring_variant, threshold)
    atomic_csv(predictions, predictions_path)

    ranked, raw_metrics, _stale_threshold_metrics = evaluate_framework_predictions(
        dataset=dataset, requirement_ids=requirement_ids, predictions=predictions,
        config=config, runtime_seconds=time.monotonic() - started, retriever=retriever,
        llm_top_k=llm_top_k, candidate_pool_k=candidate_pool_k, threshold=threshold,
    )
    # _stale_threshold_metrics (the "threshold" supplementary protocol,
    # marked stale in configs/final_framework_config.yaml) is intentionally
    # discarded -- never written anywhere by this script.
    metrics = main_metrics_only(raw_metrics)
    atomic_csv(ranked, output_dir / "ranked_results.csv")
    atomic_csv(metrics, output_dir / "metrics.csv")

    # token_records IS llm_service.records -- it already includes the
    # dependency-summary calls made (if any) inside build_v3_components()
    # above, since that call shared this same llm_service instance.
    all_token_records = token_records
    token_frame = records_frame(all_token_records)
    atomic_csv(token_frame, output_dir / "token_statistics.csv")
    token_summary = summarize_records(all_token_records)
    write_json(output_dir / "token_statistics.json", token_summary)

    write_json(output_dir / "config_fingerprint.json", fingerprint)
    run_summary = {
        "dataset": dataset_name,
        "variant": SELECTED_VARIANT_LABEL,
        "requirements": len(requirement_ids),
        "predictions_written": len(predictions),
        "real_api_calls": llm_service.real_api_calls,
        "max_api_calls": args.max_api_calls,
        "fallback_predictions": 0,
        "fallback_note": (
            "Structurally guaranteed 0: align_requirement_batch()/"
            "verify_batch() hard-fail (IncompleteBatchResponseError) "
            "the entire run before any predictions.csv is written if "
            "any chunk cannot be strictly validated -- this file only "
            "exists because every chunk succeeded."
        ),
        "parse_errors": diagnostics["parse_errors"],
        "empty_responses": diagnostics["empty_responses"],
        "estimated_cost_usd": token_summary["estimated_cost_usd"],
        "runtime_seconds": time.monotonic() - started,
        "retriever_ranking_path": str(retriever_ranking_path),
        "main_binary_protocol": "top_k",
    }
    write_json(output_dir / "run_summary.json", run_summary)

    append_audit_ledger(project_root, dataset_name, run_summary, output_dir_path)
    aggregate_main_experiment_tables(project_root)

    print(f"\n{dataset_name}/{EXPERIMENT_ROOT} live run complete -> {output_dir}")
    print(f"real_api_calls={llm_service.real_api_calls}  "
          f"estimated_cost_usd=${token_summary['estimated_cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
