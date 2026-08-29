#!/usr/bin/env python3
"""Formal, paper-facing full-corpus retrieval experiment for the non-
Proposed baselines (IR: TFIDF/BM25/LSI, Neural: Sentence-BERT/CodeBERT,
LLM: Direct Prompting/Generic RAG-LLM), matching
scripts/run_full_corpus_retrieval.py's isolation and metric conventions
exactly so the two are directly comparable:

  - Same output roots: results/full_corpus_retrieval_v1/,
    tables/full_corpus_retrieval_v1/, audit/full_corpus_retrieval_v1/.
  - Same main metric restriction: only P@10/R@10/MRR are ever written
    (main_metrics_only(), reused verbatim from run_full_corpus_retrieval.py).
  - Same config-fingerprint discipline (git commit/dirty, frozen-config
    hash, core source file hashes).
  - Never reads/writes results/proposed_framework/, results/ir_results_dir,
    results/neural_results_dir, or results/llm_results_dir -- those are
    separate experiment tracks with their own (differently-scoped) outputs.

IR and Neural methods score the FULL corpus (every requirement x every
code file -- no retriever narrowing), so their "candidates" count is the
full pairwise matrix size, not a bounded pool.

LLM methods (Direct Prompting, Generic RAG-LLM) judge ONLY the
retriever's own top LLM_SCORED_K=10 candidates per requirement --
llm_scored_k is deliberately its own, independently-derived value (read
from the frozen config's retrieval.dcar_selected_k, the same "10" DCAR
narrows Proposed's candidates to) and is NEVER the Proposed experiment's
retrieval.candidate_pool_k=50. Retrieving only the top 10 directly (not
retrieving 50 and then filtering) avoids the latent narrowing gap in
run_llm_baselines.py's own run_method(), which accepts a separate
llm_top_k parameter but does not actually use it to narrow `candidates`
before scoring -- candidates is scored in full regardless of llm_top_k
there. That gap is out of scope here (pre-existing, unrelated shared
code); this script simply never constructs a >10-candidate pool for LLM
scoring in the first place.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DatasetLoader, TraceDataset  # noqa: E402
from src.llm.cache import LLMCache  # noqa: E402
from src.llm.token_statistics import records_frame, summarize_records  # noqa: E402
from src.utils import ensure_directory, load_config, write_json  # noqa: E402
from scripts.run_ir_baselines import run_method as run_ir_method  # noqa: E402
from scripts.run_neural_baselines import run_method as run_neural_method  # noqa: E402
from scripts.run_llm_baselines import (  # noqa: E402
    RunProgress,
    create_client,
    estimate_run_budget,
    evaluate_predictions,
    load_retriever_candidates,
    pricing_for,
    print_budget_and_confirm,
    run_method as run_llm_method,
    safe_name,
    select_dataset,
)
from scripts.run_proposed_framework import load_final_framework_config  # noqa: E402
from scripts.run_full_corpus_retrieval import (  # noqa: E402
    ALLOWED_DATASETS,
    EXPERIMENT_ROOT,
    MAIN_METRIC_COLUMNS,
    aggregate_main_experiment_tables,
    append_audit_ledger,
    frozen_config_hash,
    git_commit,
    git_dirty,
    json_safe,
    main_metrics_only,
    sha256_file,
)

IR_METHODS = ("tfidf", "bm25", "lsi")
NEURAL_METHODS = ("sentencebert", "codebert")
LLM_METHODS = ("direct", "rag")
METHOD_LABELS = {
    "tfidf": "TFIDF", "bm25": "BM25", "lsi": "LSI",
    "sentencebert": "Sentence-BERT", "codebert": "CodeBERT",
    "direct": "Direct Prompting", "rag": "Generic RAG-LLM",
}
ALL_METHODS = IR_METHODS + NEURAL_METHODS + LLM_METHODS


def llm_scored_k_for(frozen_config: dict[str, Any]) -> int:
    """The number of retriever-ranked candidates Direct Prompting / Generic
    RAG-LLM judge per requirement. Deliberately independent from
    retrieval.candidate_pool_k=50 (Proposed's DCAR *input* pool size) --
    reuses retrieval.dcar_selected_k as its single source of truth for
    "10", since that is the same number DCAR narrows Proposed's own
    candidates to, but under its own name so it is never accidentally
    passed candidate_pool_k instead."""
    return int(frozen_config["retrieval"]["dcar_selected_k"])


def output_dir_for_baseline(project_root: Path, dataset_name: str, method: str) -> Path:
    """Isolated output directory for one baseline method, matching
    output_dir_for(..., results_subdir=EXPERIMENT_ROOT)'s root but keyed
    by method label instead of the Proposed-specific
    provider/model/retriever/candidate_pool_k path shape -- these baselines
    don't share that structure (IR/Neural have no provider/model at all;
    LLM baselines' provider/model/retriever/llm_scored_k are recorded in
    the config fingerprint instead of the path)."""
    return (
        project_root / "results" / EXPERIMENT_ROOT / dataset_name
        / safe_name(METHOD_LABELS[method])
    )


def _core_source_files_for(method: str) -> tuple[str, ...]:
    if method in IR_METHODS:
        return ("scripts/run_ir_baselines.py", "src/baselines/ir.py", "src/evaluation/metrics.py")
    if method in NEURAL_METHODS:
        return ("scripts/run_neural_baselines.py", "src/baselines/neural.py", "src/evaluation/metrics.py")
    return (
        "scripts/run_full_corpus_retrieval_baselines.py",
        "scripts/run_llm_baselines.py", "src/baselines/llm.py", "src/evaluation/metrics.py",
    )


def baseline_config_fingerprint(
    project_root: Path,
    method: str,
    dataset_name: str,
    frozen_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fingerprint: dict[str, Any] = {
        "experiment": EXPERIMENT_ROOT,
        "method": METHOD_LABELS[method],
        "dataset": dataset_name,
        "allowed_datasets": list(ALLOWED_DATASETS),
        "git_commit": git_commit(project_root),
        "git_dirty": git_dirty(project_root),
        "core_source_file_sha256": {
            rel: sha256_file(project_root / rel) for rel in _core_source_files_for(method)
        },
        "main_binary_protocol": "top_k (P@10/R@10/MRR only; the stale "
        "'threshold' supplementary protocol, P@5/R@5, and pair-"
        "classification F1/P/R are never computed or written by this "
        "script)",
    }
    if method in LLM_METHODS:
        fingerprint["frozen_config_path"] = str(
            project_root / "configs" / "final_framework_config.yaml"
        )
        fingerprint["frozen_config_sha256"] = frozen_config_hash(project_root)
        fingerprint["frozen_config"] = json_safe(frozen_config)
        fingerprint["llm_scored_k"] = llm_scored_k_for(frozen_config) if frozen_config else None
        fingerprint["llm_scored_k_note"] = (
            "Number of retriever-ranked candidates judged per requirement -- "
            "independently derived from retrieval.dcar_selected_k, never "
            "retrieval.candidate_pool_k (=50, Proposed's DCAR input pool "
            "size, not used by these LLM baselines at all)."
        )
    return fingerprint


def dry_run_ir_or_neural(
    dataset: TraceDataset, method: str, config: dict[str, Any],
) -> dict[str, Any]:
    """IR/Neural methods are zero-cost and local -- there is nothing to
    estimate, so this actually computes the real metrics as a preview
    (never written to disk here). candidates = the full requirement x
    code_file matrix, since these methods score every pair, not a
    retriever-narrowed pool."""
    n_requirements = len(dataset.evaluation_requirements)
    n_code_files = len(dataset.code_files)
    if method in IR_METHODS:
        _raw, _ranked, metrics, diagnostics = run_ir_method(dataset, method, config)
    else:
        cache_dir = ensure_directory(PROJECT_ROOT / config["neural"]["cache_dir"])
        _raw, _ranked, metrics, diagnostics = run_neural_method(
            dataset, method, config, cache_dir, "auto"
        )
    overall = main_metrics_only(metrics).loc[lambda df: df["Group"].eq("Overall")]
    return {
        "method": METHOD_LABELS[method],
        "candidates": n_requirements * n_code_files,
        "requirements": n_requirements,
        "code_files": n_code_files,
        "cache_hits": None,  # no cache concept for IR; see diagnostics for Neural
        "estimated_cost_usd": 0.0,
        "preview_overall_metrics": overall.to_dict(orient="records")[0] if not overall.empty else {},
        "diagnostics": diagnostics,
    }


def _llm_preflight(
    dataset: TraceDataset,
    method: str,
    config: dict[str, Any],
    frozen_config: dict[str, Any],
) -> tuple[Any, Any, dict[str, dict[str, tuple[int, float]]], Any, str, int]:
    """Shared setup for both the dry-run and the real (--live-api) LLM
    baseline path: client, cache, the llm_scored_k-limited candidates
    (never candidate_pool_k), and a real (allow_estimated_cache=False)
    preflight budget."""
    retriever = frozen_config["retrieval"]["retriever"]
    llm_scored_k = llm_scored_k_for(frozen_config)
    requirement_ids = sorted(dataset.evaluation_requirements)
    model = str(config["llm"]["providers"]["openai"]["default_model"])
    client = create_client(
        "openai", model, config, max_retries=8, rpm_limit=10,
        tpm_limit=20_000, min_call_interval=3.0,
    )
    cache = LLMCache(ensure_directory(PROJECT_ROOT / config["output"]["llm_cache_dir"]))
    input_price, output_price = pricing_for(client.model, config)
    candidates, _ranking_path = load_retriever_candidates(
        project_root=PROJECT_ROOT, config=config, dataset=dataset,
        requirement_ids=requirement_ids, retriever=retriever, top_k=llm_scored_k,
    )
    budget = estimate_run_budget(
        dataset=dataset, requirement_ids=requirement_ids, methods=(method,),
        client=client, cache=cache, input_price=input_price, output_price=output_price,
        candidates=candidates,
        # Only a genuine prior response counts as a cache hit here, never a
        # dry-run-only "estimated" placeholder -- an estimated entry is not
        # real data and a live run would still need to call the API for
        # it, so counting it as a hit would understate the true cost of
        # running --live-api now.
        allow_estimated_cache=False,
    )
    return client, cache, candidates, budget, retriever, llm_scored_k


def dry_run_llm(
    dataset: TraceDataset,
    dataset_name: str,
    method: str,
    config: dict[str, Any],
    frozen_config: dict[str, Any],
) -> dict[str, Any]:
    """Real (no-API) preflight for Direct Prompting / Generic RAG-LLM,
    scoped to the retriever's own top llm_scored_k=10 candidates per
    requirement (NOT candidate_pool_k=50)."""
    _client, _cache, candidates, budget, retriever, llm_scored_k = _llm_preflight(
        dataset, method, config, frozen_config
    )
    candidate_pool_pairs = sum(len(v) for v in candidates.values())
    return {
        "method": METHOD_LABELS[method],
        "candidates": candidate_pool_pairs,
        "requirements": len(candidates),
        "retriever": retriever,
        "llm_scored_k": llm_scored_k,
        "cache_hits": budget.cache_hits,
        "cache_misses": budget.api_calls,
        "estimated_cost_usd": round(budget.estimated_cost_usd, 6),
    }


def write_ir_or_neural_result(
    project_root: Path,
    dataset: TraceDataset,
    dataset_name: str,
    method: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute (zero-cost, local, no API) and persist one IR/Neural
    baseline's isolated full-corpus result. No confirmation gate -- there
    is no cost or API call to authorize."""
    started = time.monotonic()
    n_requirements = len(dataset.evaluation_requirements)
    n_code_files = len(dataset.code_files)
    if method in IR_METHODS:
        raw, ranked, raw_metrics, diagnostics = run_ir_method(dataset, method, config)
    else:
        cache_dir = ensure_directory(project_root / config["neural"]["cache_dir"])
        raw, ranked, raw_metrics, diagnostics = run_neural_method(
            dataset, method, config, cache_dir, "auto"
        )
    metrics = main_metrics_only(raw_metrics)

    out_dir = ensure_directory(output_dir_for_baseline(project_root, dataset_name, method))
    raw.to_csv(out_dir / "raw_scores.csv", index=False)
    ranked.to_csv(out_dir / "ranked_results.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    fingerprint = baseline_config_fingerprint(project_root, method, dataset_name)
    write_json(out_dir / "config_fingerprint.json", fingerprint)

    run_summary = {
        "dataset": dataset_name,
        "method": METHOD_LABELS[method],
        "requirements": n_requirements,
        "predictions_written": len(raw),
        "real_api_calls": 0,
        "max_api_calls": None,
        "fallback_predictions": 0,
        "parse_errors": 0,
        "empty_responses": 0,
        "estimated_cost_usd": 0.0,
        "runtime_seconds": time.monotonic() - started,
        "main_binary_protocol": "top_k",
        "diagnostics": diagnostics,
    }
    write_json(out_dir / "run_summary.json", run_summary)
    append_audit_ledger(project_root, dataset_name, run_summary, out_dir)
    return run_summary


def write_llm_result(
    project_root: Path,
    dataset: TraceDataset,
    dataset_name: str,
    method: str,
    config: dict[str, Any],
    frozen_config: dict[str, Any],
    *,
    max_api_calls: int | None,
) -> dict[str, Any] | None:
    """Real (--live-api) run for one LLM baseline, scoped to the
    retriever's own top llm_scored_k candidates per requirement. Returns
    None (writes nothing) if max_api_calls is exceeded by the preflight,
    or the user declines the confirmation prompt."""
    client, cache, candidates, budget, retriever, llm_scored_k = _llm_preflight(
        dataset, method, config, frozen_config
    )
    requirement_ids = sorted(candidates)
    if max_api_calls is not None and budget.api_calls > max_api_calls:
        print(
            f"Refusing to run {METHOD_LABELS[method]}: preflight expects "
            f"{budget.api_calls} real API calls, exceeding --max-api-calls "
            f"{max_api_calls}. No calls made, nothing written."
        )
        return None
    selected_pairs = sum(len(candidates[r]) for r in requirement_ids)
    if not print_budget_and_confirm(
        budget, dataset=dataset_name, methods=(method,), retriever=retriever,
        llm_top_k=llm_scored_k, requirements=len(requirement_ids), selected_pairs=selected_pairs,
    ):
        print(f"Aborted {METHOD_LABELS[method]}: no API calls were made, nothing written.")
        return None

    input_price, output_price = pricing_for(client.model, config)
    progress = RunProgress(
        total_pairs=selected_pairs, remaining_calls=budget.api_calls,
        remaining_cost_usd=budget.estimated_cost_usd, started_at=time.monotonic(),
    )
    started = time.monotonic()
    predictions, token_records, diagnostics = run_llm_method(
        dataset=dataset, requirement_ids=requirement_ids, method=method, client=client,
        cache=cache, estimate_only=False, input_price=input_price, output_price=output_price,
        progress=progress, miss_cost_by_digest=budget.miss_cost_by_digest, candidates=candidates,
        retriever=retriever, llm_top_k=llm_scored_k, candidate_pool_k=llm_scored_k,
    )
    runtime_seconds = time.monotonic() - started
    ranked, raw_metrics = evaluate_predictions(
        dataset=dataset, requirement_ids=requirement_ids, method=method,
        predictions=predictions, config=config, runtime_seconds=runtime_seconds,
    )
    metrics = main_metrics_only(raw_metrics)

    out_dir = ensure_directory(output_dir_for_baseline(project_root, dataset_name, method))
    predictions.to_csv(out_dir / "raw_scores.csv", index=False)
    ranked.to_csv(out_dir / "ranked_results.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    token_frame = records_frame(token_records)
    token_frame.to_csv(out_dir / "token_statistics.csv", index=False)
    token_summary = summarize_records(token_records)
    write_json(out_dir / "token_statistics.json", token_summary)

    fingerprint = baseline_config_fingerprint(project_root, method, dataset_name, frozen_config)
    write_json(out_dir / "config_fingerprint.json", fingerprint)

    run_summary = {
        "dataset": dataset_name,
        "method": METHOD_LABELS[method],
        "requirements": len(requirement_ids),
        "predictions_written": len(predictions),
        "llm_scored_k": llm_scored_k,
        "retriever": retriever,
        "real_api_calls": diagnostics["cache_misses"],
        "max_api_calls": max_api_calls,
        "fallback_predictions": 0,
        "parse_errors": diagnostics["parse_errors"],
        "empty_responses": diagnostics["empty_responses"],
        "estimated_cost_usd": token_summary["estimated_cost_usd"],
        "runtime_seconds": runtime_seconds,
        "main_binary_protocol": "top_k",
    }
    write_json(out_dir / "run_summary.json", run_summary)
    append_audit_ledger(project_root, dataset_name, run_summary, out_dir)
    return run_summary


def resolve_methods_and_write_flags(
    method_arg: str, live_api: bool,
) -> tuple[tuple[str, ...], bool, bool]:
    """(methods, write_ir_neural_now, write_llm_now) for one --method/
    --live-api combination. Pure and CLI-independent so this dispatch
    logic -- in particular the guarantee that --method llm can never
    reach an IR/Neural write and --method ir_neural can never reach an
    LLM write -- is directly unit-testable without invoking the API or
    the argparse layer."""
    methods = {
        "ir_neural": IR_METHODS + NEURAL_METHODS,
        "llm": LLM_METHODS,
    }.get(method_arg, ALL_METHODS)
    # ir_neural is zero-cost/local (no API involved either way), so an
    # ir_neural-only invocation writes directly -- there is nothing for
    # --live-api to gate. Direct Prompting/Generic RAG-LLM (whether
    # selected via --method llm or --method all) always require
    # --live-api and their own per-method confirmation prompt to write.
    any_ir_neural_selected = any(m in (IR_METHODS + NEURAL_METHODS) for m in methods)
    write_ir_neural_now = method_arg == "ir_neural" or (live_api and any_ir_neural_selected)
    any_llm_selected = any(m in LLM_METHODS for m in methods)
    write_llm_now = live_api and any_llm_selected
    return methods, write_ir_neural_now, write_llm_now


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, choices=(*ALLOWED_DATASETS, "all"),
        help="A single dataset, or 'all' to run eTour, eANCI, iTrust, "
        "LibEST in that order.",
    )
    parser.add_argument(
        "--method", choices=("all", "ir_neural", "llm"), default="all",
        help="'ir_neural' restricts to TFIDF/BM25/LSI/Sentence-BERT/"
        "CodeBERT only (no Direct Prompting/Generic RAG-LLM); zero-cost/"
        "local, so this mode writes automatically -- no --live-api needed. "
        "'llm' restricts to Direct Prompting/Generic RAG-LLM only (no IR/"
        "Neural at all -- this mode never touches their output "
        "directories); still requires --live-api (and its own "
        "confirmation prompt per method) to actually write. 'all' "
        "(default) includes every method, each gated the same way its "
        "own restricted mode gates it.",
    )
    parser.add_argument(
        "--live-api", action="store_true",
        help="Required for Direct Prompting/Generic RAG-LLM to actually "
        "write (real API calls, gated by a confirmation prompt per "
        "method). Irrelevant when --method ir_neural, which always writes.",
    )
    parser.add_argument(
        "--max-api-calls", type=int, default=None,
        help="Hard cap on real API calls for Direct Prompting / Generic "
        "RAG-LLM (checked against the preflight before any call is made; "
        "does not apply to IR/Neural, which never call an API).",
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "datasets.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = config["_project_root"]
    frozen_config = load_final_framework_config(project_root)
    loader = DatasetLoader(args.config)

    dataset_names = list(ALLOWED_DATASETS) if args.dataset == "all" else [
        select_dataset(loader, args.dataset)
    ]
    methods, write_ir_neural_now, write_llm_now = resolve_methods_and_write_flags(
        args.method, args.live_api
    )

    for dataset_name in dataset_names:
        dataset = loader.load(dataset_name)
        mode = "write" if (write_ir_neural_now or write_llm_now) else "dry-run"
        print(f"\n=== {EXPERIMENT_ROOT} baseline {mode}: {dataset_name} "
              f"(methods={args.method}) ===")

        for method in methods:
            if method in LLM_METHODS:
                report = dry_run_llm(dataset, dataset_name, method, config, frozen_config)
            else:
                report = dry_run_ir_or_neural(dataset, method, config)
            out_dir = output_dir_for_baseline(project_root, dataset_name, method)
            fingerprint = baseline_config_fingerprint(project_root, method, dataset_name, frozen_config)

            print(f"\n--- {report['method']} ---")
            for key, value in report.items():
                if key in ("diagnostics", "preview_overall_metrics", "method"):
                    continue
                print(f"  {key}: {value}")
            if "preview_overall_metrics" in report:
                print(f"  preview_overall_metrics: {report['preview_overall_metrics']}")
            print(f"  output_dir: {out_dir}")
            print(f"  git_commit: {fingerprint['git_commit']}  git_dirty: {fingerprint['git_dirty']}")

        for method in methods:
            if method in LLM_METHODS:
                if not write_llm_now:
                    continue
                summary = write_llm_result(
                    project_root, dataset, dataset_name, method, config, frozen_config,
                    max_api_calls=args.max_api_calls,
                )
            else:
                if not write_ir_neural_now:
                    continue
                summary = write_ir_or_neural_result(project_root, dataset, dataset_name, method, config)
            if summary is not None:
                print(f"{dataset_name}/{METHOD_LABELS[method]}: wrote "
                      f"{summary['predictions_written']} predictions "
                      f"(real_api_calls={summary['real_api_calls']}, "
                      f"cost=${summary['estimated_cost_usd']:.4f}) -> "
                      f"{output_dir_for_baseline(project_root, dataset_name, method)}")

    if not write_ir_neural_now and not write_llm_now:
        print("\nNo API calls made, nothing written (dry-run only).")
        return 0

    table_path = aggregate_main_experiment_tables(project_root)
    print(f"\nUpdated {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
