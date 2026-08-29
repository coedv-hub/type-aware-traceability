#!/usr/bin/env python3
"""Run the IST Proposed Framework on a controlled sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_llm_baselines import (  # noqa: E402
    create_client,
    load_retriever_candidates,
    pricing_for,
    safe_name,
    sample_requirements,
    select_dataset,
)
from src.data.loader import DatasetLoader, TraceDataset  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    evaluate_rankings,
    mean_reciprocal_rank,
    precision_at_k,
    precision_recall_f1,
    recall_at_k,
)
from src.framework.pipeline import (  # noqa: E402
    FrameworkConfig,
    TypeAwareTraceabilityPipeline,
)
from src.framework.llm_support import (  # noqa: E402
    FrameworkLLMService,
    IncompleteBatchResponseError,
    RunBudgetExceededError,
    chunk_cache_suffix,
)
from src.framework.alignment import (  # noqa: E402
    ALIGNMENT_BATCH_SIZE,
    AlignmentPromptBuilder,
    BATCH_ALIGNMENT_MAX_COMPLETION_TOKENS,
    BidirectionalSemanticAligner,
)
from src.framework.understanding import (  # noqa: E402
    TypeAwareRequirementUnderstanding,
)
from src.framework.verification import (  # noqa: E402
    BATCH_VERIFICATION_MAX_COMPLETION_TOKENS,
    VERIFICATION_BATCH_SIZE,
    SelfReflectiveVerifier,
    VerificationPromptBuilder,
)
from src.llm.cache import CacheIdentity  # noqa: E402
from src.llm.cache import LLMCache  # noqa: E402
from src.llm.token_statistics import (  # noqa: E402
    TokenRecord,
    estimate_cost,
    records_frame,
    summarize_records,
)
from src.retrieval.dcar import DynamicContextAdaptiveRetriever  # noqa: E402
from src.retrieval.summarizer import (  # noqa: E402
    CachedCodeSummarizer,
    CodeSummaryPromptBuilder,
    SUMMARY_CACHE_MODULE,
    SUMMARY_CACHE_PROMPT,
    SUMMARY_CACHE_REQUIREMENT_ID,
)
from src.utils import ensure_directory, load_config, write_json  # noqa: E402


METHOD_NAME = "Proposed Framework"
RETRIEVERS = ("tfidf", "bm25", "lsi", "sentencebert", "codebert", "hybrid", "hybrid3")
IR_RETRIEVERS = {"tfidf", "bm25", "lsi"}
DEFAULT_LLM_TOP_K = 10
GROUPS = ("Overall", "FR", "NFR", "Mixed")
SCORING_VARIANTS = {
    "A": "current_multiplicative_calibration",
    "B": "alignment_primary_weighted_sum",
    "C": "verification_gated_calibration",
}
DCAR_SELECTION_STRATEGIES = ("rank_fallback", "rank_window_diverse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="eTour")
    parser.add_argument("--provider", choices=["openai"], default="openai")
    parser.add_argument("--model", help="Override provider default model name.")
    parser.add_argument(
        "--live-api",
        action="store_true",
        help="Explicitly allow OpenAI API calls. Omit for no-api local mode.",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="Write metrics, rankings, token statistics, and paper tables.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing proposed-framework predictions.csv.",
    )
    parser.add_argument(
        "--retriever",
        choices=RETRIEVERS,
        default=None,
        help=(
            "Existing Stage 2/3 ranking used to select framework candidates. "
            "Ignored by the frozen formal configuration."
        ),
    )
    parser.add_argument(
        "--llm-top-k",
        type=int,
        default=None,
        help=(
            "Maximum number of DCAR-selected candidates scored by the framework. "
            "Ignored by the frozen formal configuration."
        ),
    )
    parser.add_argument(
        "--candidate-pool-k",
        type=int,
        default=None,
        help=(
            "Retriever candidate-pool size shared with Direct/RAG. "
            "Defaults to --llm-top-k for backward compatibility."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Final decision threshold. Ignored by the frozen formal configuration.",
    )
    parser.add_argument(
        "--scoring-variant",
        choices=SCORING_VARIANTS,
        default=None,
        help=(
            "Final scoring variant. Ignored by the frozen formal configuration."
        ),
    )
    parser.add_argument(
        "--dcar-selection-strategy",
        choices=DCAR_SELECTION_STRATEGIES,
        default=None,
        help=(
            "Candidate selection strategy inside DCAR. Ignored by the frozen "
            "formal configuration."
        ),
    )
    parser.add_argument(
        "--exploratory-config",
        action="store_true",
        help=(
            "Use explicitly supplied retrieval/scoring parameters instead of "
            "configs/final_framework_config.yaml. This mode is for validation "
            "or debugging only and must not be used for formal paper results."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "datasets.yaml"),
    )
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--rpm-limit", type=int, default=10)
    parser.add_argument("--tpm-limit", type=int, default=20_000)
    parser.add_argument("--min-call-interval", type=float, default=3.0)
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=None,
        help="Hard cap on real (non-cached) API calls for this run. The "
        "call that would exceed the cap is refused before it happens "
        "(RunBudgetExceededError) and the run fails with no predictions "
        "written.",
    )
    return parser.parse_args()


def provider_model(args: argparse.Namespace, config: dict[str, Any]) -> tuple[str, str, bool]:
    item = config["llm"]["providers"][args.provider]
    model = args.model or str(item["default_model"])
    api_key_env = str(item["api_key_env"])
    return model, api_key_env, bool(os.environ.get(api_key_env))


def output_dir_for(
    project_root: Path,
    dataset: str,
    provider: str,
    model: str,
    retriever: str,
    top_k: int,
    candidate_pool_k: int | None = None,
    dcar_selection_strategy: str = "rank_fallback",
    sample_size: int | None = None,
    *,
    results_subdir: str = "proposed_framework",
) -> Path:
    """`results_subdir` defaults to "proposed_framework" (every existing
    caller, unchanged). A separate, isolated experiment (e.g. the frozen-v3
    full-corpus retrieval run) passes its own subdir name so its outputs
    can never collide with -- or be resumed from -- an older/invalidated
    results/proposed_framework/... run."""
    base = (
        project_root
        / "results"
        / results_subdir
        / dataset
        / provider
        / safe_name(model)
        / "retriever_top_k"
        / retriever
    )
    if candidate_pool_k is None or candidate_pool_k == top_k:
        output = base / f"top_{top_k}"
    else:
        output = base / f"pool_{candidate_pool_k}" / f"dcar_top_{top_k}"
    if dcar_selection_strategy != "rank_fallback":
        output = output / f"selection_{dcar_selection_strategy}"
    # Sampled smoke tests must never share a resume/output directory with a
    # complete dataset run.  Otherwise --resume could silently reuse a sample
    # prediction as if it belonged to the formal experiment.
    if sample_size is not None:
        output = output / "samples" / f"sample_size_{sample_size}"
    return output


def resolve_candidate_pool_k(args: argparse.Namespace) -> int:
    candidate_pool_k = (
        args.llm_top_k if args.candidate_pool_k is None else args.candidate_pool_k
    )
    if candidate_pool_k < 1:
        raise SystemExit("--candidate-pool-k must be at least 1.")
    if candidate_pool_k < args.llm_top_k:
        raise SystemExit("--candidate-pool-k must be >= --llm-top-k.")
    return candidate_pool_k


def load_final_framework_config(project_root: Path) -> dict[str, Any]:
    """Load and validate the one frozen configuration for formal reruns."""
    path = project_root / "configs" / "final_framework_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen framework configuration is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    retrieval = config.get("retrieval", {})
    scoring = config.get("scoring", {})
    evaluation = config.get("evaluation", {})
    required = {
        "retrieval.retriever": retrieval.get("retriever"),
        "retrieval.candidate_pool_k": retrieval.get("candidate_pool_k"),
        "retrieval.dcar_selected_k": retrieval.get("dcar_selected_k"),
        "retrieval.dcar_selection_strategy": retrieval.get("dcar_selection_strategy"),
        "scoring.selected_variant": scoring.get("selected_variant"),
        "scoring.threshold": scoring.get("threshold"),
        "evaluation.main_binary_protocol": evaluation.get("main_binary_protocol"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Frozen framework configuration is incomplete; missing "
            + ", ".join(missing)
        )
    if retrieval["retriever"] not in RETRIEVERS:
        raise ValueError(f"Unsupported frozen retriever: {retrieval['retriever']!r}")
    if retrieval["dcar_selection_strategy"] not in DCAR_SELECTION_STRATEGIES:
        raise ValueError(
            "Unsupported frozen DCAR selection strategy: "
            f"{retrieval['dcar_selection_strategy']!r}"
        )
    if scoring["selected_variant"] not in SCORING_VARIANTS:
        raise ValueError(
            f"Unsupported frozen scoring variant: {scoring['selected_variant']!r}"
        )
    if evaluation["main_binary_protocol"] not in {"top_k", "threshold"}:
        raise ValueError(
            "Frozen main binary protocol must be either 'top_k' or 'threshold'."
        )
    return config


def resolve_framework_arguments(
    args: argparse.Namespace, project_root: Path
) -> tuple[dict[str, Any] | None, str]:
    """Resolve formal frozen settings or explicit exploratory settings.

    Formal runs cannot silently drift from the configuration selected on the
    frozen validation split. An explicit exploratory flag is required for
    any alternative setting.
    """
    supplied = {
        "--retriever": args.retriever,
        "--llm-top-k": args.llm_top_k,
        "--candidate-pool-k": args.candidate_pool_k,
        "--threshold": args.threshold,
        "--scoring-variant": args.scoring_variant,
        "--dcar-selection-strategy": args.dcar_selection_strategy,
    }
    if args.exploratory_config:
        args.retriever = args.retriever or "sentencebert"
        args.llm_top_k = args.llm_top_k or DEFAULT_LLM_TOP_K
        args.threshold = 0.20 if args.threshold is None else args.threshold
        args.scoring_variant = args.scoring_variant or "A"
        args.dcar_selection_strategy = (
            args.dcar_selection_strategy or "rank_fallback"
        )
        return None, "exploratory"

    overrides = [option for option, value in supplied.items() if value is not None]
    if overrides:
        raise SystemExit(
            "Formal runs always use configs/final_framework_config.yaml. "
            "Remove "
            + ", ".join(overrides)
            + " or add --exploratory-config for a non-formal validation/debug run."
        )

    frozen = load_final_framework_config(project_root)
    retrieval = frozen["retrieval"]
    scoring = frozen["scoring"]
    args.retriever = str(retrieval["retriever"])
    args.candidate_pool_k = int(retrieval["candidate_pool_k"])
    args.llm_top_k = int(retrieval["dcar_selected_k"])
    args.dcar_selection_strategy = str(retrieval["dcar_selection_strategy"])
    args.scoring_variant = str(scoring["selected_variant"])
    args.threshold = float(scoring["threshold"])
    return frozen, "frozen"


def dcar_base_k_for(candidate_pool_k: int, dcar_selected_k: int) -> int:
    if candidate_pool_k <= dcar_selected_k:
        return dcar_selected_k
    return max(1, min(dcar_selected_k, dcar_selected_k // 2))


def completed_pairs(
    path: Path, resume: bool, mode: str
) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    if not resume or not path.is_file():
        return pd.DataFrame(), set()
    existing = pd.read_csv(path)
    required = {"Req_ID", "Code_File"}
    if not required.issubset(existing.columns):
        return pd.DataFrame(), set()
    resumable = existing
    if "Mode" in existing.columns:
        resumable = existing.loc[existing["Mode"].astype(str).eq(mode)]
    pairs = {
        (str(row.Req_ID), str(row.Code_File))
        for row in resumable.itertuples(index=False)
    }
    return resumable.copy(), pairs


def ensure_resume_scope(output_dir: Path, sample_size: int | None, resume: bool) -> None:
    """Reject resume when the directory was produced for a different scope."""
    if not resume:
        return
    summary_path = output_dir / "run_summary.json"
    if not summary_path.is_file():
        return
    try:
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    expected_scope = "sample" if sample_size is not None else "full_dataset"
    previous_scope = previous.get("run_scope")
    if previous_scope and previous_scope != expected_scope:
        raise SystemExit(
            "Refusing --resume because the existing output directory belongs "
            f"to a {previous_scope!r} run, not {expected_scope!r}. "
            "Rerun without --resume to replace it, or use the isolated "
            "sample output directory."
        )


def cache_identity(
    client_model: str,
    provider: str,
    dataset: str,
    module: str,
    prompt: str,
    requirement_id: str,
    code_file: str = "",
    cache_prompt: str | None = None,
    cache_requirement_id: str | None = None,
) -> CacheIdentity:
    identity_prompt = prompt if cache_prompt is None else cache_prompt
    identity_requirement_id = (
        requirement_id
        if cache_requirement_id is None
        else cache_requirement_id
    )
    return CacheIdentity(
        model=client_model,
        prompt=f"[{module}]\n{identity_prompt}",
        dataset=dataset,
        requirement_id=identity_requirement_id,
        code_file=code_file or module,
        provider=provider,
    )


def batch_cache_key(
    retriever: str,
    top_k: int,
    ranked_candidates: dict[str, tuple[int, float]],
) -> str:
    ordered_ids = [
        code_id
        for code_id, _ in sorted(
            ranked_candidates.items(),
            key=lambda item: (item[1][0], item[0]),
        )
    ]
    return f"batch::{retriever}::top_{top_k}::" + "|".join(ordered_ids)


def estimate_framework_budget(
    *,
    dataset: TraceDataset,
    requirement_ids: list[str],
    candidates: dict[str, dict[str, tuple[int, float]]],
    resume_pairs: set[tuple[str, str]],
    client: Any,
    cache: LLMCache,
    input_price: float,
    output_price: float,
    retriever_name: str,
    llm_top_k: int,
    candidate_pool_k: int,
    dcar_selection_strategy: str = "rank_fallback",
    alignment_prompt_builder: "AlignmentPromptBuilder | None" = None,
    verification_prompt_builder: "VerificationPromptBuilder | None" = None,
    understanding: "TypeAwareRequirementUnderstanding | None" = None,
    understanding_cache_module: str = "understanding",
    skip_verification: bool = False,
) -> dict[str, Any]:
    """alignment_prompt_builder/verification_prompt_builder default to None
    (the frozen AlignmentPromptBuilder/VerificationPromptBuilder), which is
    what every existing caller gets -- pass an alternate builder (e.g. an
    isolated evidence-anchored variant) to get an accurate preflight for
    that variant's actual (differently sized) prompts instead of silently
    estimating against the wrong prompt text.

    understanding/understanding_cache_module default to None/"understanding"
    (the frozen TypeAwareRequirementUnderstanding under its normal cache
    module name), matching every existing caller exactly. Pass an alternate
    understanding object (e.g. GenericRequirementUnderstanding, for the w/o
    Type Understanding ablation) plus its own cache module name to preflight
    that variant's actual prompt/cache identity instead of silently checking
    the wrong one.

    skip_verification=True (used only by the w/o Verification ablation)
    omits the verification category entirely -- that ablation never calls
    verification at all, live or cached, so estimating it here would
    overstate its cost and call count.

    Real-cache-only preflight accuracy: `preflight_llm_service` below is
    live_api=True (so cache.get() only ever accepts a genuine, non-
    estimated response -- matching a real live run) with max_api_calls=0,
    so FrameworkLLMService.complete_json raises RunBudgetExceededError
    BEFORE any real network call whenever there is no such cache entry --
    this can never trigger the API. It is used as the DEFAULT understanding
    object's llm_service (so requirement_profile carries the real,
    previously-cached extracted fields when available, instead of an
    empty-payload local approximation) and for the "preview" aligner used
    only to build a representative alignment_evidence for the verification
    prompt below (never for the alignment category's own cache check,
    which already uses the real, passed-in alignment_prompt_builder
    directly). Both call sites fall back to today's exact always-local
    behavior on any miss/incompatibility, so this can only ever IMPROVE
    preflight accuracy for every caller (v1/v2/v3 and every ablation
    alike), never regress it."""
    preflight_llm_service = FrameworkLLMService(
        client=client, cache=cache, dataset=dataset.name, live_api=True, max_api_calls=0,
    )
    understanding = understanding or TypeAwareRequirementUnderstanding(
        llm_service=preflight_llm_service
    )
    # The retriever's OWN internal summarizer (used to build every
    # candidate's CodeSemanticProfile -- functions/classes/quality_related_
    # elements/selected_context, all embedded via repr into the alignment
    # prompt) defaulted to a local, non-LLM regex heuristic here, which
    # measurably differs from the real LLM-authored summary a live run
    # actually cached (verified byte-for-byte: e.g. est.c's real cached
    # description is a genuine natural-language summary, vs the local
    # heuristic's "est.c | functions: ..." stub) -- this was the dominant
    # cause of false alignment-category cache misses, not candidate
    # ordering (which was verified to already match). Wiring a real-
    # cache-only summarizer here fixes that for every caller.
    retriever = DynamicContextAdaptiveRetriever(
        summarizer=CachedCodeSummarizer(llm_service=preflight_llm_service),
        base_k=dcar_base_k_for(candidate_pool_k, llm_top_k),
        max_k=llm_top_k,
        selection_strategy=dcar_selection_strategy,
    )
    local_retriever = DynamicContextAdaptiveRetriever(
        base_k=dcar_base_k_for(candidate_pool_k, llm_top_k),
        max_k=llm_top_k,
        selection_strategy=dcar_selection_strategy,
    )
    summarizer_prompt_builder = CodeSummaryPromptBuilder()
    alignment_prompt_builder = alignment_prompt_builder or AlignmentPromptBuilder()
    verification_prompt_builder = verification_prompt_builder or VerificationPromptBuilder()
    aligner = BidirectionalSemanticAligner(
        llm_service=preflight_llm_service, prompt_builder=alignment_prompt_builder
    )
    # The offline fallback must be the SAME understanding CLASS the caller
    # actually configured (e.g. GenericRequirementUnderstanding for the
    # w/o Type Understanding ablation), never hardcoded to the frozen
    # type-aware default -- falling back to TypeAwareRequirementUnderstanding
    # here would silently reconstruct a real FR/NFR/Mixed profile on any
    # cache miss, reintroducing exactly the type-leakage a type-neutral
    # ablation is meant to remove. Every understanding class in use here
    # shares the (prompt_builder=None, llm_service=None) constructor
    # signature, so this works generically for any of them.
    local_understanding = type(understanding)(llm_service=None)
    local_aligner = BidirectionalSemanticAligner()
    code_corpus = {
        code_id: artifact.text for code_id, artifact in dataset.code_files.items()
    }
    call_counts = {
        "requirement_understanding": 0,
        "code_summarization": 0,
        "alignment": 0,
        "verification": 0,
    }
    category_stats = {
        name: {
            "estimated_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "expected_api_calls": 0,
            "estimated_prompt_tokens": 0,
            "estimated_completion_tokens": 0,
        }
        for name in call_counts
    }
    cache_hits = 0
    prompt_tokens = 0
    completion_tokens = 0
    estimated_calls = 0
    expected_api_calls = 0
    pending_requirements = 0

    def account(
        *,
        category: str,
        module: str,
        prompt: str,
        requirement_id: str,
        code_file: str = "",
        cache_prompt: str | None = None,
        cache_requirement_id: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> None:
        nonlocal cache_hits, prompt_tokens, completion_tokens
        nonlocal estimated_calls, expected_api_calls
        estimated_calls += 1
        category_stats[category]["estimated_calls"] += 1
        identity = cache_identity(
            client.model,
            client.provider,
            dataset.name,
            module,
            prompt,
            requirement_id,
            code_file,
            cache_prompt,
            cache_requirement_id,
        )
        if cache.get(identity, allow_estimated=False) is not None:
            cache_hits += 1
            category_stats[category]["cache_hits"] += 1
            return
        expected_api_calls += 1
        category_stats[category]["cache_misses"] += 1
        category_stats[category]["expected_api_calls"] += 1
        estimated_prompt_tokens = client.estimate_tokens(prompt)
        estimated_completion_tokens = max_completion_tokens or client.max_completion_tokens
        prompt_tokens += estimated_prompt_tokens
        completion_tokens += estimated_completion_tokens
        category_stats[category]["estimated_prompt_tokens"] += estimated_prompt_tokens
        category_stats[category]["estimated_completion_tokens"] += estimated_completion_tokens

    active_requirement_ids = [
        requirement_id
        for requirement_id in requirement_ids
        if any(
            (requirement_id, code_id) not in resume_pairs
            for code_id in candidates[requirement_id]
        )
    ]
    pending_summary_code_ids = sorted(
        {
            code_id
            for requirement_id in active_requirement_ids
            for code_id in candidates[requirement_id]
            if code_id in code_corpus
        }
    )
    for code_id in pending_summary_code_ids:
        prompt = summarizer_prompt_builder.build_summary_prompt(
            code_id,
            code_corpus[code_id],
        )
        call_counts["code_summarization"] += 1
        account(
            category="code_summarization",
            module=SUMMARY_CACHE_MODULE,
            prompt=prompt,
            requirement_id=SUMMARY_CACHE_REQUIREMENT_ID,
            code_file=code_id,
            cache_prompt=SUMMARY_CACHE_PROMPT,
            cache_requirement_id=SUMMARY_CACHE_REQUIREMENT_ID,
        )

    for requirement_id in active_requirement_ids:
        requirement = dataset.requirements[requirement_id]
        requirement_candidates = candidates[requirement_id]
        pending_requirements += 1
        try:
            requirement_profile = understanding.understand(
                requirement.text,
                requirement.req_type,
                requirement_id,
            )
        except RunBudgetExceededError:
            # No real cache entry for this requirement's understanding
            # prompt -- fall back to today's exact local (empty-payload)
            # approximation; the miss itself is what account() below
            # records for the requirement_understanding category.
            requirement_profile = local_understanding.understand(
                requirement.text,
                requirement.req_type,
                requirement_id,
            )
        call_counts["requirement_understanding"] += 1
        account(
            category="requirement_understanding",
            module=understanding_cache_module,
            prompt=requirement_profile.prompt,
            requirement_id=requirement_id,
        )
        try:
            evidence_context = retriever.retrieve(
                requirement_profile=requirement_profile,
                code_corpus=code_corpus,
                ranked_candidates=requirement_candidates,
            )
        except RunBudgetExceededError:
            # At least one candidate's summary is missing from the real
            # cache (summarize_corpus() raises on the first such miss,
            # losing every result in that dict comprehension) -- fall back
            # to today's exact local (non-LLM heuristic) summarizer for
            # this requirement's candidates entirely.
            evidence_context = local_retriever.retrieve(
                requirement_profile=requirement_profile,
                code_corpus=code_corpus,
                ranked_candidates=requirement_candidates,
            )
        batch_key = batch_cache_key(
            retriever_name,
            candidate_pool_k,
            requirement_candidates,
        )
        candidates_tuple = tuple(evidence_context.candidates)
        code_profiles = tuple(
            aligner.build_code_profile(
                candidate.code_id,
                summary=candidate.summary,
                selected_context=candidate.selected_context,
            )
            for candidate in candidates_tuple
        )
        # Mirror align_requirement_batch()'s deterministic chunking exactly
        # (one LLM call per ALIGNMENT_BATCH_SIZE candidates, never one big
        # call for the whole requirement) so this preflight's call count and
        # cost estimate reflect the real, batch-truncation-safe mechanism.
        for start in range(0, len(code_profiles), ALIGNMENT_BATCH_SIZE):
            chunk_candidates = candidates_tuple[start : start + ALIGNMENT_BATCH_SIZE]
            chunk_profiles = code_profiles[start : start + ALIGNMENT_BATCH_SIZE]
            call_counts["alignment"] += 1
            account(
                category="alignment",
                module="alignment_requirement_batch_chunked_b" + str(ALIGNMENT_BATCH_SIZE),
                prompt=alignment_prompt_builder.build_batch_prompt(
                    requirement_profile,
                    chunk_candidates,
                    chunk_profiles,
                ),
                requirement_id=requirement_id,
                code_file=chunk_cache_suffix(
                    batch_key, tuple(p.code_id for p in chunk_profiles)
                ),
                max_completion_tokens=BATCH_ALIGNMENT_MAX_COMPLETION_TOKENS,
            )
        if not skip_verification:
            try:
                alignment_evidence = aligner.align(
                    requirement_profile=requirement_profile,
                    evidence_context=evidence_context,
                )
            except (RunBudgetExceededError, IncompleteBatchResponseError):
                # No complete real cache entry for this requirement's real
                # alignment prompt (or an incompatible prompt_builder for
                # this preview, e.g. a unidirectional one) -- fall back to
                # today's exact local (non-LLM) approximation. This never
                # affects the alignment category's own cache accounting
                # above, which already checks the real, passed-in
                # alignment_prompt_builder directly; it only affects how
                # representative the verification-prompt preview below is.
                alignment_evidence = local_aligner.align(
                    requirement_profile=requirement_profile,
                    evidence_context=evidence_context,
                )
            for start in range(0, len(alignment_evidence), VERIFICATION_BATCH_SIZE):
                chunk_evidence = alignment_evidence[start : start + VERIFICATION_BATCH_SIZE]
                call_counts["verification"] += 1
                account(
                    category="verification",
                    module="verification_requirement_batch_chunked_b" + str(VERIFICATION_BATCH_SIZE),
                    prompt=verification_prompt_builder.build_requirement_batch_prompt(
                        requirement_profile,
                        chunk_evidence,
                    ),
                    requirement_id=requirement_id,
                    code_file=chunk_cache_suffix(
                        batch_key, tuple(e.code_id for e in chunk_evidence)
                    ),
                    max_completion_tokens=BATCH_VERIFICATION_MAX_COMPLETION_TOKENS,
                )

    total_tokens = prompt_tokens + completion_tokens
    return {
        "selected_requirements": len(requirement_ids),
        "pending_requirements": pending_requirements,
        "candidate_pool_pairs": sum(len(candidates[req_id]) for req_id in requirement_ids),
        "pending_pairs": sum(
            (requirement_id, code_id) not in resume_pairs
            for requirement_id in requirement_ids
            for code_id in candidates[requirement_id]
        ),
        "estimated_framework_llm_calls": estimated_calls,
        "call_counts": call_counts,
        "category_stats": category_stats,
        "summary_unique_code_files": len(pending_summary_code_ids),
        "candidate_pool_k": candidate_pool_k,
        "dcar_selected_k": llm_top_k,
        "dcar_selection_strategy": dcar_selection_strategy,
        "summary_cache_hits": category_stats["code_summarization"]["cache_hits"],
        "summary_cache_misses": category_stats["code_summarization"]["cache_misses"],
        "summary_api_calls": category_stats["code_summarization"]["expected_api_calls"],
        "summary_reused": category_stats["code_summarization"]["cache_hits"],
        "cache_hits": cache_hits,
        "expected_api_calls": expected_api_calls,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "estimated_total_tokens": total_tokens,
        "estimated_cost_usd": estimate_cost(
            prompt_tokens,
            completion_tokens,
            input_price,
            output_price,
        ),
    }


def print_preflight_and_confirm(
    *,
    dataset: str,
    budget: dict[str, Any],
) -> bool:
    print("\nProposed Framework live-api preflight budget")
    print(f"  Dataset: {dataset}")
    print(f"  Selected requirements: {budget['selected_requirements']}")
    print(f"  Pending requirements after resume: {budget['pending_requirements']}")
    print(f"  Candidate-pool pairs: {budget['candidate_pool_pairs']}")
    print(f"  Candidate pool k: {budget['candidate_pool_k']}")
    print(f"  DCAR selected k: {budget['dcar_selected_k']}")
    print(f"  DCAR selection strategy: {budget['dcar_selection_strategy']}")
    print(f"  Pending pairs after resume: {budget['pending_pairs']}")
    print("  Framework call categories:")
    labels = {
        "requirement_understanding": "requirement understanding",
        "code_summarization": "code summarization",
        "alignment": "batch alignment",
        "verification": "batch verification",
    }
    for key, label in labels.items():
        stats = budget["category_stats"][key]
        print(
            f"    {label}: calls={stats['estimated_calls']}, "
            f"cache_hits={stats['cache_hits']}, "
            f"cache_misses={stats['cache_misses']}, "
            f"expected_api_calls={stats['expected_api_calls']}"
        )
    print(
        "  Summary cache detail: "
        f"unique_code_files={budget['summary_unique_code_files']}, "
        f"hits={budget['summary_cache_hits']}, "
        f"misses={budget['summary_cache_misses']}, "
        f"api_calls={budget['summary_api_calls']}, "
        f"reused={budget['summary_reused']}"
    )
    print(
        "  Estimated framework LLM calls: "
        f"{budget['estimated_framework_llm_calls']}"
    )
    print(f"  Cache hits: {budget['cache_hits']}")
    print(f"  Expected API calls: {budget['expected_api_calls']}")
    print(f"  Estimated prompt tokens: {budget['estimated_prompt_tokens']}")
    print(
        "  Estimated completion tokens: "
        f"{budget['estimated_completion_tokens']}"
    )
    print(f"  Estimated total tokens: {budget['estimated_total_tokens']}")
    print(f"  Estimated cost: ${budget['estimated_cost_usd']:.6f}")
    print(
        "  Note: Proposed Framework may include requirement understanding, "
        "code summarization, alignment, and batched verification calls."
    )
    if budget["expected_api_calls"] == 0:
        print("All framework LLM calls are cached; no new API calls are expected.")
    try:
        return input("Continue? [y/N] ").strip() == "y"
    except EOFError:
        return False


def run_framework(
    dataset: TraceDataset,
    requirement_ids: list[str],
    candidates: dict[str, dict[str, tuple[int, float]]],
    *,
    provider: str,
    model: str,
    retriever: str,
    llm_top_k: int,
    candidate_pool_k: int,
    threshold: float,
    resume_pairs: set[tuple[str, str]],
    llm_service: FrameworkLLMService | None = None,
    dcar_selection_strategy: str = "rank_fallback",
    aligner: BidirectionalSemanticAligner | None = None,
    verifier: SelfReflectiveVerifier | None = None,
    understanding: TypeAwareRequirementUnderstanding | None = None,
) -> tuple[pd.DataFrame, list[TokenRecord], dict[str, Any]]:
    """aligner/verifier/understanding default to None, which
    TypeAwareTraceabilityPipeline turns into its own default (frozen)
    BidirectionalSemanticAligner/SelfReflectiveVerifier/
    TypeAwareRequirementUnderstanding -- passing None (the default for every
    existing caller, including every formal top-k run) leaves this
    function's behavior completely unchanged. Only an isolated experiment
    that explicitly constructs and passes a custom aligner/verifier/
    understanding (e.g. with an alternate prompt_builder, or the w/o Type
    Understanding ablation's GenericRequirementUnderstanding) gets different
    prompts."""
    started = time.monotonic()
    summarizer = CachedCodeSummarizer(llm_service=llm_service)
    pipeline = TypeAwareTraceabilityPipeline(
        understanding=understanding,
        retriever=DynamicContextAdaptiveRetriever(
            summarizer=summarizer,
            base_k=dcar_base_k_for(candidate_pool_k, llm_top_k),
            max_k=llm_top_k,
            selection_strategy=dcar_selection_strategy,
        ),
        aligner=aligner,
        verifier=verifier,
        config=FrameworkConfig(decision_threshold=threshold),
        llm_service=llm_service,
    )
    code_corpus = {
        code_id: artifact.text for code_id, artifact in dataset.code_files.items()
    }
    rows: list[dict[str, Any]] = []
    token_records: list[TokenRecord] = []
    skipped_pairs = 0
    empty_contexts = 0

    for requirement_id in requirement_ids:
        requirement = dataset.requirements[requirement_id]
        pending_candidates = {
            code_id: rank_score
            for code_id, rank_score in candidates[requirement_id].items()
            if (requirement_id, code_id) not in resume_pairs
        }
        skipped_pairs += len(candidates[requirement_id]) - len(pending_candidates)
        if not pending_candidates:
            continue
        result = pipeline.run_requirement(
            requirement_id=requirement_id,
            requirement_text=requirement.text,
            requirement_type=requirement.req_type,
            code_corpus=code_corpus,
            ranked_candidates=candidates[requirement_id],
            batch_cache_key=batch_cache_key(
                retriever,
                candidate_pool_k,
                candidates[requirement_id],
            ),
        )
        empty_contexts += int(not result.evidence_context.candidates)
        prediction_lookup = {
            prediction.code_id: prediction for prediction in result.predictions
        }
        for code_id, (retriever_rank, retriever_score) in pending_candidates.items():
            prediction = prediction_lookup.get(code_id)
            if prediction is None:
                continue
            rows.append(
                {
                    "Dataset": dataset.name,
                    "Method": METHOD_NAME,
                    "Provider": provider,
                    "Model": model,
                    "Mode": "live_api" if llm_service is not None else "no_api_local",
                    "CandidateSelection": "retriever_top_k",
                    "Retriever": retriever,
                    "CandidatePoolK": candidate_pool_k,
                    "LLMTopK": llm_top_k,
                    "DCARBaseK": result.evidence_context.budget.base_k,
                    "DCARSelectedK": result.evidence_context.budget.selected_k,
                    "DCARMaxK": result.evidence_context.budget.max_k,
                    "DCARStrategy": result.evidence_context.budget.strategy,
                    "DCARSelectionStrategy": dcar_selection_strategy,
                    "Retriever_Rank": retriever_rank,
                    "Retriever_Score": retriever_score,
                    "Req_ID": requirement_id,
                    "Type": requirement.req_type,
                    "Code_File": code_id,
                    "Prediction": prediction.prediction,
                    "Confidence": prediction.final_score,
                    "Final_Score": prediction.final_score,
                    "Alignment_Score": prediction.alignment_score,
                    "Verification_Score": prediction.verification_score,
                    "Functional_Alignment_Score": (
                        prediction.alignment.functional_alignment_score
                    ),
                    "Quality_Alignment_Score": (
                        prediction.alignment.quality_alignment_score
                    ),
                    "Functional_Verification_Score": (
                        prediction.verification.functional_verification_score
                    ),
                    "Quality_Verification_Score": (
                        prediction.verification.quality_verification_score
                    ),
                    "Explanation": prediction.explanation,
                    "Alignment_Explanation": prediction.alignment.alignment_explanation,
                    "Verification_Explanation": (
                        prediction.verification.verification_explanation
                    ),
                    "Semantic_Consistency": (
                        prediction.verification.semantic_consistency.score
                    ),
                    "Evidence_Sufficiency": (
                        prediction.verification.evidence_sufficiency.score
                    ),
                    "Evidence_Grounding": (
                        prediction.verification.evidence_grounding.score
                    ),
                    "Matched_Elements": "|".join(prediction.alignment.matched_elements),
                    "Is_Positive": (requirement_id, code_id) in dataset.positive_links,
                    "Cache_Hit": False,
                    "Parse_Error": "",
                    "Empty_Response": False,
                }
            )
            if llm_service is None:
                token_records.append(
                    TokenRecord(
                        dataset=dataset.name,
                        baseline=METHOD_NAME,
                        provider=provider,
                        model=model,
                        requirement_id=requirement_id,
                        code_file=code_id,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        estimated_cost_usd=0.0,
                        token_source="no_api_local",
                        cache_hit=False,
                    )
                )

    diagnostics = {
        "runtime_seconds": time.monotonic() - started,
        "pairs_generated": len(rows),
        "skipped_resume_pairs": skipped_pairs,
        "empty_contexts": empty_contexts,
        "parse_errors": llm_service.parse_errors if llm_service else 0,
        "empty_responses": llm_service.empty_responses if llm_service else 0,
        "framework_cache_hits": llm_service.cache_hits if llm_service else 0,
        "framework_cache_misses": llm_service.cache_misses if llm_service else 0,
        "real_api_calls": llm_service.real_api_calls if llm_service else 0,
        "max_api_calls": llm_service.max_api_calls if llm_service else None,
        "api_called": (
            llm_service.client.statistics.api_calls > 0 if llm_service else False
        ),
        **summarizer.stats(),
    }
    return pd.DataFrame(rows), (
        llm_service.records if llm_service is not None else token_records
    ), diagnostics


def token_records_from_predictions(predictions: pd.DataFrame) -> list[TokenRecord]:
    """Create local no-API token records for all persisted predictions."""
    records: list[TokenRecord] = []
    for row in predictions.itertuples(index=False):
        records.append(
            TokenRecord(
                dataset=str(row.Dataset),
                baseline=str(row.Method),
                provider=str(row.Provider),
                model=str(row.Model),
                requirement_id=str(row.Req_ID),
                code_file=str(row.Code_File),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                estimated_cost_usd=0.0,
                token_source="no_api_local",
                cache_hit=bool(getattr(row, "Cache_Hit", False)),
            )
        )
    return records


def evaluate_framework_predictions(
    dataset: TraceDataset,
    requirement_ids: list[str],
    predictions: pd.DataFrame,
    config: dict[str, Any],
    runtime_seconds: float,
    retriever: str,
    llm_top_k: int,
    candidate_pool_k: int,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rankings: dict[str, list[tuple[str, float]]] = {}
    ranked_rows: list[dict[str, Any]] = []
    for requirement_id in requirement_ids:
        rows = predictions.loc[predictions["Req_ID"].astype(str).eq(requirement_id)]
        pairs = sorted(
            zip(rows["Code_File"], rows["Final_Score"]),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
        rankings[requirement_id] = [
            (str(code_file), float(score)) for code_file, score in pairs
        ]
        lookup = rows.set_index("Code_File")
        for rank, (code_file, score) in enumerate(rankings[requirement_id], start=1):
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
                    "CandidateSelection": "retriever_top_k",
                    "Retriever": retriever,
                    "CandidatePoolK": candidate_pool_k,
                    "LLMTopK": llm_top_k,
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
    metrics.insert(0, "Method", METHOD_NAME)
    metrics.insert(0, "Dataset", dataset.name)
    metrics["RuntimeSeconds"] = runtime_seconds
    metrics["BinaryTopK"] = int(evaluation["binary_top_k"])
    metrics["BinaryProtocol"] = "top_k"
    metrics["BinaryThreshold"] = float("nan")
    metrics["CandidateSelection"] = "retriever_top_k"
    metrics["Retriever"] = retriever
    metrics["CandidatePoolK"] = candidate_pool_k
    metrics["LLMTopK"] = llm_top_k
    threshold_metrics = evaluate_threshold_rankings(
        ranked=rankings,
        predictions=predictions,
        gold=dataset.ground_truth_by_requirement,
        requirement_types={
            req_id: dataset.requirements[req_id].req_type
            for req_id in requirement_ids
        },
        threshold=threshold,
        cutoffs=tuple(int(value) for value in evaluation["ranking_cutoffs"]),
    )
    threshold_metrics.insert(0, "Method", METHOD_NAME)
    threshold_metrics.insert(0, "Dataset", dataset.name)
    threshold_metrics["RuntimeSeconds"] = runtime_seconds
    threshold_metrics["BinaryTopK"] = int(evaluation["binary_top_k"])
    threshold_metrics["BinaryProtocol"] = "threshold"
    threshold_metrics["BinaryThreshold"] = threshold
    threshold_metrics["CandidateSelection"] = "retriever_top_k"
    threshold_metrics["Retriever"] = retriever
    threshold_metrics["CandidatePoolK"] = candidate_pool_k
    threshold_metrics["LLMTopK"] = llm_top_k
    return pd.DataFrame(ranked_rows), metrics, threshold_metrics


def apply_scoring_variant(
    predictions: pd.DataFrame,
    variant: str,
    threshold: float,
) -> pd.DataFrame:
    """Apply the configured final-scoring formula without changing signals."""
    output = predictions.copy()
    alignment = pd.to_numeric(output["Alignment_Score"], errors="coerce").fillna(0.0)
    verification = pd.to_numeric(
        output["Verification_Score"], errors="coerce"
    ).fillna(0.0)
    if variant == "A":
        final_score = alignment * (0.5 + 0.5 * verification)
    elif variant == "B":
        final_score = 0.7 * alignment + 0.3 * verification
    elif variant == "C":
        gate = verification.ge(0.5).map({True: 1.0, False: 0.5})
        final_score = alignment * gate
    else:
        raise ValueError(f"Unknown scoring variant: {variant}")
    output["ScoringVariant"] = variant
    output["ScoringVariantName"] = SCORING_VARIANTS[variant]
    output["Final_Score"] = final_score.clip(0.0, 1.0)
    output["Confidence"] = output["Final_Score"]
    output["Prediction"] = output["Final_Score"].ge(threshold).map(
        {True: "Yes", False: "No"}
    )
    return output


def evaluate_threshold_rankings(
    *,
    ranked: dict[str, list[tuple[str, float]]],
    predictions: pd.DataFrame,
    gold: dict[str, set[str]],
    requirement_types: dict[str, str],
    threshold: float,
    cutoffs: tuple[int, ...],
) -> pd.DataFrame:
    """Compute threshold binary metrics and score-ranking metrics."""
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
    scored = predictions.copy()
    scored["Final_Score"] = pd.to_numeric(scored["Final_Score"], errors="coerce").fillna(0.0)
    for group, group_ids in groups.items():
        eligible = [
            req_id
            for req_id in group_ids
            if req_id in ranked and req_id in gold and bool(gold[req_id])
        ]
        if not eligible:
            row: dict[str, object] = {
                "Group": group,
                "Requirements": 0,
                "Precision": float("nan"),
                "Recall": float("nan"),
                "F1": float("nan"),
                "MRR": float("nan"),
            }
            for cutoff in cutoffs:
                row[f"P@{cutoff}"] = float("nan")
                row[f"R@{cutoff}"] = float("nan")
            rows.append(row)
            continue
        subset = scored.loc[scored["Req_ID"].astype(str).isin(eligible)]
        predicted = {
            (str(row.Req_ID), str(row.Code_File))
            for row in subset.loc[subset["Final_Score"].ge(threshold)].itertuples(index=False)
        }
        gold_pairs = {
            (req_id, code_id) for req_id in eligible for code_id in gold[req_id]
        }
        precision, recall, f1 = precision_recall_f1(predicted, gold_pairs)
        row = {
            "Group": group,
            "Requirements": len(eligible),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MRR": mean_reciprocal_rank(ranked, gold, eligible),
        }
        for cutoff in cutoffs:
            row[f"P@{cutoff}"] = precision_at_k(ranked, gold, eligible, cutoff)
            row[f"R@{cutoff}"] = recall_at_k(ranked, gold, eligible, cutoff)
        rows.append(row)
    return pd.DataFrame(rows)


def update_paper_table(rows: pd.DataFrame, path: Path, dataset_order: list[str]) -> None:
    if path.is_file():
        existing = pd.read_csv(path)
        key_columns = [
            "Dataset",
            "Method",
            "Group",
            "CandidateSelection",
            "Retriever",
            "CandidatePoolK",
            "LLMTopK",
        ]
        for column in key_columns:
            if column not in existing.columns:
                existing[column] = ""
        keys = set(rows[key_columns].itertuples(index=False, name=None))
        existing = existing[
            ~existing.apply(
                lambda item: tuple(item[column] for column in key_columns) in keys,
                axis=1,
            )
        ]
        rows = pd.concat([existing, rows], ignore_index=True)
    rows["_dataset_order"] = pd.Categorical(
        rows["Dataset"], categories=dataset_order, ordered=True
    )
    rows["_group_order"] = pd.Categorical(
        rows["Group"], categories=list(GROUPS), ordered=True
    )
    rows = (
        rows.sort_values(["_dataset_order", "Method", "_group_order"])
        .drop(columns=["_dataset_order", "_group_order"])
        .reset_index(drop=True)
    )
    rows.to_csv(path, index=False)


def main_binary_protocol(config: dict[str, Any]) -> str:
    """Read only the frozen table-selection field.

    Formal runs use ``load_final_framework_config`` and therefore still require
    the complete frozen configuration.  Paper-table selection only depends on
    this one field, which keeps the helper independently testable and avoids
    coupling it to unrelated retrieval/scoring settings.
    """
    path = Path(config["_project_root"]) / "configs" / "final_framework_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen framework configuration is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        final_config = yaml.safe_load(handle) or {}
    protocol = final_config.get("evaluation", {}).get("main_binary_protocol")
    if protocol not in {"top_k", "threshold"}:
        raise ValueError(
            "Frozen framework configuration must define evaluation.main_binary_protocol "
            "as 'top_k' or 'threshold'."
        )
    return str(protocol)


def paper_table_metrics(
    metrics: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    protocol = main_binary_protocol(config)
    selected = metrics if protocol == "top_k" else threshold_metrics
    return selected.copy()


def update_cost_summary(row: pd.DataFrame, path: Path) -> None:
    if path.is_file():
        existing = pd.read_csv(path)
        key_columns = [
            "Dataset",
            "Provider",
            "Model",
            "Mode",
            "CandidateSelection",
            "Retriever",
            "CandidatePoolK",
            "LLMTopK",
        ]
        for column in key_columns:
            if column not in existing.columns:
                existing[column] = ""
        key = tuple(row.iloc[0][column] for column in key_columns)
        existing = existing[
            ~existing.apply(
                lambda item: tuple(item[column] for column in key_columns) == key,
                axis=1,
            )
        ]
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(path, index=False)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> int:
    started = time.monotonic()
    args = parse_args()
    config = load_config(args.config)
    project_root = config["_project_root"]
    frozen_config, configuration_mode = resolve_framework_arguments(
        args, project_root
    )
    if args.paper_mode and args.sample_size is not None:
        raise SystemExit(
            "--paper-mode requires a full dataset run; omit --sample-size. "
            "Use a sampled run only as a smoke test without paper-table updates."
        )
    if args.threshold < 0.0 or args.threshold > 1.0:
        raise SystemExit("--threshold must be between 0 and 1.")
    candidate_pool_k = resolve_candidate_pool_k(args)
    loader = DatasetLoader(args.config)
    dataset_name = select_dataset(loader, args.dataset)
    dataset = loader.load(dataset_name)
    seed = int(config["llm"]["random_seed"])
    requirement_ids = sample_requirements(
        dataset,
        args.sample_size,
        seed,
        paper_mode=args.paper_mode,
    )
    model, api_key_env, has_api_key = provider_model(args, config)
    mode = "live_api" if args.live_api else "no_api_local"
    llm_service: FrameworkLLMService | None = None
    if args.live_api and not has_api_key:
        print(
            f"{api_key_env} is not configured. Falling back to no-api local mode; "
            "no API call will be made."
        )
        mode = "no_api_local"
    elif args.live_api:
        client = create_client(
            args.provider,
            model,
            config,
            max_retries=args.max_retries,
            rpm_limit=args.rpm_limit,
            tpm_limit=args.tpm_limit,
            min_call_interval=args.min_call_interval,
        )
        cache = LLMCache(
            ensure_directory(project_root / config["output"]["llm_cache_dir"])
        )
        input_price, output_price = pricing_for(client.model, config)
        llm_service = FrameworkLLMService(
            client,
            cache,
            dataset.name,
            live_api=True,
            input_price=input_price,
            output_price=output_price,
            max_api_calls=args.max_api_calls,
        )
        model = client.model
        print(
            "Live API mode enabled for Proposed Framework. Calls will still "
            "use Stage 4 cache/resume before hitting OpenAI."
        )
    else:
        if has_api_key:
            print(
                f"{api_key_env} is configured, but --live-api was not set; "
                "no API call will be made."
            )
        else:
            print(
                f"{api_key_env} is not configured. Entering no-api local mode; "
                "no API call will be made."
            )

    candidates, retriever_ranking_path = load_retriever_candidates(
        project_root=project_root,
        config=config,
        dataset=dataset,
        requirement_ids=requirement_ids,
        retriever=args.retriever,
        top_k=candidate_pool_k,
    )
    candidate_pool_pairs = sum(len(candidates[req_id]) for req_id in requirement_ids)
    output_dir_path = output_dir_for(
        project_root,
        dataset.name,
        args.provider,
        model,
        args.retriever,
        args.llm_top_k,
        candidate_pool_k,
        args.dcar_selection_strategy,
        args.sample_size,
    )
    predictions_path = output_dir_path / "predictions.csv"
    ensure_resume_scope(output_dir_path, args.sample_size, args.resume)
    existing_predictions, resume_pairs = completed_pairs(
        predictions_path, args.resume, mode
    )
    if llm_service is not None:
        budget = estimate_framework_budget(
            dataset=dataset,
            requirement_ids=requirement_ids,
            candidates=candidates,
            resume_pairs=resume_pairs,
            client=llm_service.client,
            cache=llm_service.cache,
            input_price=llm_service.input_price,
            output_price=llm_service.output_price,
            retriever_name=args.retriever,
            llm_top_k=args.llm_top_k,
            candidate_pool_k=candidate_pool_k,
            dcar_selection_strategy=args.dcar_selection_strategy,
        )
        if not print_preflight_and_confirm(dataset=dataset.name, budget=budget):
            print("Aborted: no API calls were made and no formal results were written.")
            return 0

    output_dir = ensure_directory(output_dir_path)
    tables_root = ensure_directory(project_root / config["output"]["tables_dir"])
    predictions, token_records, diagnostics = run_framework(
        dataset=dataset,
        requirement_ids=requirement_ids,
        candidates=candidates,
        provider=args.provider,
        model=model,
        retriever=args.retriever,
        llm_top_k=args.llm_top_k,
        candidate_pool_k=candidate_pool_k,
        threshold=args.threshold,
        resume_pairs=resume_pairs,
        llm_service=llm_service,
        dcar_selection_strategy=args.dcar_selection_strategy,
    )
    if not existing_predictions.empty:
        predictions = pd.concat([existing_predictions, predictions], ignore_index=True)
        predictions = predictions.drop_duplicates(
            subset=["Req_ID", "Code_File"], keep="last"
        )
    predictions = apply_scoring_variant(
        predictions,
        args.scoring_variant,
        args.threshold,
    )
    atomic_csv(predictions, predictions_path)

    ranked, metrics, threshold_metrics = evaluate_framework_predictions(
        dataset=dataset,
        requirement_ids=requirement_ids,
        predictions=predictions,
        config=config,
        runtime_seconds=float(diagnostics["runtime_seconds"]),
        retriever=args.retriever,
        llm_top_k=args.llm_top_k,
        candidate_pool_k=candidate_pool_k,
        threshold=args.threshold,
    )
    atomic_csv(ranked, output_dir / "ranked_results.csv")
    atomic_csv(metrics, output_dir / "metrics.csv")
    atomic_csv(threshold_metrics, output_dir / "threshold_metrics.csv")
    all_token_records = (
        token_records
        if llm_service is not None
        else token_records_from_predictions(predictions)
    )
    token_frame = records_frame(all_token_records)
    atomic_csv(token_frame, output_dir / "token_statistics.csv")
    token_summary = summarize_records(all_token_records)
    write_json(output_dir / "token_statistics.json", token_summary)
    cost_summary = pd.DataFrame(
        [
            {
                "Dataset": dataset.name,
                "Method": METHOD_NAME,
                "Provider": args.provider,
                "Model": model,
                "Mode": mode,
                "CandidateSelection": "retriever_top_k",
                "Retriever": args.retriever,
                "CandidatePoolK": candidate_pool_k,
                "LLMTopK": args.llm_top_k,
                "PromptTokens": token_summary["prompt_tokens"],
                "CompletionTokens": token_summary["completion_tokens"],
                "TotalTokens": token_summary["total_tokens"],
                "EstimatedCostUSD": token_summary["estimated_cost_usd"],
                "RuntimeSeconds": float(diagnostics["runtime_seconds"]),
                "SummaryCacheHits": int(diagnostics["summary_cache_hits"]),
                "SummaryCacheMisses": int(diagnostics["summary_cache_misses"]),
                "SummaryAPICalls": int(diagnostics["summary_api_calls"]),
                "SummaryReused": int(diagnostics["summary_reused"]),
            }
        ]
    )
    atomic_csv(cost_summary, output_dir / "cost_summary.csv")

    if args.paper_mode:
        paper_metrics = paper_table_metrics(metrics, threshold_metrics, config)
        update_paper_table(
            paper_metrics.loc[paper_metrics["Group"].eq("Overall")].copy(),
            tables_root / "proposed_framework_overall_metrics.csv",
            loader.dataset_names,
        )
        update_paper_table(
            paper_metrics.copy(),
            tables_root / "proposed_framework_type_wise_metrics.csv",
            loader.dataset_names,
        )
        update_cost_summary(
            cost_summary,
            tables_root / "proposed_framework_cost_summary.csv",
        )

    elapsed = time.monotonic() - started
    summary = {
        "stage": "5h",
        "configuration_mode": configuration_mode,
        "frozen_configuration_path": (
            str(project_root / "configs" / "final_framework_config.yaml")
            if frozen_config is not None
            else None
        ),
        "mode": mode,
        "api_called": bool(diagnostics["api_called"]),
        "provider": args.provider,
        "model": model,
        "api_key_environment_variable": api_key_env,
        "api_key_configured": has_api_key,
        "dataset": dataset.name,
        "sample_size": len(requirement_ids),
        "run_scope": "sample" if args.sample_size is not None else "full_dataset",
        "formal_result_eligible": args.sample_size is None,
        "paper_mode": args.paper_mode,
        "resume": args.resume,
        "sampled_requirement_ids": requirement_ids,
        "candidate_selection": "retriever_top_k",
        "retriever": args.retriever,
        "retriever_ranking_path": str(retriever_ranking_path),
        "candidate_pool_k": candidate_pool_k,
        "llm_top_k": args.llm_top_k,
        "dcar_selected_k": args.llm_top_k,
        "dcar_selection_strategy": args.dcar_selection_strategy,
        "threshold": args.threshold,
        "scoring_variant": args.scoring_variant,
        "scoring_variant_name": SCORING_VARIANTS[args.scoring_variant],
        "candidate_pool_pairs": candidate_pool_pairs,
        "generated_pairs": int(diagnostics["pairs_generated"]),
        "skipped_resume_pairs": int(diagnostics["skipped_resume_pairs"]),
        "parse_errors": int(diagnostics["parse_errors"]),
        "empty_responses": int(diagnostics["empty_responses"]),
        "empty_contexts": int(diagnostics["empty_contexts"]),
        "framework_cache_hits": int(diagnostics["framework_cache_hits"]),
        "framework_cache_misses": int(diagnostics["framework_cache_misses"]),
        "real_api_calls": int(diagnostics["real_api_calls"]),
        "max_api_calls": diagnostics["max_api_calls"],
        "summary_cache_hits": int(diagnostics["summary_cache_hits"]),
        "summary_cache_misses": int(diagnostics["summary_cache_misses"]),
        "summary_memory_hits": int(diagnostics["summary_memory_hits"]),
        "summary_persistent_cache_hits": int(
            diagnostics["summary_persistent_cache_hits"]
        ),
        "summary_api_calls": int(diagnostics["summary_api_calls"]),
        "summary_reused": int(diagnostics["summary_reused"]),
        "summary_local_generations": int(diagnostics["summary_local_generations"]),
        "live_api": args.live_api,
        "runtime_seconds": elapsed,
        "token_statistics": token_summary,
        "outputs": {
            "predictions": str(predictions_path),
            "ranked_results": str(output_dir / "ranked_results.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "threshold_metrics": str(output_dir / "threshold_metrics.csv"),
            "token_statistics_csv": str(output_dir / "token_statistics.csv"),
            "token_statistics_json": str(output_dir / "token_statistics.json"),
            "cost_summary": str(output_dir / "cost_summary.csv"),
            "paper_tables": str(tables_root) if args.paper_mode else None,
        },
    }
    write_json(output_dir / "run_summary.json", summary)

    print("\nStage 5h proposed framework run summary")
    print(f"  Dataset: {dataset.name}")
    print(f"  Configuration mode: {configuration_mode}")
    print(f"  Requirements: {len(requirement_ids)}")
    print(f"  Candidate-pool pairs: {candidate_pool_pairs}")
    print(f"  Generated pairs: {diagnostics['pairs_generated']}")
    print(f"  Resume skipped pairs: {diagnostics['skipped_resume_pairs']}")
    print(f"  Summary cache hits: {diagnostics['summary_cache_hits']}")
    print(f"  Summary cache misses: {diagnostics['summary_cache_misses']}")
    print(f"  Summary API calls: {diagnostics['summary_api_calls']}")
    print(f"  API called: {bool(diagnostics['api_called'])}")
    print(f"  Output: {output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
