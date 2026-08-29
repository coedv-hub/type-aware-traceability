#!/usr/bin/env python3
"""Independent pair-classification runner/evaluator.

Scores every pair in a frozen configs/pair_classification_manifests/{dataset}.csv
manifest, decoupled from the existing top-k recommendation experiments. Does
not read or write results/, tables/, or configs/final_framework_config.yaml.
See configs/pair_classification_protocol.yaml for the full protocol this
implements.

IR/neural methods (TFIDF, BM25, LSI, Sentence-BERT, CodeBERT) are scored by
free lookup against the existing *_raw_scores.csv files, which already cover
every requirement x code_file pair. LLM-based methods (Direct Prompting,
Generic RAG-LLM, Proposed Framework) require fresh live-API calls, since the
existing experiments only ever scored a bounded top-k candidate pool per
requirement and manifest negatives mostly fall outside that pool; this
script never calls the API unless --live-api is explicitly passed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.llm import create_llm_baseline  # noqa: E402
from src.data.loader import DatasetLoader  # noqa: E402
from src.framework.alignment import BidirectionalSemanticAligner  # noqa: E402
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
from src.framework.evidence_anchored_prompts import (  # noqa: E402
    EvidenceAnchoredAlignmentPromptBuilder,
    EvidenceAnchoredVerificationPromptBuilder,
)
from src.framework.generic_understanding import (  # noqa: E402
    GenericRequirementUnderstanding,
    TypeNeutralAlignmentPromptBuilder,
    TypeNeutralVerificationPromptBuilder,
)
from src.framework.llm_support import (  # noqa: E402
    FrameworkLLMService,
    IncompleteBatchResponseError,
    RunBudgetExceededError,
)
from src.framework.null_verifier import NullVerifier  # noqa: E402
from src.framework.pipeline import FrameworkConfig, TypeAwareTraceabilityPipeline  # noqa: E402
from src.framework.understanding import TypeAwareRequirementUnderstanding  # noqa: E402
from src.framework.unidirectional_alignment import (  # noqa: E402
    RequirementToCodeOnlyAligner,
    UnidirectionalAlignmentPromptBuilder,
)
from src.framework.verification import SelfReflectiveVerifier  # noqa: E402
from src.llm.cache import LLMCache  # noqa: E402
from src.llm.prompt_templates import direct_prompt, rag_prompt  # noqa: E402
from src.retrieval.dcar import DynamicContextAdaptiveRetriever  # noqa: E402
from src.retrieval.summarizer import CachedCodeSummarizer  # noqa: E402
from src.utils import ensure_directory, load_config  # noqa: E402
from scripts.run_llm_baselines import (  # noqa: E402
    HYBRID_RETRIEVERS,
    RRF_CONSTANT,
    create_client,
    load_ranking_frame,
    pricing_for,
    prediction_score,
    safe_name,
)
from scripts.run_proposed_framework import (  # noqa: E402
    apply_scoring_variant,
    batch_cache_key,
    estimate_framework_budget,
    load_final_framework_config,
    run_framework,
)

# Display label -> internal create_llm_baseline() method key.
LLM_METHOD_KEYS = {"Direct Prompting": "direct", "Generic RAG-LLM": "rag"}

MANIFEST_DIR = PROJECT_ROOT / "configs" / "pair_classification_manifests"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "pair_classification"
DATASETS = ("eTour", "eANCI", "iTrust", "LibEST")
# Fully external held-out dataset for this track only -- see
# scripts/generate_pair_classification_manifest.py's HELDOUT_ONLY_DATASETS.
# Its manifest is 100% Split="heldout" (--split validation yields an empty
# manifest and is never a meaningful invocation for it); it must never be
# added to DATASETS itself, since DATASETS also backs the 4-dataset
# calibration scope in scripts/calibrate_pair_classification_threshold.py.
HELDOUT_ONLY_DATASETS = ("Industrial",)
RUNNABLE_DATASETS = DATASETS + HELDOUT_ONLY_DATASETS

# RAG's real, formal baseline retriever is hybrid3 (matches
# `--retriever hybrid3` used for the existing formal Direct/RAG runs).
RAG_RETRIEVER = "hybrid3"

# Pair-classification variants bypass DCAR candidate selection so every
# frozen manifest pair reaches alignment and verification. This protocol
# measures pair-level discrimination; candidate selection is evaluated
# separately by scripts/run_full_corpus_retrieval.py. The label below is
# the original variant; the paper's full configuration uses v3 below.
PROPOSED_PAIR_CLASSIFIER_LABEL = "Proposed-PairClassifier (BAN+SRCV, DCAR-bypassed)"

# Evidence-anchored alignment/verification prompt variant; see
# src/framework/evidence_anchored_prompts.py. Each variant has a distinct
# method label, output directory, and frozen threshold.
PROPOSED_PAIR_CLASSIFIER_V2_LABEL = (
    "Proposed-PairClassifier (BAN+SRCV, DCAR-bypassed, evidence_anchored_v2)"
)

# v3 = v2 + a limited, verifiable per-candidate call/dependency context
# block (src/framework/dependency_context.py, dependency_context_prompts.py)
# -- targets a different failure pattern: genuine links scored ~0 because
# the code's surface text has no lexical bridge to the requirement at all.
PROPOSED_PAIR_CLASSIFIER_V3_LABEL = (
    "Proposed-PairClassifier (BAN+SRCV, DCAR-bypassed, evidence_anchored_v3_dependency_context)"
)

# Four ablations of v3 (Full), isolated to this 1:1 pair-classification
# track only -- see src/framework/generic_understanding.py,
# unidirectional_alignment.py, null_verifier.py, and the ABLATIONS table
# below. Each keeps v3's 1:1 manifests, requirement-level validation/heldout
# split, GPT-4.1, B=3 strict batching, 3-retry/no-fallback policy, and DCAR
# bypass -- only the one named component differs from Full v3. Each gets its
# own method label (-> its own safe_name() output directory, never
# overwriting Full/v1/v2/v3 or Direct/RAG), and is calibrated with its own
# pooled global threshold on the 4 public datasets' validation split (see
# scripts/calibrate_pair_classification_threshold.py's CALIBRATION_METHODS)
# -- never v3's frozen threshold, and Industrial never participates in that
# calibration (see HELDOUT_ONLY_DATASETS above).
ABLATION_WO_TYPE_LABEL = "Proposed-PairClassifier (Ablation: w/o Type Understanding)"
ABLATION_WO_DEPENDENCY_LABEL = "Proposed-PairClassifier (Ablation: w/o Dependency Context)"
ABLATION_WO_BI_ALIGNMENT_LABEL = "Proposed-PairClassifier (Ablation: w/o Bi-Alignment)"
ABLATION_WO_VERIFICATION_LABEL = "Proposed-PairClassifier (Ablation: w/o Verification)"
ABLATION_LABELS = (
    ABLATION_WO_TYPE_LABEL,
    ABLATION_WO_DEPENDENCY_LABEL,
    ABLATION_WO_BI_ALIGNMENT_LABEL,
    ABLATION_WO_VERIFICATION_LABEL,
)

IR_NEURAL_METHODS = {
    "TFIDF": ("ir_baselines", "tfidf_raw_scores.csv"),
    "BM25": ("ir_baselines", "bm25_raw_scores.csv"),
    "LSI": ("ir_baselines", "lsi_raw_scores.csv"),
    "Sentence-BERT": ("neural_baselines", "sentencebert_raw_scores.csv"),
    "CodeBERT": ("neural_baselines", "codebert_raw_scores.csv"),
}
LLM_METHODS = (
    "Direct Prompting", "Generic RAG-LLM",
    PROPOSED_PAIR_CLASSIFIER_LABEL, PROPOSED_PAIR_CLASSIFIER_V2_LABEL,
    PROPOSED_PAIR_CLASSIFIER_V3_LABEL,
) + ABLATION_LABELS
PROPOSED_PAIR_CLASSIFIER_LABELS = (
    PROPOSED_PAIR_CLASSIFIER_LABEL, PROPOSED_PAIR_CLASSIFIER_V2_LABEL,
    PROPOSED_PAIR_CLASSIFIER_V3_LABEL,
)
ALL_METHODS = tuple(IR_NEURAL_METHODS) + LLM_METHODS

# Large enough that DCAR's own budget formula never narrows the candidate
# set below the manifest's pairs for this requirement -- see
# configs/pair_classification_protocol.yaml "bypass DCAR narrowing" note.
DCAR_BYPASS_MAX_K = 1000


def full_cartesian_hybrid_ranking(
    project_root: Path, config: dict[str, Any], dataset: Any, retriever: str = RAG_RETRIEVER
) -> dict[tuple[str, str], tuple[int, float]]:
    """Real rank+score for every (Req_ID, Code_File) pair under the RRF
    fusion of `retriever`'s components, computed over the full requirement
    x code_file cartesian product (each component's *_ranked_results.csv
    already covers every pair -- verified for eANCI: 39 x 55 = 2145 rows).
    This is the same formula run_llm_baselines.load_hybrid_retriever_candidates
    uses for the real Direct/RAG baselines, just not truncated to top-k, so
    a manifest pair gets the genuine rank/score it would have had if it had
    been within that baseline's candidate window."""
    components = HYBRID_RETRIEVERS[retriever]
    frames = {
        component: load_ranking_frame(project_root, config, dataset, component)
        for component in components
    }
    fused_scores: dict[tuple[str, str], float] = {}
    best_ranks: dict[tuple[str, str], int] = {}
    for frame in frames.values():
        for row in frame.itertuples(index=False):
            key = (str(row.Req_ID), str(row.Code_File))
            rank = int(row.Rank)
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (RRF_CONSTANT + rank)
            best_ranks[key] = min(best_ranks.get(key, rank), rank)

    result: dict[tuple[str, str], tuple[int, float]] = {}
    by_requirement: dict[str, list[tuple[str, float, int]]] = {}
    for (req_id, code_id), score in fused_scores.items():
        by_requirement.setdefault(req_id, []).append((code_id, score, best_ranks[(req_id, code_id)]))
    for req_id, entries in by_requirement.items():
        ordered = sorted(entries, key=lambda item: (-item[1], item[2], item[0]))
        for fused_rank, (code_id, score, _) in enumerate(ordered, start=1):
            result[(req_id, code_id)] = (fused_rank, score)
    return result


def load_manifest(dataset_name: str) -> pd.DataFrame:
    """(Req_ID, Code_File) is NOT guaranteed unique in the manifest: the
    with-replacement fallback (a requirement with more gold links than
    distinct non-linked code files, e.g. several LibEST requirements) can
    legitimately reuse the same negative Code_File across two different
    PairGroupIDs for the same requirement. Every consumer must key off
    PairGroupID (or row position), never off (Req_ID, Code_File) alone."""
    path = MANIFEST_DIR / f"{dataset_name}.csv"
    if not path.is_file():
        raise SystemExit(
            f"Missing manifest: {path}. Run "
            "scripts/generate_pair_classification_manifest.py first."
        )
    return pd.read_csv(path)


def output_dir_for(dataset_name: str, method: str, split: str = "all") -> Path:
    """split='all' (the default, used by every existing caller) is the
    full-manifest output path and is what the eventual full-dataset run
    will write to. A restricted split (e.g. 'validation', used to freeze a
    threshold before committing to the full run) gets its own nested
    subfolder so it can never collide with or be silently overwritten by
    the future full run, and vice versa."""
    base = OUTPUT_ROOT / dataset_name / safe_name(method)
    if split == "all":
        return base
    return base / f"_split_{split}"


def score_ir_neural(dataset_name: str, method: str, manifest: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    subdir, filename = IR_NEURAL_METHODS[method]
    raw_path = PROJECT_ROOT / config["output"][
        "results_dir" if subdir == "ir_baselines" else "neural_results_dir"
    ] / dataset_name / filename
    raw = pd.read_csv(raw_path)
    lookup = raw.set_index(["Req_ID", "Code_File"])["Score"]
    scores = []
    missing = 0
    for row in manifest.itertuples(index=False):
        key = (row.Req_ID, row.Code_File)
        if key in lookup.index:
            scores.append(float(lookup.loc[key]))
        else:
            missing += 1
            scores.append(float("nan"))
    if missing:
        raise SystemExit(
            f"{method}/{dataset_name}: {missing} manifest pair(s) missing from "
            f"{raw_path}; expected full requirement x code_file coverage."
        )
    predictions = manifest.copy()
    predictions["Score"] = scores
    predictions["Method"] = method
    predictions["Dataset"] = dataset_name
    predictions["CacheHit"] = True
    predictions["ApiCall"] = False
    return predictions


def build_candidates_for_manifest(
    method: str,
    dataset_name: str,
    dataset: Any,
    manifest: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, dict[str, tuple[int, float]]]:
    """Build the requirement -> {code_file: (rank, score)} map
    create_llm_baseline() expects, restricted to exactly the manifest's
    pairs. Direct Prompting's prompt does not use rank/score at all, so a
    placeholder is harmless there; Generic RAG-LLM's prompt embeds
    rank/score directly, so it must get the real hybrid3 RRF fusion values
    (computed over the full cartesian product, not just top-k) -- using a
    placeholder would silently produce a different prompt than the real
    RAG baseline and break cache reuse against it."""
    grouped: dict[str, list[str]] = {}
    for row in manifest.itertuples(index=False):
        grouped.setdefault(row.Req_ID, []).append(row.Code_File)

    if method == "Generic RAG-LLM":
        real_rank_score = full_cartesian_hybrid_ranking(PROJECT_ROOT, config, dataset)
        candidates: dict[str, dict[str, tuple[int, float]]] = {}
        missing = 0
        for req_id, code_ids in grouped.items():
            per_req = {}
            for code_id in code_ids:
                rank_score = real_rank_score.get((req_id, code_id))
                if rank_score is None:
                    missing += 1
                    continue
                per_req[code_id] = rank_score
            candidates[req_id] = per_req
        if missing:
            raise SystemExit(
                f"Generic RAG-LLM/{dataset_name}: {missing} manifest pair(s) "
                "missing from the full hybrid3 cartesian ranking; expected "
                "full requirement x code_file coverage from "
                "load_ranking_frame()."
            )
        return candidates

    # Direct Prompting: rank/score unused by the prompt; placeholder is fine.
    return {
        req_id: {code_id: (0, 0.0) for code_id in code_ids}
        for req_id, code_ids in grouped.items()
    }


def run_llm_method(
    dataset_name: str,
    method: str,
    manifest: pd.DataFrame,
    client: Any,
    cache: LLMCache,
    config: dict[str, Any],
    estimate_only: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every manifest pair via the same LLMBaseline used by the real
    top-k experiments (create_llm_baseline), so prompt construction, cache
    identity, and response parsing are identical -- just applied to the
    manifest's pairs instead of a retriever's top-k window. Works in both
    estimate_only (no real calls, cache + token-estimate only) and live
    (estimate_only=False) modes via the same code path.

    (Req_ID, Code_File) is NOT a unique key for the manifest: the
    with-replacement fallback can legitimately reuse the same negative
    Code_File across two different PairGroupIDs for one requirement. The
    output therefore iterates the manifest ROW BY ROW (preserving every
    row, including such repeats, with its own correct PairGroupID/Label/
    etc. read directly off that row) rather than re-joining metadata back
    via a (Req_ID, Code_File)-keyed lookup, which silently corrupts every
    column whenever the manifest has any repeat anywhere in it. Each
    distinct (Req_ID, Code_File) pair still gets scored (and, live, called)
    exactly once; a repeated row reuses that same in-memory result rather
    than re-querying the cache or the API.

    LLMBaseline.run_pair (src/baselines/llm.py) guarantees result.cache_hit
    only ever reflects a genuine, previously real LLM response -- a dry
    run's own estimated placeholder is never persisted and never counts as
    a hit, in either mode -- so cost/hit accounting here can rely on it
    directly."""
    loader = DatasetLoader(PROJECT_ROOT / "configs" / "datasets.yaml")
    dataset = loader.load(dataset_name)
    candidates = build_candidates_for_manifest(method, dataset_name, dataset, manifest, config)
    method_key = LLM_METHOD_KEYS[method]
    baseline = create_llm_baseline(method_key, client, cache, dataset, estimate_only, candidates)
    prompts_by_req_id = {req_id: baseline.build_prompts(req_id) for req_id in candidates}

    rows: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    estimated_prompt_tokens = 0
    pair_results: dict[tuple[str, str], Any] = {}

    for row in manifest.itertuples(index=False):
        req_id = row.Req_ID
        code_id = row.Code_File
        key = (req_id, code_id)
        result = pair_results.get(key)
        if result is None:
            prompt = prompts_by_req_id[req_id][code_id]
            result = baseline.run_pair(req_id, code_id, prompt)
            pair_results[key] = result
            if result.cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1
                estimated_prompt_tokens += client.estimate_tokens(prompt)
        rows.append(
            {
                "Dataset": dataset_name,
                "Req_ID": req_id,
                "Type": row.Type,
                "Split": row.Split,
                "PairGroupID": row.PairGroupID,
                "GoldLinkCount": row.GoldLinkCount,
                "DensityGroup": row.DensityGroup,
                "Code_File": code_id,
                "Label": row.Label,
                "PairRole": row.PairRole,
                "SampledWithReplacement": row.SampledWithReplacement,
                "Method": method,
                "Prediction": result.prediction.prediction,
                "Confidence": result.prediction.confidence,
                "Score": prediction_score(
                    result.prediction.prediction, result.prediction.confidence
                ),
                "ParseError": result.prediction.parse_error,
                "CacheHit": result.cache_hit,
                "ApiCall": not result.cache_hit and not estimate_only,
            }
        )

    predictions = pd.DataFrame(rows)
    input_price, output_price = pricing_for(client.model, config)
    estimated_cost = (
        estimated_prompt_tokens * input_price + cache_misses * 300 * output_price
    ) / 1_000_000
    summary = {
        "dataset": dataset_name,
        "method": method,
        "total_pairs": len(predictions),
        "unique_pairs": len(pair_results),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "estimated_new_prompt_tokens": estimated_prompt_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
    }
    return predictions, summary


def proposed_pair_classifier_candidates(
    dataset_name: str, dataset: Any, manifest: pd.DataFrame, config: dict[str, Any]
) -> dict[str, dict[str, tuple[int, float]]]:
    """Per requirement, every DISTINCT manifest code file (deduped, since
    with-replacement negatives can repeat a code_id across PairGroupIDs),
    with a placeholder (rank, score) -- exactly Direct Prompting's shape,
    reused here because retrieval rank/score never reaches the
    alignment/verification prompts either (build_code_profile only reads
    candidate.summary/selected_context, never .rank/.retrieval_score)."""
    return build_candidates_for_manifest("Direct Prompting", dataset_name, dataset, manifest, config)


# Must match DependencyContext*PromptBuilder's own default so the
# neighbor set we fetch/estimate summaries for is exactly the set that
# actually gets rendered -- never more (wasted cost), never less
# (under-priced preflight).
DEPENDENCY_CONTEXT_MAX_SHOWN = 3


def manifest_candidate_code_ids(
    candidates: dict[str, dict[str, tuple[int, float]]],
) -> set[str]:
    """Every distinct code_id appearing anywhere in this split's manifest
    candidates -- the scope for both coverage reporting and dependency-
    neighbor summarization (never the whole dataset corpus)."""
    return {code_id for per_req in candidates.values() for code_id in per_req}


def code_summaries_for_dependency_context(
    dataset: Any,
    code_ids: set[str],
    client: Any,
    cache: LLMCache,
    config: dict[str, Any],
    *,
    live_api: bool,
    llm_service: FrameworkLLMService | None = None,
) -> dict[str, Any]:
    """code_id -> CodeSummary, scoped to EXACTLY code_ids (the dependency
    neighbors this run's manifest candidates will actually display -- see
    dependency_neighbors_to_summarize), never the whole dataset corpus.

    live_api=False (the preflight/estimate path) never attempts a real
    call: FrameworkLLMService's own live_api gate only avoids new API
    calls, it does not stop it writing an "estimated" placeholder to the
    real, shared cache on a miss (see LLMBaseline.run_pair's docstring in
    src/baselines/llm.py for the exact same class of bug already found and
    fixed there) -- rather than risk that here too, the preflight path
    returns an empty map (dependency descriptions render as "" in the
    schema used for token-length estimation; the real cost of these scoped
    summaries is separately and accurately accounted for by
    estimate_dependency_summary_cost, a real-cache-only check, so this
    does not silently under-price the preflight). live_api=True (the
    real, already-confirmed run) fetches every summary in scope for real.

    `llm_service`, when given, is reused as-is (not rebuilt) so these
    dependency-summary calls share the SAME max_api_calls budget cap,
    real_api_calls counter, and token-statistics records as whatever
    other calls (alignment/verification/understanding) that caller is
    making with it -- a fresh, disconnected FrameworkLLMService here would
    silently let this category of real spend bypass the caller's budget
    cap and cost accounting. Omitting it (the default) preserves the
    original self-contained behavior for callers that don't share a
    service (e.g. a preflight-only or standalone invocation)."""
    if not live_api or not code_ids:
        return {}
    if llm_service is None:
        input_price, output_price = pricing_for(client.model, config)
        llm_service = FrameworkLLMService(
            client=client, cache=cache, dataset=dataset.name, live_api=True,
            input_price=input_price, output_price=output_price,
        )
    summarizer = CachedCodeSummarizer(llm_service=llm_service)
    code_corpus = {
        code_id: dataset.code_files[code_id].text
        for code_id in code_ids
        if code_id in dataset.code_files
    }
    return summarizer.summarize_corpus(code_corpus)


def estimate_dependency_summary_cost(
    dataset: Any,
    dataset_name: str,
    code_ids: set[str],
    client: Any,
    cache: LLMCache,
    input_price: float,
    output_price: float,
) -> dict[str, Any]:
    """Real (non-estimated-cache) preflight for summarizing exactly the
    scoped dependency-neighbor code_ids v3's prompts will display --
    estimate_framework_budget() only knows about the manifest's own
    candidates' summarization cost, not these extra neighbor files, so
    this must be computed and added separately or the v3 preflight would
    silently under-count real cost. Mirrors CachedCodeSummarizer's own
    cache identity exactly (see src/retrieval/summarizer.py:
    SUMMARY_CACHE_MODULE/SUMMARY_CACHE_PROMPT/SUMMARY_CACHE_REQUIREMENT_ID),
    reading the real cache only (allow_estimated=False), never writing."""
    from src.llm.cache import CacheIdentity
    from src.llm.token_statistics import estimate_cost
    from src.retrieval.summarizer import (
        CodeSummaryPromptBuilder,
        SUMMARY_CACHE_MODULE,
        SUMMARY_CACHE_PROMPT,
        SUMMARY_CACHE_REQUIREMENT_ID,
    )

    prompt_builder = CodeSummaryPromptBuilder()
    hits = 0
    misses = 0
    prompt_tokens = 0
    completion_tokens = 0
    for code_id in sorted(code_ids):
        identity = CacheIdentity(
            model=client.model,
            prompt=f"[{SUMMARY_CACHE_MODULE}]\n{SUMMARY_CACHE_PROMPT}",
            dataset=dataset_name,
            requirement_id=SUMMARY_CACHE_REQUIREMENT_ID,
            code_file=code_id,
            provider=client.provider,
        )
        if cache.get(identity, allow_estimated=False) is not None:
            hits += 1
            continue
        misses += 1
        if code_id in dataset.code_files:
            prompt = prompt_builder.build_summary_prompt(code_id, dataset.code_files[code_id].text)
            prompt_tokens += client.estimate_tokens(prompt)
            completion_tokens += client.max_completion_tokens
    return {
        "dependency_summary_code_ids_in_scope": len(code_ids),
        "dependency_summary_cache_hits": hits,
        "dependency_summary_cache_misses": misses,
        "dependency_summary_estimated_prompt_tokens": prompt_tokens,
        "dependency_summary_estimated_cost_usd": round(
            estimate_cost(prompt_tokens, completion_tokens, input_price, output_price), 6
        ),
    }


def prompt_builders_for_proposed_pair_classifier(
    method: str,
    dependency_graph: dict[str, set[str]] | None = None,
    code_summaries: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """(alignment_prompt_builder, verification_prompt_builder), both None
    for v1 (the frozen AlignmentPromptBuilder/VerificationPromptBuilder
    default), the isolated evidence-anchored v2 builders for
    PROPOSED_PAIR_CLASSIFIER_V2_LABEL, or the v3 (v2 + dependency_context)
    builders for PROPOSED_PAIR_CLASSIFIER_V3_LABEL. Never touches v1/v2's
    prompt builders -- each variant is an entirely separate pair of
    classes."""
    if method == PROPOSED_PAIR_CLASSIFIER_V2_LABEL:
        return EvidenceAnchoredAlignmentPromptBuilder(), EvidenceAnchoredVerificationPromptBuilder()
    if method == PROPOSED_PAIR_CLASSIFIER_V3_LABEL:
        return (
            DependencyContextAlignmentPromptBuilder(dependency_graph or {}, code_summaries or {}),
            DependencyContextVerificationPromptBuilder(dependency_graph or {}, code_summaries or {}),
        )
    return None, None


def estimate_proposed_pair_classifier_cost(
    dataset_name: str,
    manifest: pd.DataFrame,
    client: Any,
    cache: LLMCache,
    config: dict[str, Any],
    method: str = PROPOSED_PAIR_CLASSIFIER_LABEL,
) -> dict[str, Any]:
    """Real preflight for the DCAR-bypassed pair classifier (NOT the
    end-to-end Proposed Framework -- see PROPOSED_PAIR_CLASSIFIER_LABEL
    docstring above), via the exact same estimate_framework_budget() the
    real top-k Proposed Framework experiments use for their own preflight
    -- it checks the real (non-estimated) cache per prompt category
    (understanding, code summarization, alignment, verification) rather
    than a rough call-count upper bound. Setting both candidate_pool_k and
    llm_top_k to DCAR_BYPASS_MAX_K forces dcar_base_k_for(...) to also
    return DCAR_BYPASS_MAX_K, so the retriever constructed inside this
    estimate has base_k == max_k == DCAR_BYPASS_MAX_K -- retrieval_budget_for
    then always selects DCAR_BYPASS_MAX_K candidates
    (min(max_k, max(base_k, multiplier*base_k)) == max_k whenever
    base_k == max_k), which exceeds every dataset's code-file count, so
    SemanticFilter's rank_fallback never truncates below the full manifest
    candidate set for this requirement -- see filters.py's _rank_fallback,
    which backfills from the complete candidate list until `limit` is
    reached and therefore returns everything when limit exceeds the
    candidate count."""
    loader = DatasetLoader(PROJECT_ROOT / "configs" / "datasets.yaml")
    dataset = loader.load(dataset_name)
    candidates = proposed_pair_classifier_candidates(dataset_name, dataset, manifest, config)
    requirement_ids = sorted(candidates)
    input_price, output_price = pricing_for(client.model, config)

    is_v3 = method == PROPOSED_PAIR_CLASSIFIER_V3_LABEL
    dependency_graph = None
    neighbor_ids_to_summarize: set[str] = set()
    dependency_summary_estimate = None
    if is_v3:
        dependency_graph = build_dependency_graph(
            {code_id: a.text for code_id, a in dataset.code_files.items()}
        )
        candidate_code_ids = manifest_candidate_code_ids(candidates)
        neighbor_ids_to_summarize = dependency_neighbors_to_summarize(
            dependency_graph, candidate_code_ids, DEPENDENCY_CONTEXT_MAX_SHOWN
        )
        dependency_summary_estimate = estimate_dependency_summary_cost(
            dataset, dataset_name, neighbor_ids_to_summarize, client, cache,
            input_price, output_price,
        )
    code_summaries = code_summaries_for_dependency_context(
        dataset, neighbor_ids_to_summarize, client, cache, config, live_api=False
    ) if is_v3 else None
    alignment_prompt_builder, verification_prompt_builder = prompt_builders_for_proposed_pair_classifier(
        method, dependency_graph, code_summaries
    )
    budget = estimate_framework_budget(
        dataset=dataset,
        requirement_ids=requirement_ids,
        candidates=candidates,
        resume_pairs=set(),
        client=client,
        cache=cache,
        input_price=input_price,
        output_price=output_price,
        retriever_name="dcar_bypassed",
        llm_top_k=DCAR_BYPASS_MAX_K,
        candidate_pool_k=DCAR_BYPASS_MAX_K,
        dcar_selection_strategy="rank_fallback",
        alignment_prompt_builder=alignment_prompt_builder,
        verification_prompt_builder=verification_prompt_builder,
    )
    total_cost = budget["estimated_cost_usd"]
    total_tokens = budget["estimated_prompt_tokens"]
    if dependency_summary_estimate is not None:
        total_cost += dependency_summary_estimate["dependency_summary_estimated_cost_usd"]
        total_tokens += dependency_summary_estimate["dependency_summary_estimated_prompt_tokens"]
    total_cache_hits = budget["cache_hits"]
    total_cache_misses = budget["expected_api_calls"]
    if dependency_summary_estimate is not None:
        # Must be folded in here, not just reported separately -- main()'s
        # --live-api confirmation gate checks estimate["cache_misses"] > 0;
        # missing these would silently skip the confirmation prompt whenever
        # the manifest's own categories happened to be fully cached but
        # dependency-neighbor summaries were not.
        total_cache_hits += dependency_summary_estimate["dependency_summary_cache_hits"]
        total_cache_misses += dependency_summary_estimate["dependency_summary_cache_misses"]
    result = {
        "dataset": dataset_name,
        "method": method,
        "requirements": budget["selected_requirements"],
        "total_pairs": budget["candidate_pool_pairs"],
        "cache_hits": total_cache_hits,
        "cache_misses": total_cache_misses,
        "estimated_new_prompt_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 6),
        "call_breakdown_by_category": budget["call_counts"],
        "cache_hits_by_category": {
            name: stats["cache_hits"] for name, stats in budget["category_stats"].items()
        },
        "cache_misses_by_category": {
            name: stats["cache_misses"] for name, stats in budget["category_stats"].items()
        },
    }
    if dependency_graph is not None:
        candidate_code_ids = manifest_candidate_code_ids(candidates)
        result["manifest_dependency_context_coverage_rate"] = round(
            manifest_dependency_coverage_rate(dependency_graph, candidate_code_ids), 4
        )
        result["full_dataset_dependency_graph_coverage_rate"] = round(
            dependency_coverage_rate(dependency_graph), 4
        )
        result.update(dependency_summary_estimate)
    return result


def rows_from_manifest_and_framework_predictions(
    dataset_name: str,
    manifest: pd.DataFrame,
    framework_predictions: pd.DataFrame,
    method: str = PROPOSED_PAIR_CLASSIFIER_LABEL,
) -> pd.DataFrame:
    """Build the pair-classification predictions.csv by iterating the
    ORIGINAL manifest row by row (preserving every row, including
    with-replacement repeats, with its own native PairGroupID/Label/etc.
    read directly off that row) and looking up that (Req_ID, Code_File)'s
    FrameworkPrediction from framework_predictions -- which has exactly one
    row per distinct (Req_ID, Code_File) candidate (never a duplicate key),
    since it comes from run_framework()'s own per-unique-candidate output.
    A naive manifest-keyed re-join (manifest.set_index([...]).loc[...])
    would be unsafe here for the same reason it was for run_llm_method:
    the manifest itself is not guaranteed unique on (Req_ID, Code_File)."""
    framework_lookup = {
        (row.Req_ID, row.Code_File): row
        for row in framework_predictions.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        key = (row.Req_ID, row.Code_File)
        framework_row = framework_lookup.get(key)
        if framework_row is None:
            raise SystemExit(
                f"{dataset_name}/{method}: missing "
                f"framework prediction for manifest pair {key} (should be "
                "impossible -- every manifest pair is included in the "
                "DCAR-bypassed candidate set)."
            )
        rows.append(
            {
                "Dataset": dataset_name,
                "Req_ID": row.Req_ID,
                "Type": row.Type,
                "Split": row.Split,
                "PairGroupID": row.PairGroupID,
                "GoldLinkCount": row.GoldLinkCount,
                "DensityGroup": row.DensityGroup,
                "Code_File": row.Code_File,
                "Label": row.Label,
                "PairRole": row.PairRole,
                "SampledWithReplacement": row.SampledWithReplacement,
                "Method": method,
                "Prediction": framework_row.Prediction,
                "Confidence": framework_row.Final_Score,
                "Score": framework_row.Final_Score,
            }
        )
    return pd.DataFrame(rows)


def run_proposed_pair_classifier_live(
    dataset_name: str,
    manifest: pd.DataFrame,
    client: Any,
    cache: LLMCache,
    config: dict[str, Any],
    method: str = PROPOSED_PAIR_CLASSIFIER_LABEL,
    *,
    max_api_calls: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Real live run of the DCAR-bypassed BAN+SRCV pair classifier, via the
    exact same run_framework() the real top-k Proposed Framework
    experiments use (see estimate_proposed_pair_classifier_cost's
    docstring for why candidate_pool_k=llm_top_k=DCAR_BYPASS_MAX_K makes
    DCAR select every manifest candidate rather than narrowing to a
    budget). Output rows are built by iterating the ORIGINAL manifest
    (preserving every row, including with-replacement repeats, with its
    own native metadata) and looking up that (Req_ID, Code_File)'s
    FrameworkPrediction from run_framework()'s result -- run_framework()'s
    own output has one row per distinct (Req_ID, Code_File) candidate, so
    that lookup is safe even though the manifest itself is not guaranteed
    unique on that key (see run_llm_method's docstring for the same
    pattern and why a naive manifest-keyed re-join is unsafe instead).

    run_framework() itself always computes Final_Score via the pipeline's
    own hardcoded formula (FrameworkConfig.verification_lambda=0.5,
    equivalent to scoring variant A) -- exactly like the real top-k
    formal runner's main(), this must be immediately overwritten by
    apply_scoring_variant() using the FROZEN configs/final_framework_config.yaml
    scoring.selected_variant (currently "B") before the score is used for
    anything, or every downstream number silently reflects the wrong,
    unfrozen variant. The variant is read from the frozen config every
    call -- never hardcoded -- so a future re-freeze is picked up
    automatically."""
    loader = DatasetLoader(PROJECT_ROOT / "configs" / "datasets.yaml")
    dataset = loader.load(dataset_name)
    candidates = proposed_pair_classifier_candidates(dataset_name, dataset, manifest, config)
    requirement_ids = sorted(candidates)
    input_price, output_price = pricing_for(client.model, config)
    llm_service = FrameworkLLMService(
        client=client,
        cache=cache,
        dataset=dataset_name,
        live_api=True,
        input_price=input_price,
        output_price=output_price,
        max_api_calls=max_api_calls,
    )
    is_v3 = method == PROPOSED_PAIR_CLASSIFIER_V3_LABEL
    dependency_graph = None
    neighbor_ids_to_summarize: set[str] = set()
    if is_v3:
        dependency_graph = build_dependency_graph(
            {code_id: a.text for code_id, a in dataset.code_files.items()}
        )
        candidate_code_ids = manifest_candidate_code_ids(candidates)
        neighbor_ids_to_summarize = dependency_neighbors_to_summarize(
            dependency_graph, candidate_code_ids, DEPENDENCY_CONTEXT_MAX_SHOWN
        )
    code_summaries = code_summaries_for_dependency_context(
        dataset, neighbor_ids_to_summarize, client, cache, config, live_api=True,
        llm_service=llm_service,
    ) if is_v3 else None
    alignment_prompt_builder, verification_prompt_builder = prompt_builders_for_proposed_pair_classifier(
        method, dependency_graph, code_summaries
    )
    custom_aligner = (
        BidirectionalSemanticAligner(llm_service=llm_service, prompt_builder=alignment_prompt_builder)
        if alignment_prompt_builder is not None else None
    )
    custom_verifier = (
        SelfReflectiveVerifier(llm_service=llm_service, prompt_builder=verification_prompt_builder)
        if verification_prompt_builder is not None else None
    )
    framework_predictions, _token_records, diagnostics = run_framework(
        dataset=dataset,
        requirement_ids=requirement_ids,
        candidates=candidates,
        provider=client.provider,
        model=client.model,
        retriever="dcar_bypassed",
        llm_top_k=DCAR_BYPASS_MAX_K,
        candidate_pool_k=DCAR_BYPASS_MAX_K,
        threshold=0.2,
        resume_pairs=set(),
        llm_service=llm_service,
        dcar_selection_strategy="rank_fallback",
        aligner=custom_aligner,
        verifier=custom_verifier,
    )
    frozen_config = load_final_framework_config(PROJECT_ROOT)
    scoring_variant = str(frozen_config["scoring"]["selected_variant"])
    scoring_threshold = float(frozen_config["scoring"]["threshold"])
    framework_predictions = apply_scoring_variant(
        framework_predictions, scoring_variant, scoring_threshold
    )
    predictions = rows_from_manifest_and_framework_predictions(
        dataset_name, manifest, framework_predictions, method
    )
    summary = {
        "dataset": dataset_name,
        "method": method,
        "scoring_variant": scoring_variant,
        "total_pairs": len(predictions),
        "unique_pairs": len(framework_predictions),
        "requirements": len(requirement_ids),
        "framework_cache_hits": diagnostics.get("framework_cache_hits"),
        "framework_cache_misses": diagnostics.get("framework_cache_misses"),
        "parse_errors": diagnostics.get("parse_errors"),
        "empty_responses": diagnostics.get("empty_responses"),
        "real_api_calls": llm_service.real_api_calls,
        "max_api_calls": max_api_calls,
    }
    if dependency_graph is not None:
        candidate_code_ids = manifest_candidate_code_ids(candidates)
        summary["manifest_dependency_context_coverage_rate"] = round(
            manifest_dependency_coverage_rate(dependency_graph, candidate_code_ids), 4
        )
        summary["full_dataset_dependency_graph_coverage_rate"] = round(
            dependency_coverage_rate(dependency_graph), 4
        )
    return predictions, summary


@dataclass(frozen=True)
class AblationSpec:
    """One row of Table 6's ablation study. Each factory receives exactly
    what it needs to build that ablation's version of the one component
    being isolated; every other stage of the DCAR-bypassed BAN+SRCV pair
    classifier stays byte-identical to Full v3 (evidence_anchored +
    dependency_context)."""

    label: str
    # Whether this ablation needs the v3 dependency graph/neighbor code
    # summaries at all (False only for w/o Dependency Context).
    uses_dependency_context: bool
    # (dependency_graph, code_summaries) -> alignment prompt builder.
    alignment_prompt_builder_factory: Callable[[dict[str, set[str]], dict[str, Any]], Any]
    # (dependency_graph, code_summaries) -> verification prompt builder, or
    # None if verification is never even built (w/o Verification only).
    verification_prompt_builder_factory: (
        Callable[[dict[str, set[str]], dict[str, Any]], Any] | None
    )
    # llm_service -> understanding object, or None for the frozen
    # TypeAwareRequirementUnderstanding default (every ablation except w/o
    # Type Understanding).
    understanding_factory: Callable[[FrameworkLLMService | None], Any] | None
    # True only for w/o Verification: no verification call is made, live or
    # cached, and the final score is the alignment score directly.
    skip_verification: bool
    # (llm_service, alignment_prompt_builder) -> aligner override, or None
    # for the frozen BidirectionalSemanticAligner(prompt_builder=...)
    # default (every ablation except w/o Bi-Alignment).
    aligner_factory: Callable[[FrameworkLLMService | None, Any], Any] | None


ABLATIONS: dict[str, AblationSpec] = {
    ABLATION_WO_TYPE_LABEL: AblationSpec(
        label=ABLATION_WO_TYPE_LABEL,
        uses_dependency_context=True,
        # Type-neutral builders (NOT v3's DependencyContext* ones): never
        # print "Requirement type: ..." or interpolate the profile's repr
        # (which would include requirement_type) -- see
        # src/framework/generic_understanding.py for why this ablation can
        # no longer reuse v3's alignment/verification prompt templates
        # unchanged (they are what leaked the true FR/NFR/Mixed label).
        alignment_prompt_builder_factory=(
            lambda dg, cs: TypeNeutralAlignmentPromptBuilder(dg, cs)
        ),
        verification_prompt_builder_factory=(
            lambda dg, cs: TypeNeutralVerificationPromptBuilder(dg, cs)
        ),
        understanding_factory=(
            lambda llm_service: GenericRequirementUnderstanding(llm_service=llm_service)
        ),
        skip_verification=False,
        aligner_factory=None,
    ),
    ABLATION_WO_DEPENDENCY_LABEL: AblationSpec(
        label=ABLATION_WO_DEPENDENCY_LABEL,
        uses_dependency_context=False,
        alignment_prompt_builder_factory=(
            lambda dg, cs: EvidenceAnchoredAlignmentPromptBuilder()
        ),
        verification_prompt_builder_factory=(
            lambda dg, cs: EvidenceAnchoredVerificationPromptBuilder()
        ),
        understanding_factory=None,
        skip_verification=False,
        aligner_factory=None,
    ),
    ABLATION_WO_BI_ALIGNMENT_LABEL: AblationSpec(
        label=ABLATION_WO_BI_ALIGNMENT_LABEL,
        uses_dependency_context=True,
        alignment_prompt_builder_factory=(
            lambda dg, cs: UnidirectionalAlignmentPromptBuilder(dg, cs)
        ),
        verification_prompt_builder_factory=(
            lambda dg, cs: DependencyContextVerificationPromptBuilder(dg, cs)
        ),
        understanding_factory=None,
        skip_verification=False,
        aligner_factory=(
            lambda llm_service, prompt_builder: RequirementToCodeOnlyAligner(
                llm_service=llm_service, prompt_builder=prompt_builder,
            )
        ),
    ),
    ABLATION_WO_VERIFICATION_LABEL: AblationSpec(
        label=ABLATION_WO_VERIFICATION_LABEL,
        uses_dependency_context=True,
        alignment_prompt_builder_factory=(
            lambda dg, cs: DependencyContextAlignmentPromptBuilder(dg, cs)
        ),
        verification_prompt_builder_factory=None,
        understanding_factory=None,
        skip_verification=True,
        aligner_factory=None,
    ),
}


def _wo_verification_cache_only_setup(
    dataset_name: str,
    manifest: pd.DataFrame,
    client: Any,
    cache: LLMCache,
    config: dict[str, Any],
) -> tuple[Any, dict[str, dict[str, tuple[int, float]]], FrameworkLLMService, Any, Any, Any]:
    """Shared construction for w/o Verification's cache-only path (used by
    both the hard-fail replay and the non-crashing audit below): the
    dataset, candidates, a real-cache-only llm_service (live_api=True,
    max_api_calls=0 -- see module-level FrameworkLLMService docs), a
    retriever wired to a real-cache-only summarizer, and Full v3's own
    DependencyContextAlignmentPromptBuilder (byte-identical builder, so a
    real v3 cache entry is reused whenever the reconstructed prompt
    matches). Building the dependency-context code summaries themselves
    is NOT wrapped here -- a miss there must surface to each caller
    exactly like any other cache gap (hard-fail for the replay, a caught
    per-requirement miss for the audit)."""
    loader = DatasetLoader(PROJECT_ROOT / "configs" / "datasets.yaml")
    dataset = loader.load(dataset_name)
    candidates = proposed_pair_classifier_candidates(dataset_name, dataset, manifest, config)
    cache_only_service = FrameworkLLMService(
        client=client, cache=cache, dataset=dataset_name, live_api=True, max_api_calls=0,
    )
    code_corpus = {code_id: a.text for code_id, a in dataset.code_files.items()}
    dependency_graph = build_dependency_graph(code_corpus)
    candidate_code_ids = manifest_candidate_code_ids(candidates)
    neighbor_ids = dependency_neighbors_to_summarize(
        dependency_graph, candidate_code_ids, DEPENDENCY_CONTEXT_MAX_SHOWN
    )
    code_summaries = code_summaries_for_dependency_context(
        dataset, neighbor_ids, client, cache, config,
        live_api=True, llm_service=cache_only_service,
    )
    alignment_prompt_builder = DependencyContextAlignmentPromptBuilder(
        dependency_graph, code_summaries
    )
    retriever = DynamicContextAdaptiveRetriever(
        summarizer=CachedCodeSummarizer(llm_service=cache_only_service),
        base_k=DCAR_BYPASS_MAX_K, max_k=DCAR_BYPASS_MAX_K,
        selection_strategy="rank_fallback",
    )
    aligner = BidirectionalSemanticAligner(
        llm_service=cache_only_service, prompt_builder=alignment_prompt_builder,
    )
    understanding = TypeAwareRequirementUnderstanding(llm_service=cache_only_service)
    return dataset, candidates, cache_only_service, retriever, aligner, understanding


def replay_wo_verification_alignment_from_cache(
    dataset_name: str,
    manifest: pd.DataFrame,
    client: Any,
    cache: LLMCache,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """w/o Verification is a pure CACHE-ONLY REPLAY of Full v3's real
    alignment results -- it must never reconstruct-and-call alignment
    fresh, and never fall back to a locally-computed score. This reuses
    BidirectionalSemanticAligner.align_requirement_batch completely
    unmodified (the exact production parser/validator every live v1/v2/v3
    run already uses) with a real-cache-only llm_service (live_api=True,
    so cache.get() only accepts a genuine non-estimated response, exactly
    like a real live run would see; max_api_calls=0, so
    FrameworkLLMService.complete_json raises RunBudgetExceededError
    BEFORE any real network call the instant a batch is missing --
    and, via align_requirement_batch's own retry-on-invalid loop, an
    INCOMPLETE cached payload surfaces the same way, since every retry
    attempt after the first also re-hits this same zero-budget check).

    Both RunBudgetExceededError and IncompleteBatchResponseError are left
    UNCAUGHT here deliberately: a missing or incomplete cached batch for
    ANY requirement is a hard failure for the WHOLE run -- no predictions
    are written, and there is no fallback path of any kind.

    Final_Score is set to Alignment_Score directly: there is no
    verification stage in this ablation at all, so no verification
    prompt is ever built and no verification cache entry is ever read."""
    dataset, candidates, cache_only_service, retriever, aligner, understanding = (
        _wo_verification_cache_only_setup(dataset_name, manifest, client, cache, config)
    )
    requirement_ids = sorted(candidates)
    code_corpus = {code_id: a.text for code_id, a in dataset.code_files.items()}

    rows: list[dict[str, Any]] = []
    for requirement_id in requirement_ids:
        requirement = dataset.requirements[requirement_id]
        requirement_profile = understanding.understand(
            requirement.text, requirement.req_type, requirement_id,
        )
        evidence_context = retriever.retrieve(
            requirement_profile=requirement_profile,
            code_corpus=code_corpus,
            ranked_candidates=candidates[requirement_id],
        )
        batch_key = batch_cache_key(
            "dcar_bypassed", DCAR_BYPASS_MAX_K, candidates[requirement_id],
        )
        alignment_evidence = aligner.align(
            requirement_profile=requirement_profile,
            evidence_context=evidence_context,
            batch_cache_key=batch_key,
        )
        for evidence in alignment_evidence:
            rows.append(
                {
                    "Req_ID": requirement_id,
                    "Code_File": evidence.code_id,
                    "Alignment_Score": evidence.alignment_score,
                    "Final_Score": evidence.alignment_score,
                    "Prediction": "Yes" if evidence.alignment_score >= 0.2 else "No",
                }
            )

    framework_predictions = pd.DataFrame(rows)
    predictions = rows_from_manifest_and_framework_predictions(
        dataset_name, manifest, framework_predictions, ABLATION_WO_VERIFICATION_LABEL,
    )
    summary = {
        "dataset": dataset_name,
        "method": ABLATION_WO_VERIFICATION_LABEL,
        "scoring_variant": "alignment_score_only (cache-only replay of Full v3 alignment)",
        "total_pairs": len(predictions),
        "unique_pairs": len(framework_predictions),
        "requirements": len(requirement_ids),
        "real_api_calls": cache_only_service.real_api_calls,
        "source": "cache-only replay of Full v3's real alignment cache; zero API calls possible",
    }
    return predictions, summary


def audit_wo_verification_cache_coverage(
    dataset_name: str,
    manifest: pd.DataFrame,
    client: Any,
    cache: LLMCache,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Non-hard-failing DIAGNOSTIC counterpart to
    replay_wo_verification_alignment_from_cache: attempts the exact same
    cache-only replay per requirement, but catches
    (RunBudgetExceededError, IncompleteBatchResponseError) per requirement
    to report coverage against Full v3's real cache, rather than crashing
    the whole audit on the first gap. Never calls the API (same
    max_api_calls=0 mechanism as the hard-fail replay). Use this for
    --split preflight reporting; use
    replay_wo_verification_alignment_from_cache (which never catches
    anything) for the real, predictions-writing run."""
    try:
        dataset, candidates, cache_only_service, retriever, aligner, understanding = (
            _wo_verification_cache_only_setup(dataset_name, manifest, client, cache, config)
        )
        dependency_context_covered = True
    except (RunBudgetExceededError, IncompleteBatchResponseError):
        # A dependency-context neighbor summary itself is missing --
        # nothing downstream can be checked meaningfully.
        loader = DatasetLoader(PROJECT_ROOT / "configs" / "datasets.yaml")
        dataset = loader.load(dataset_name)
        candidates = proposed_pair_classifier_candidates(dataset_name, dataset, manifest, config)
        total_pairs = sum(len(v) for v in candidates.values())
        return {
            "dataset": dataset_name,
            "method": ABLATION_WO_VERIFICATION_LABEL,
            "requirements": len(candidates),
            "requirements_fully_covered_by_v3_cache": 0,
            "requirements_missing_or_incomplete": sorted(candidates),
            "unique_pairs_covered": 0,
            "unique_pairs_total": total_pairs,
            "dependency_summary_fully_covered": False,
            "estimated_cost_usd": 0.0,
            "cache_hits": 0,
            "cache_misses": total_pairs,
        }
    requirement_ids = sorted(candidates)
    code_corpus = {code_id: a.text for code_id, a in dataset.code_files.items()}

    covered_requirements = 0
    missing_requirements: list[str] = []
    covered_pairs = 0
    for requirement_id in requirement_ids:
        requirement = dataset.requirements[requirement_id]
        try:
            requirement_profile = understanding.understand(
                requirement.text, requirement.req_type, requirement_id,
            )
            evidence_context = retriever.retrieve(
                requirement_profile=requirement_profile,
                code_corpus=code_corpus,
                ranked_candidates=candidates[requirement_id],
            )
            batch_key = batch_cache_key(
                "dcar_bypassed", DCAR_BYPASS_MAX_K, candidates[requirement_id],
            )
            alignment_evidence = aligner.align(
                requirement_profile=requirement_profile,
                evidence_context=evidence_context,
                batch_cache_key=batch_key,
            )
            covered_requirements += 1
            covered_pairs += len(alignment_evidence)
        except (RunBudgetExceededError, IncompleteBatchResponseError):
            missing_requirements.append(requirement_id)

    total_pairs = sum(len(v) for v in candidates.values())
    return {
        "dataset": dataset_name,
        "method": ABLATION_WO_VERIFICATION_LABEL,
        "requirements": len(requirement_ids),
        "requirements_fully_covered_by_v3_cache": covered_requirements,
        "requirements_missing_or_incomplete": missing_requirements,
        "unique_pairs_covered": covered_pairs,
        "unique_pairs_total": total_pairs,
        "dependency_summary_fully_covered": dependency_context_covered,
        "estimated_cost_usd": 0.0,
        "cache_hits": covered_pairs,
        "cache_misses": total_pairs - covered_pairs,
    }


def estimate_ablation_cost(
    dataset_name: str,
    manifest: pd.DataFrame,
    client: Any,
    cache: LLMCache,
    config: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    """Real preflight for one ablation of v3 -- same DCAR-bypass mechanism
    as estimate_proposed_pair_classifier_cost (checks the real,
    non-estimated cache per prompt category via estimate_framework_budget),
    driven by ABLATIONS[method] so each ablation's actual prompt/cache
    identity -- not v3's -- is what gets estimated.

    w/o Verification is special-cased to audit_wo_verification_cache_coverage
    instead: it is a cache-only replay of Full v3's alignment cache, never
    a reconstruct-and-call preflight (see that function and
    replay_wo_verification_alignment_from_cache's docstrings).

    `preflight_llm_service` (live_api=True, max_api_calls=0) lets
    understanding and the dependency-context code summaries below carry
    their REAL, previously-cached content when available (instead of an
    empty-payload/empty-dict approximation) -- max_api_calls=0 guarantees
    FrameworkLLMService.complete_json raises RunBudgetExceededError before
    any real network call whenever nothing is cached, so this can never
    trigger the API. Feeding the real content in here is what lets the
    alignment/verification categories' own cache checks (further inside
    estimate_framework_budget, which embeds these objects via repr into
    its prompts) correctly detect a hit against Full v3's real cache when
    this ablation's prompt is otherwise byte-identical to v3's."""
    if method == ABLATION_WO_VERIFICATION_LABEL:
        return audit_wo_verification_cache_coverage(
            dataset_name, manifest, client, cache, config
        )
    spec = ABLATIONS[method]
    loader = DatasetLoader(PROJECT_ROOT / "configs" / "datasets.yaml")
    dataset = loader.load(dataset_name)
    candidates = proposed_pair_classifier_candidates(dataset_name, dataset, manifest, config)
    requirement_ids = sorted(candidates)
    input_price, output_price = pricing_for(client.model, config)
    preflight_llm_service = FrameworkLLMService(
        client=client, cache=cache, dataset=dataset_name, live_api=True, max_api_calls=0,
    )

    dependency_graph: dict[str, set[str]] | None = None
    neighbor_ids_to_summarize: set[str] = set()
    dependency_summary_estimate = None
    code_summaries: dict[str, Any] = {}
    if spec.uses_dependency_context:
        dependency_graph = build_dependency_graph(
            {code_id: a.text for code_id, a in dataset.code_files.items()}
        )
        candidate_code_ids = manifest_candidate_code_ids(candidates)
        neighbor_ids_to_summarize = dependency_neighbors_to_summarize(
            dependency_graph, candidate_code_ids, DEPENDENCY_CONTEXT_MAX_SHOWN
        )
        dependency_summary_estimate = estimate_dependency_summary_cost(
            dataset, dataset_name, neighbor_ids_to_summarize, client, cache,
            input_price, output_price,
        )
        try:
            code_summaries = code_summaries_for_dependency_context(
                dataset, neighbor_ids_to_summarize, client, cache, config,
                live_api=True, llm_service=preflight_llm_service,
            )
        except RunBudgetExceededError:
            # Not every neighbor summary is in the real cache yet -- fall
            # back to today's empty-dict approximation rather than a
            # partial (and therefore misleading) summary set.
            code_summaries = {}

    alignment_prompt_builder = spec.alignment_prompt_builder_factory(
        dependency_graph or {}, code_summaries or {}
    )
    verification_prompt_builder = (
        spec.verification_prompt_builder_factory(dependency_graph or {}, code_summaries or {})
        if spec.verification_prompt_builder_factory is not None else None
    )
    understanding = (
        spec.understanding_factory(preflight_llm_service)
        if spec.understanding_factory is not None else None
    )
    understanding_cache_module = (
        getattr(understanding, "CACHE_MODULE", "understanding")
        if understanding is not None else "understanding"
    )

    budget = estimate_framework_budget(
        dataset=dataset,
        requirement_ids=requirement_ids,
        candidates=candidates,
        resume_pairs=set(),
        client=client,
        cache=cache,
        input_price=input_price,
        output_price=output_price,
        retriever_name="dcar_bypassed",
        llm_top_k=DCAR_BYPASS_MAX_K,
        candidate_pool_k=DCAR_BYPASS_MAX_K,
        dcar_selection_strategy="rank_fallback",
        alignment_prompt_builder=alignment_prompt_builder,
        verification_prompt_builder=verification_prompt_builder,
        understanding=understanding,
        understanding_cache_module=understanding_cache_module,
        skip_verification=spec.skip_verification,
    )
    total_cost = budget["estimated_cost_usd"]
    total_tokens = budget["estimated_prompt_tokens"]
    if dependency_summary_estimate is not None:
        total_cost += dependency_summary_estimate["dependency_summary_estimated_cost_usd"]
        total_tokens += dependency_summary_estimate["dependency_summary_estimated_prompt_tokens"]
    total_cache_hits = budget["cache_hits"]
    total_cache_misses = budget["expected_api_calls"]
    if dependency_summary_estimate is not None:
        total_cache_hits += dependency_summary_estimate["dependency_summary_cache_hits"]
        total_cache_misses += dependency_summary_estimate["dependency_summary_cache_misses"]
    result = {
        "dataset": dataset_name,
        "method": method,
        "requirements": budget["selected_requirements"],
        "total_pairs": budget["candidate_pool_pairs"],
        "cache_hits": total_cache_hits,
        "cache_misses": total_cache_misses,
        "estimated_new_prompt_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 6),
        "call_breakdown_by_category": budget["call_counts"],
        "cache_hits_by_category": {
            name: stats["cache_hits"] for name, stats in budget["category_stats"].items()
        },
        "cache_misses_by_category": {
            name: stats["cache_misses"] for name, stats in budget["category_stats"].items()
        },
    }
    if dependency_graph is not None:
        candidate_code_ids = manifest_candidate_code_ids(candidates)
        result["manifest_dependency_context_coverage_rate"] = round(
            manifest_dependency_coverage_rate(dependency_graph, candidate_code_ids), 4
        )
        result["full_dataset_dependency_graph_coverage_rate"] = round(
            dependency_coverage_rate(dependency_graph), 4
        )
        result.update(dependency_summary_estimate)
    return result


def run_ablation_live(
    dataset_name: str,
    manifest: pd.DataFrame,
    client: Any,
    cache: LLMCache,
    config: dict[str, Any],
    method: str,
    *,
    max_api_calls: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Real live run of one ablation of v3, via the exact same
    run_framework() the real top-k Proposed Framework experiments and v1/v2/v3
    use. Only the ONE component ABLATIONS[method] names differs from Full
    v3 -- see run_proposed_pair_classifier_live's docstring for why
    candidate_pool_k=llm_top_k=DCAR_BYPASS_MAX_K makes DCAR select every
    manifest candidate rather than narrowing to a budget.

    w/o Verification never reaches this function's own reconstruct-and-
    call path at all: it is dispatched straight to
    replay_wo_verification_alignment_from_cache, a pure cache-only replay
    of Full v3's real alignment cache that hard-fails (no fallback, no
    API call ever possible) on any missing or incomplete batch -- see
    that function's docstring."""
    if method == ABLATION_WO_VERIFICATION_LABEL:
        return replay_wo_verification_alignment_from_cache(
            dataset_name, manifest, client, cache, config
        )
    spec = ABLATIONS[method]
    loader = DatasetLoader(PROJECT_ROOT / "configs" / "datasets.yaml")
    dataset = loader.load(dataset_name)
    candidates = proposed_pair_classifier_candidates(dataset_name, dataset, manifest, config)
    requirement_ids = sorted(candidates)
    input_price, output_price = pricing_for(client.model, config)
    llm_service = FrameworkLLMService(
        client=client,
        cache=cache,
        dataset=dataset_name,
        live_api=True,
        input_price=input_price,
        output_price=output_price,
        max_api_calls=max_api_calls,
    )

    dependency_graph: dict[str, set[str]] | None = None
    neighbor_ids_to_summarize: set[str] = set()
    if spec.uses_dependency_context:
        dependency_graph = build_dependency_graph(
            {code_id: a.text for code_id, a in dataset.code_files.items()}
        )
        candidate_code_ids = manifest_candidate_code_ids(candidates)
        neighbor_ids_to_summarize = dependency_neighbors_to_summarize(
            dependency_graph, candidate_code_ids, DEPENDENCY_CONTEXT_MAX_SHOWN
        )
    code_summaries = code_summaries_for_dependency_context(
        dataset, neighbor_ids_to_summarize, client, cache, config, live_api=True,
        llm_service=llm_service,
    ) if spec.uses_dependency_context else {}

    alignment_prompt_builder = spec.alignment_prompt_builder_factory(
        dependency_graph or {}, code_summaries or {}
    )
    verification_prompt_builder = (
        spec.verification_prompt_builder_factory(dependency_graph or {}, code_summaries or {})
        if spec.verification_prompt_builder_factory is not None else None
    )
    understanding = (
        spec.understanding_factory(llm_service) if spec.understanding_factory is not None else None
    )
    custom_aligner = (
        spec.aligner_factory(llm_service, alignment_prompt_builder)
        if spec.aligner_factory is not None
        else BidirectionalSemanticAligner(
            llm_service=llm_service, prompt_builder=alignment_prompt_builder
        )
    )
    custom_verifier = (
        NullVerifier()
        if spec.skip_verification
        else SelfReflectiveVerifier(
            llm_service=llm_service, prompt_builder=verification_prompt_builder
        )
    )

    framework_predictions, _token_records, diagnostics = run_framework(
        dataset=dataset,
        requirement_ids=requirement_ids,
        candidates=candidates,
        provider=client.provider,
        model=client.model,
        retriever="dcar_bypassed",
        llm_top_k=DCAR_BYPASS_MAX_K,
        candidate_pool_k=DCAR_BYPASS_MAX_K,
        threshold=0.2,
        resume_pairs=set(),
        llm_service=llm_service,
        dcar_selection_strategy="rank_fallback",
        aligner=custom_aligner,
        verifier=custom_verifier,
        understanding=understanding,
    )

    if spec.skip_verification:
        # w/o Verification: the final score IS the alignment score, forced
        # explicitly here rather than relying solely on NullVerifier's
        # verification_score=1.0 collapsing the pipeline's own formula --
        # and the frozen scoring-variant B recombination (0.7*align +
        # 0.3*verify) is skipped entirely, since applying it would blend in
        # that neutral placeholder instead of leaving the score untouched.
        framework_predictions = framework_predictions.copy()
        framework_predictions["Final_Score"] = framework_predictions["Alignment_Score"]
        framework_predictions["Confidence"] = framework_predictions["Final_Score"]
        scoring_variant = "alignment_score_only (w/o Verification ablation)"
    else:
        frozen_config = load_final_framework_config(PROJECT_ROOT)
        scoring_variant = str(frozen_config["scoring"]["selected_variant"])
        scoring_threshold = float(frozen_config["scoring"]["threshold"])
        framework_predictions = apply_scoring_variant(
            framework_predictions, scoring_variant, scoring_threshold
        )

    predictions = rows_from_manifest_and_framework_predictions(
        dataset_name, manifest, framework_predictions, method
    )
    summary = {
        "dataset": dataset_name,
        "method": method,
        "scoring_variant": scoring_variant,
        "total_pairs": len(predictions),
        "unique_pairs": len(framework_predictions),
        "requirements": len(requirement_ids),
        "framework_cache_hits": diagnostics.get("framework_cache_hits"),
        "framework_cache_misses": diagnostics.get("framework_cache_misses"),
        "parse_errors": diagnostics.get("parse_errors"),
        "empty_responses": diagnostics.get("empty_responses"),
        "real_api_calls": llm_service.real_api_calls,
        "max_api_calls": max_api_calls,
    }
    if dependency_graph is not None:
        candidate_code_ids = manifest_candidate_code_ids(candidates)
        summary["manifest_dependency_context_coverage_rate"] = round(
            manifest_dependency_coverage_rate(dependency_graph, candidate_code_ids), 4
        )
        summary["full_dataset_dependency_graph_coverage_rate"] = round(
            dependency_coverage_rate(dependency_graph), 4
        )
    return predictions, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=RUNNABLE_DATASETS)
    parser.add_argument("--method", required=True, choices=ALL_METHODS)
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "validation", "heldout"),
        help="Restrict to manifest rows with this Split value. Use "
        "'validation' to score only the pairs needed for global threshold "
        "calibration, without touching the full (heldout+validation) "
        "manifest.",
    )
    parser.add_argument(
        "--live-api",
        action="store_true",
        help="Required to make any real API call for LLM-based methods. "
        "Omit to only estimate cost.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "datasets.yaml"),
    )
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=None,
        help="Hard cap on real (non-cached) API calls for this run. The "
        "call that would exceed the cap is refused before it happens "
        "(RunBudgetExceededError) and the run fails with no predictions "
        "written. Only applies to Proposed-PairClassifier methods "
        "(including its w/o-* ablations), which share FrameworkLLMService "
        "with the formal Proposed Framework runner.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    manifest = load_manifest(args.dataset)
    if args.split != "all":
        manifest = manifest[manifest["Split"] == args.split].reset_index(drop=True)
        if manifest.empty:
            raise SystemExit(
                f"No manifest rows with Split={args.split!r} for {args.dataset}."
            )

    if args.method in IR_NEURAL_METHODS:
        predictions = score_ir_neural(args.dataset, args.method, manifest, config)
        output_dir = ensure_directory(output_dir_for(args.dataset, args.method, args.split))
        predictions.to_csv(output_dir / "predictions.csv", index=False)
        print(
            f"{args.dataset}/{args.method} (split={args.split}): scored "
            f"{len(predictions)} pairs for free via existing raw_scores.csv "
            f"-> {output_dir / 'predictions.csv'}"
        )
        print(
            "Note: metrics.csv/density_group_metrics.csv require a frozen "
            "per-method threshold from "
            "scripts/calibrate_pair_classification_threshold.py "
            "(not yet run)."
        )
        return 0

    # LLM-based methods (Direct Prompting, Generic RAG-LLM,
    # Proposed-PairClassifier) share client/cache construction.
    model = args.model or "gpt-4.1"
    client = create_client(
        "openai", model, config, max_retries=8, rpm_limit=10,
        tpm_limit=20_000, min_call_interval=3.0,
    )
    cache = LLMCache(ensure_directory(PROJECT_ROOT / config["output"]["llm_cache_dir"]))
    output_dir = output_dir_for(args.dataset, args.method, args.split)

    if args.method in PROPOSED_PAIR_CLASSIFIER_LABELS:
        estimate = estimate_proposed_pair_classifier_cost(
            args.dataset, manifest, client, cache, config, args.method
        )
        print(f"{args.dataset}/{args.method} (split={args.split}) preflight:")
        for key, value in estimate.items():
            print(f"  {key}: {value}")
        print(f"  output_path: {output_dir / 'predictions.csv'}")

        if not args.live_api:
            return 0

        if estimate["cache_misses"] > 0:
            try:
                confirmed = input("Continue with real API calls? [y/N] ").strip() == "y"
            except EOFError:
                confirmed = False
            if not confirmed:
                print("Aborted: no API calls were made.")
                return 0

        predictions, live_summary = run_proposed_pair_classifier_live(
            args.dataset, manifest, client, cache, config, args.method,
            max_api_calls=args.max_api_calls,
        )
        ensure_directory(output_dir)
        predictions.to_csv(output_dir / "predictions.csv", index=False)
        print(f"{args.dataset}/{args.method} (split={args.split}) live run complete:")
        for key, value in live_summary.items():
            print(f"  {key}: {value}")
        print(f"  -> {output_dir / 'predictions.csv'}")
        return 0

    if args.method in ABLATION_LABELS:
        estimate = estimate_ablation_cost(
            args.dataset, manifest, client, cache, config, args.method
        )
        print(f"{args.dataset}/{args.method} (split={args.split}) preflight:")
        for key, value in estimate.items():
            print(f"  {key}: {value}")
        print(f"  output_path: {output_dir / 'predictions.csv'}")

        if not args.live_api:
            return 0

        if estimate["cache_misses"] > 0:
            try:
                confirmed = input("Continue with real API calls? [y/N] ").strip() == "y"
            except EOFError:
                confirmed = False
            if not confirmed:
                print("Aborted: no API calls were made.")
                return 0

        predictions, live_summary = run_ablation_live(
            args.dataset, manifest, client, cache, config, args.method,
            max_api_calls=args.max_api_calls,
        )
        ensure_directory(output_dir)
        predictions.to_csv(output_dir / "predictions.csv", index=False)
        print(f"{args.dataset}/{args.method} (split={args.split}) live run complete:")
        for key, value in live_summary.items():
            print(f"  {key}: {value}")
        print(f"  -> {output_dir / 'predictions.csv'}")
        return 0

    # Direct Prompting / Generic RAG-LLM
    # Preflight: always estimate first (no real calls), regardless of --live-api.
    _, estimate = run_llm_method(
        args.dataset, args.method, manifest, client, cache, config, estimate_only=True
    )
    print(f"{args.dataset}/{args.method} (split={args.split}) preflight:")
    for key, value in estimate.items():
        print(f"  {key}: {value}")
    print(f"  output_path: {output_dir / 'predictions.csv'}")

    if not args.live_api:
        return 0

    if estimate["cache_misses"] > 0:
        try:
            confirmed = input("Continue with real API calls? [y/N] ").strip() == "y"
        except EOFError:
            confirmed = False
        if not confirmed:
            print("Aborted: no API calls were made.")
            return 0

    predictions, live_summary = run_llm_method(
        args.dataset, args.method, manifest, client, cache, config, estimate_only=False
    )
    ensure_directory(output_dir)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    print(f"{args.dataset}/{args.method} (split={args.split}) live run complete:")
    for key, value in live_summary.items():
        print(f"  {key}: {value}")
    print(f"  -> {output_dir / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
