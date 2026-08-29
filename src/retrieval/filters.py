"""Type-aware semantic filtering for DCAR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.framework.evidence_cues import evidence_cue_text
from src.framework.understanding import (
    FRRequirementProfile,
    MixedRequirementProfile,
    NFRRequirementProfile,
    RequirementProfile,
)

from .summarizer import CodeSummary


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?")


@dataclass(frozen=True)
class FilteredCandidate:
    """Candidate retained after semantic filtering."""

    code_id: str
    rank: int
    retrieval_score: float
    relevance_score: float
    summary: CodeSummary
    strategy: str
    selected_context: str = ""
    functional_relevance_score: float | None = None
    quality_relevance_score: float | None = None


class FRSemanticFilter:
    """Filter candidates for functional requirements."""

    strategy = "fr"

    def score(
        self,
        requirement_profile: FRRequirementProfile,
        summary: CodeSummary,
        retrieval_score: float,
    ) -> float:
        query = " ".join(
            (
                requirement_profile.intent,
                requirement_profile.action,
                requirement_profile.object,
                requirement_profile.expected_behavior,
                " ".join(requirement_profile.entities),
            )
        )
        evidence = " ".join(
            (
                summary.name,
                summary.description,
                " ".join(summary.key_functions),
                " ".join(summary.key_classes),
            )
        )
        return _blend_score(retrieval_score, _overlap_score(query, evidence), 0.65)


class NFRSemanticFilter:
    """Filter candidates for non-functional requirements."""

    strategy = "nfr"

    def score(
        self,
        requirement_profile: NFRRequirementProfile,
        summary: CodeSummary,
        retrieval_score: float,
    ) -> float:
        query = " ".join(
            (
                requirement_profile.quality_attribute,
                requirement_profile.quality_metric,
                requirement_profile.scope,
                requirement_profile.constraint,
                requirement_profile.context,
                requirement_profile.evaluation_condition,
                evidence_cue_text(requirement_profile.evidence_cues),
            )
        )
        evidence = " ".join(
            (
                summary.description,
                " ".join(summary.api_calls),
                " ".join(summary.quality_related_elements),
                " ".join(summary.key_functions),
                " ".join(summary.key_classes),
            )
        )
        cue_score = _overlap_score(
            evidence_cue_text(requirement_profile.evidence_cues),
            evidence,
        )
        return _blend_score(
            retrieval_score,
            max(_overlap_score(query, evidence), cue_score),
            0.5,
        )


class MixedSemanticFilter:
    """Filter candidates for mixed requirements."""

    strategy = "mixed"

    def __init__(
        self,
        fr_filter: FRSemanticFilter | None = None,
        nfr_filter: NFRSemanticFilter | None = None,
    ):
        self.fr_filter = fr_filter or FRSemanticFilter()
        self.nfr_filter = nfr_filter or NFRSemanticFilter()

    def score(
        self,
        requirement_profile: MixedRequirementProfile,
        summary: CodeSummary,
        retrieval_score: float,
    ) -> float:
        fr_score, nfr_score = self.score_components(
            requirement_profile, summary, retrieval_score
        )
        return 0.5 * fr_score + 0.5 * nfr_score

    def score_components(
        self,
        requirement_profile: MixedRequirementProfile,
        summary: CodeSummary,
        retrieval_score: float,
    ) -> tuple[float, float]:
        fr_score = self.fr_filter.score(
            requirement_profile.functional_profile, summary, retrieval_score
        )
        nfr_score = self.nfr_filter.score(
            requirement_profile.quality_profile, summary, retrieval_score
        )
        return fr_score, nfr_score


class GenericSemanticFilter:
    """Type-neutral relevance score for a profile that carries no
    FR/NFR/Mixed-specific structure (e.g. the w/o Type Understanding
    ablation's GenericRequirementProfile). Reads only the generic
    `summary`/`key_terms`/`expected_behavior`/`raw_text` fields any such
    profile exposes -- never an FR-specific (intent/action/object) or
    NFR-specific (quality_attribute/evidence_cues) field, and not a copy
    of either's tuned blend weight."""

    strategy = "generic"

    def score(
        self,
        requirement_profile: Any,
        summary: CodeSummary,
        retrieval_score: float,
    ) -> float:
        query = " ".join(
            part
            for part in (
                getattr(requirement_profile, "summary", ""),
                getattr(requirement_profile, "expected_behavior", ""),
                " ".join(getattr(requirement_profile, "key_terms", ())),
            )
            if part
        ) or requirement_profile.raw_text
        evidence = " ".join(
            (
                summary.name,
                summary.description,
                " ".join(summary.key_functions),
                " ".join(summary.key_classes),
            )
        )
        return _blend_score(retrieval_score, _overlap_score(query, evidence), 0.5)


class SemanticFilter:
    """Dispatch semantic filtering according to requirement type."""

    def __init__(
        self,
        threshold: float = 0.5,
        selection_strategy: str = "rank_fallback",
        fr_filter: FRSemanticFilter | None = None,
        nfr_filter: NFRSemanticFilter | None = None,
        mixed_filter: MixedSemanticFilter | None = None,
        generic_filter: GenericSemanticFilter | None = None,
    ):
        if selection_strategy not in ("rank_fallback", "rank_window_diverse"):
            raise ValueError(
                "selection_strategy must be 'rank_fallback' or "
                "'rank_window_diverse'."
            )
        self.threshold = threshold
        self.selection_strategy = selection_strategy
        self.fr_filter = fr_filter or FRSemanticFilter()
        self.nfr_filter = nfr_filter or NFRSemanticFilter()
        self.mixed_filter = mixed_filter or MixedSemanticFilter(
            self.fr_filter, self.nfr_filter
        )
        self.generic_filter = generic_filter or GenericSemanticFilter()

    def score_relevance(
        self,
        requirement_profile: RequirementProfile,
        summary: CodeSummary,
        retrieval_score: float,
    ) -> tuple[float, str]:
        """Return type-aware relevance score and selected strategy.

        A profile that is none of FRRequirementProfile/NFRRequirementProfile/
        MixedRequirementProfile (e.g. GenericRequirementProfile) falls
        through to generic_filter -- a genuinely type-neutral scoring path,
        not a reuse of any of the three type-specific ones."""
        if isinstance(requirement_profile, FRRequirementProfile):
            return (
                self.fr_filter.score(
                    requirement_profile, summary, retrieval_score
                ),
                self.fr_filter.strategy,
            )
        if isinstance(requirement_profile, NFRRequirementProfile):
            return (
                self.nfr_filter.score(
                    requirement_profile, summary, retrieval_score
                ),
                self.nfr_filter.strategy,
            )
        if isinstance(requirement_profile, MixedRequirementProfile):
            return (
                self.mixed_filter.score(
                    requirement_profile, summary, retrieval_score
                ),
                self.mixed_filter.strategy,
            )
        return (
            self.generic_filter.score(requirement_profile, summary, retrieval_score),
            self.generic_filter.strategy,
        )

    def filter_candidates(
        self,
        requirement_profile: RequirementProfile,
        candidates: list[tuple[str, int, float, CodeSummary]],
        limit: int,
    ) -> list[FilteredCandidate]:
        """Retain the highest-scoring candidates under the type-aware strategy."""
        all_scored: list[FilteredCandidate] = []
        for code_id, rank, retrieval_score, summary in candidates:
            relevance, strategy = self.score_relevance(
                requirement_profile, summary, retrieval_score
            )
            functional_score, quality_score = self._component_scores(
                requirement_profile, summary, retrieval_score
            )
            all_scored.append(
                FilteredCandidate(
                    code_id=code_id,
                    rank=rank,
                    retrieval_score=retrieval_score,
                    relevance_score=relevance,
                    summary=summary,
                    strategy=strategy,
                    selected_context=_selected_context(summary),
                    functional_relevance_score=functional_score,
                    quality_relevance_score=quality_score,
                )
            )

        if self.selection_strategy == "rank_window_diverse":
            return self._rank_window_diverse(all_scored, limit)

        return self._rank_fallback(all_scored, limit)

    def _rank_fallback(
        self,
        all_scored: list[FilteredCandidate],
        limit: int,
    ) -> list[FilteredCandidate]:
        scored: list[FilteredCandidate] = []
        for candidate in all_scored:
            if candidate.relevance_score >= self.threshold:
                scored.append(candidate)

        scored.sort(key=lambda item: (-item.relevance_score, item.rank, item.code_id))
        if len(scored) >= limit:
            return scored[:limit]

        retained_ids = {item.code_id for item in scored}
        for candidate in all_scored:
            if candidate.code_id in retained_ids:
                continue
            scored.append(candidate)
            if len(scored) >= limit:
                break
        return scored[:limit]

    def _rank_window_diverse(
        self,
        all_scored: list[FilteredCandidate],
        limit: int,
    ) -> list[FilteredCandidate]:
        """Reserve a small validation-tested slot for deeper candidate windows."""
        if limit < 2 or len(all_scored) <= limit:
            return self._rank_fallback(all_scored, limit)

        baseline = self._rank_fallback(all_scored, limit)
        exploration_slots = min(2, max(1, limit // 5))
        core_slots = max(1, limit - exploration_slots)
        selected = baseline[:core_slots]
        selected_ids = {item.code_id for item in selected}
        deep_start_rank = max(limit + 1, 2 * limit + 1)
        deep_candidates = [
            item
            for item in all_scored
            if item.code_id not in selected_ids and item.rank >= deep_start_rank
        ]
        deep_candidates.sort(
            key=lambda item: (-item.relevance_score, item.rank, item.code_id)
        )

        for candidate in deep_candidates:
            if len(selected) >= limit:
                break
            selected.append(candidate)
            selected_ids.add(candidate.code_id)

        if len(selected) < limit:
            for candidate in baseline:
                if len(selected) >= limit:
                    break
                if candidate.code_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.code_id)

        selected.sort(key=lambda item: (-item.relevance_score, item.rank, item.code_id))
        return selected[:limit]

    def _component_scores(
        self,
        requirement_profile: RequirementProfile,
        summary: CodeSummary,
        retrieval_score: float,
    ) -> tuple[float | None, float | None]:
        if isinstance(requirement_profile, MixedRequirementProfile):
            return self.mixed_filter.score_components(
                requirement_profile, summary, retrieval_score
            )
        if isinstance(requirement_profile, FRRequirementProfile):
            return (
                self.fr_filter.score(requirement_profile, summary, retrieval_score),
                None,
            )
        if isinstance(requirement_profile, NFRRequirementProfile):
            return (
                None,
                self.nfr_filter.score(requirement_profile, summary, retrieval_score),
            )
        return None, None


SemanticRelevanceFilter = SemanticFilter


def _tokens(value: str) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(value)}


def _overlap_score(query: str, evidence: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    evidence_tokens = _tokens(evidence)
    return len(query_tokens & evidence_tokens) / len(query_tokens)


def _blend_score(
    retrieval_score: float,
    semantic_score: float,
    retrieval_weight: float,
) -> float:
    retrieval_score = max(0.0, min(1.0, retrieval_score))
    semantic_score = max(0.0, min(1.0, semantic_score))
    return (
        retrieval_weight * retrieval_score
        + (1.0 - retrieval_weight) * semantic_score
    )


def _selected_context(summary: CodeSummary) -> str:
    fields = [
        f"Name: {summary.name}",
        f"Description: {summary.description}",
    ]
    if summary.key_functions:
        fields.append("Key functions: " + ", ".join(summary.key_functions))
    if summary.key_classes:
        fields.append("Key classes: " + ", ".join(summary.key_classes))
    if summary.api_calls:
        fields.append("API calls: " + ", ".join(summary.api_calls[:10]))
    if summary.quality_related_elements:
        fields.append(
            "Quality elements: "
            + ", ".join(summary.quality_related_elements)
        )
    return "\n".join(fields)
