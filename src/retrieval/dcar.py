"""Dynamic Context-Adaptive Retrieval for the IST framework."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.framework.evidence_cues import EVIDENCE_CUE_TERMS, evidence_cue_text
from src.framework.understanding import (
    FRRequirementProfile,
    MixedRequirementProfile,
    NFRRequirementProfile,
    RequirementProfile,
)

from .filters import FilteredCandidate, SemanticFilter
from .summarizer import CachedCodeSummarizer, CodeSummary


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?")
_CONDITION_TERMS = {"if", "when", "unless", "must", "shall", "within", "only"}
_QUALITY_TERMS = {
    "security",
    "secure",
    "authentication",
    "authorization",
    "performance",
    "reliability",
    "availability",
    "maintainability",
    "scalability",
    "usability",
    "audit",
    "timeout",
    "error",
    "exception",
    *EVIDENCE_CUE_TERMS,
}
_ACTION_TERMS = {
    "create",
    "read",
    "update",
    "delete",
    "search",
    "return",
    "send",
    "receive",
    "validate",
    "display",
    "store",
    "calculate",
    "generate",
}


@dataclass(frozen=True)
class RequirementComplexitySignals:
    """Lightweight signals used to choose a retrieval budget."""

    length: int = 0
    action_density: float = 0.0
    condition_density: float = 0.0
    quality_density: float = 0.0
    mixed_indicator: float = 0.0
    complexity_score: float = 0.0


@dataclass(frozen=True)
class RetrievalBudget:
    """Type-aware retrieval budget for one requirement."""

    base_k: int
    selected_k: int
    max_k: int
    strategy: str
    complexity: RequirementComplexitySignals


@dataclass(frozen=True)
class DCAREvidenceContext:
    """Compact evidence context produced by DCAR."""

    requirement_id: str
    requirement_type: str
    query: str
    budget: RetrievalBudget
    candidates: tuple[FilteredCandidate, ...] = ()
    summaries: dict[str, CodeSummary] | None = None
    context_text: str = ""


class DynamicContextAdaptiveRetriever:
    """Coordinate cached summaries, type-aware budgeting, and filtering."""

    def __init__(
        self,
        summarizer: CachedCodeSummarizer | None = None,
        semantic_filter: SemanticFilter | None = None,
        prompt_builder: "DCARPromptBuilder | None" = None,
        base_k: int = 5,
        max_k: int = 10,
        selection_strategy: str = "rank_fallback",
    ):
        if base_k < 1:
            raise ValueError("base_k must be at least 1.")
        if max_k < base_k:
            raise ValueError("max_k must be greater than or equal to base_k.")
        self.summarizer = summarizer or CachedCodeSummarizer()
        self.semantic_filter = semantic_filter or SemanticFilter(
            selection_strategy=selection_strategy
        )
        self.prompt_builder = prompt_builder or DCARPromptBuilder()
        self.base_k = base_k
        self.max_k = max_k
        self.selection_strategy = selection_strategy

    def build_query(self, requirement_profile: RequirementProfile) -> str:
        """Construct a type-aware retrieval query from the requirement profile."""
        if isinstance(requirement_profile, FRRequirementProfile):
            return " ".join(
                item
                for item in (
                    requirement_profile.intent,
                    requirement_profile.action,
                    requirement_profile.object,
                    requirement_profile.expected_behavior,
                    " ".join(requirement_profile.entities),
                )
                if item
            ).strip()
        if isinstance(requirement_profile, NFRRequirementProfile):
            return " ".join(
                item
                for item in (
                    requirement_profile.quality_attribute,
                    requirement_profile.quality_metric,
                    requirement_profile.scope,
                    requirement_profile.constraint,
                    requirement_profile.context,
                    requirement_profile.evaluation_condition,
                    evidence_cue_text(requirement_profile.evidence_cues),
                )
                if item
            ).strip()
        if isinstance(requirement_profile, MixedRequirementProfile):
            return " ".join(
                item
                for item in (
                    self.build_query(requirement_profile.functional_profile),
                    self.build_query(requirement_profile.quality_profile),
                )
                if item
            ).strip()
        return requirement_profile.raw_text.strip()

    def analyze_complexity(
        self, requirement_profile: RequirementProfile
    ) -> RequirementComplexitySignals:
        """Compute local complexity signals used by DCAR budgeting."""
        tokens = _tokens(requirement_profile.raw_text)
        length = len(tokens)
        denominator = max(1, length)
        action_density = len(tokens & _ACTION_TERMS) / denominator
        condition_density = len(tokens & _CONDITION_TERMS) / denominator
        quality_density = len(tokens & _QUALITY_TERMS) / denominator
        mixed_indicator = (
            1.0 if isinstance(requirement_profile, MixedRequirementProfile) else 0.0
        )
        complexity = min(
            1.0,
            0.35 * min(1.0, length / 60.0)
            + 0.20 * min(1.0, action_density * 6.0)
            + 0.20 * min(1.0, condition_density * 6.0)
            + 0.15 * min(1.0, quality_density * 6.0)
            + 0.10 * mixed_indicator,
        )
        return RequirementComplexitySignals(
            length=length,
            action_density=action_density,
            condition_density=condition_density,
            quality_density=quality_density,
            mixed_indicator=mixed_indicator,
            complexity_score=complexity,
        )

    def retrieval_budget_for(
        self, requirement_profile: RequirementProfile
    ) -> RetrievalBudget:
        """Choose a type-aware retrieval budget."""
        complexity = self.analyze_complexity(requirement_profile)
        if isinstance(requirement_profile, FRRequirementProfile):
            multiplier = 2
            strategy = "fr-behavior-focused"
        elif isinstance(requirement_profile, NFRRequirementProfile):
            multiplier = 2
            strategy = "nfr-quality-evidence-focused"
        elif isinstance(requirement_profile, MixedRequirementProfile):
            multiplier = 2
            strategy = "mixed-functional-and-quality"
        else:
            multiplier = 1
            strategy = "generic"

        if complexity.complexity_score >= 0.7:
            multiplier += 1
        elif complexity.complexity_score < 0.35 and multiplier > 1:
            multiplier -= 1

        selected = min(self.max_k, max(self.base_k, multiplier * self.base_k))
        return RetrievalBudget(
            base_k=self.base_k,
            selected_k=selected,
            max_k=self.max_k,
            strategy=strategy,
            complexity=complexity,
        )

    def retrieve(
        self,
        requirement_profile: RequirementProfile,
        code_corpus: dict[str, str],
        ranked_candidates: dict[str, tuple[int, float]] | None = None,
    ) -> DCAREvidenceContext:
        """Return filtered candidate context for one requirement profile."""
        summary_corpus = self._summary_corpus_for(code_corpus, ranked_candidates)
        summaries = self.summarizer.summarize_corpus(summary_corpus)
        query = self.build_query(requirement_profile)
        budget = self.retrieval_budget_for(requirement_profile)
        ordered = self._candidate_order(query, summaries, ranked_candidates)
        candidates = self.semantic_filter.filter_candidates(
            requirement_profile,
            ordered,
            limit=budget.selected_k,
        )
        return DCAREvidenceContext(
            requirement_id=requirement_profile.requirement_id,
            requirement_type=requirement_profile.requirement_type,
            query=query,
            budget=budget,
            candidates=tuple(candidates),
            summaries=summaries,
            context_text=self._context_text(candidates),
        )

    @staticmethod
    def _summary_corpus_for(
        code_corpus: dict[str, str],
        ranked_candidates: dict[str, tuple[int, float]] | None,
    ) -> dict[str, str]:
        """Summarize only selected candidates when a retriever ranking is supplied."""
        if ranked_candidates is None:
            return code_corpus
        return {
            code_id: code_corpus[code_id]
            for code_id in ranked_candidates
            if code_id in code_corpus
        }

    def _candidate_order(
        self,
        query: str,
        summaries: dict[str, CodeSummary],
        ranked_candidates: dict[str, tuple[int, float]] | None,
    ) -> list[tuple[str, int, float, CodeSummary]]:
        if ranked_candidates is not None:
            ordered = [
                (code_id, rank_score[0], rank_score[1], summaries[code_id])
                for code_id, rank_score in ranked_candidates.items()
                if code_id in summaries
            ]
            return sorted(ordered, key=lambda item: (item[1], item[0]))

        scored = [
            (
                code_id,
                index,
                _overlap_score(query, _summary_text(summary)),
                summary,
            )
            for index, (code_id, summary) in enumerate(summaries.items(), start=1)
        ]
        return sorted(scored, key=lambda item: (-item[2], item[0]))

    @staticmethod
    def _context_text(candidates: list[FilteredCandidate]) -> str:
        return "\n\n".join(
            f"[{candidate.code_id}]\n{candidate.selected_context}"
            for candidate in candidates
        )


def _tokens(value: str) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(value)}


def _overlap_score(query: str, evidence: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.5
    evidence_tokens = _tokens(evidence)
    return len(query_tokens & evidence_tokens) / len(query_tokens)


def _summary_text(summary: CodeSummary) -> str:
    return " ".join(
        (
            summary.name,
            summary.description,
            " ".join(summary.key_functions),
            " ".join(summary.key_classes),
            " ".join(summary.api_calls),
            " ".join(summary.quality_related_elements),
        )
    )


class DCARPromptBuilder:
    """Build prompts for DCAR query construction and semantic filtering."""

    def build_query_prompt(self, requirement_profile: RequirementProfile) -> str:
        return f"""Build a type-aware retrieval query for DCAR.

Requirement type: {requirement_profile.requirement_type}
Requirement text:
{requirement_profile.raw_text}

Return only one JSON object:
{{"query": "compact query emphasizing type-specific evidence"}}"""

    def build_filter_prompt(
        self,
        requirement_profile: RequirementProfile,
        summaries: list[CodeSummary],
    ) -> str:
        summary_lines = "\n".join(
            f"- {summary.code_id}: {summary.description}" for summary in summaries
        )
        return f"""Filter retrieved code summaries for DCAR.

Requirement type: {requirement_profile.requirement_type}
Requirement text:
{requirement_profile.raw_text}

Candidate summaries:
{summary_lines}

Return only one JSON object:
{{"retained_code_ids": ["code ids that provide compact, type-aware evidence"]}}"""
