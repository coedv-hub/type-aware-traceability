"""Bidirectional semantic alignment for the IST framework."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.retrieval.dcar import DCAREvidenceContext
from src.retrieval.filters import FilteredCandidate
from src.retrieval.summarizer import CodeSummary

from .evidence_cues import evidence_cue_text
from .fusion import DEFAULT_MIXED_FUSION, FixedMixedSignalFusion
from .llm_support import (
    FrameworkLLMService,
    IncompleteBatchResponseError,
    chunk_cache_suffix,
)
from .understanding import (
    FRRequirementProfile,
    MixedRequirementProfile,
    NFRRequirementProfile,
    RequirementProfile,
)


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?")
_VARIABLE_PATTERN = re.compile(
    r"\b(?:var|let|const|int|float|double|boolean|bool|String|char)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)
_RETURN_PATTERN = re.compile(r"\breturn\b\s*([^;\n]*)")
_CONTROL_TERMS = ("if", "else", "for", "while", "switch", "try", "catch")
BATCH_ALIGNMENT_MAX_COMPLETION_TOKENS = 1800

# Candidates per LLM call. Batches larger than this reliably truncate the
# JSON response under BATCH_ALIGNMENT_MAX_COMPLETION_TOKENS (measured
# ~280 completion tokens/candidate for evidence-anchored prompts), which
# previously caused align_requirement_batch to silently fall back to a
# non-LLM local score for every candidate past the truncation point. Every
# chunk's returned candidate-ID set is validated before being trusted; an
# incomplete chunk is retried, never silently accepted.
ALIGNMENT_BATCH_SIZE = 3
ALIGNMENT_BATCH_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class CodeSemanticProfile:
    """Structured code profile used by bidirectional alignment."""

    code_id: str
    functions: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()
    quality_related_elements: tuple[str, ...] = ()
    control_flow: tuple[str, ...] = ()
    return_states: tuple[str, ...] = ()
    semantic_keywords: tuple[str, ...] = ()
    selected_context: str = ""


@dataclass(frozen=True)
class DirectionalAlignment:
    """One directional alignment result."""

    direction: str
    score: float
    coverage_label: str
    matched_elements: tuple[str, ...] = ()
    missing_elements: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def evidence_snippets(self) -> tuple[str, ...]:
        """Backward-compatible evidence field."""
        return self.matched_elements

    @property
    def placeholder(self) -> bool:
        """Alignment is implemented locally in Stage 5d."""
        return False


@dataclass(frozen=True)
class AlignmentEvidence:
    """Bidirectional alignment evidence for one requirement-code candidate."""

    requirement_id: str
    code_id: str
    requirement_type: str
    code_profile: CodeSemanticProfile
    requirement_to_code_alignment: DirectionalAlignment
    code_to_requirement_alignment: DirectionalAlignment
    matched_elements: tuple[str, ...]
    alignment_score: float
    alignment_explanation: str
    alpha: float = 0.5
    functional_alignment_score: float | None = None
    quality_alignment_score: float | None = None

    @property
    def requirement_to_code(self) -> DirectionalAlignment:
        """Backward-compatible directional field."""
        return self.requirement_to_code_alignment

    @property
    def code_to_requirement(self) -> DirectionalAlignment:
        """Backward-compatible directional field."""
        return self.code_to_requirement_alignment

    @property
    def placeholder(self) -> bool:
        """Alignment is implemented locally in Stage 5d."""
        return False


class BidirectionalSemanticAligner:
    """Align requirement profiles with DCAR evidence in both directions."""

    def __init__(
        self,
        alpha: float = 0.5,
        mixed_fusion: FixedMixedSignalFusion = DEFAULT_MIXED_FUSION,
        llm_service: FrameworkLLMService | None = None,
        prompt_builder: "AlignmentPromptBuilder | None" = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha
        self.mixed_fusion = mixed_fusion
        self.llm_service = llm_service
        self.prompt_builder = prompt_builder or AlignmentPromptBuilder()

    def align(
        self,
        requirement_profile: RequirementProfile,
        evidence_context: DCAREvidenceContext,
        batch_cache_key: str = "",
    ) -> tuple[AlignmentEvidence, ...]:
        """Align every candidate in a DCAR evidence context."""
        if (
            self.llm_service is not None
            and self.llm_service.live_api
            and evidence_context.candidates
        ):
            batched = self.align_requirement_batch(
                requirement_profile,
                evidence_context,
                batch_cache_key=batch_cache_key,
            )
            if batched:
                return batched
        return tuple(
            self.align_candidate(requirement_profile, candidate, use_llm=False)
            for candidate in evidence_context.candidates
        )

    def align_candidate(
        self,
        requirement_profile: RequirementProfile,
        candidate: FilteredCandidate,
        code_text: str = "",
        *,
        use_llm: bool = True,
    ) -> AlignmentEvidence:
        """Align one filtered candidate with one requirement profile."""
        code_profile = self.build_code_profile(
            candidate.code_id,
            code_text=code_text,
            summary=candidate.summary,
            selected_context=candidate.selected_context,
        )
        return self.align_bidirectional(
            requirement_profile, code_profile, use_llm=use_llm
        )

    def align_requirement_batch(
        self,
        requirement_profile: RequirementProfile,
        evidence_context: DCAREvidenceContext,
        *,
        batch_cache_key: str = "",
    ) -> tuple[AlignmentEvidence, ...]:
        """Align all candidates for one requirement, chunked into
        deterministic batches of ALIGNMENT_BATCH_SIZE candidates per LLM
        call. Each chunk's returned candidate-ID set is validated; an
        incomplete or unparseable response is retried (never silently
        replaced by a local fallback score), and if still incomplete after
        ALIGNMENT_BATCH_MAX_ATTEMPTS, raises IncompleteBatchResponseError to
        hard-fail the whole run."""
        candidates = tuple(evidence_context.candidates)
        code_profiles = tuple(
            self.build_code_profile(
                candidate.code_id,
                summary=candidate.summary,
                selected_context=candidate.selected_context,
            )
            for candidate in candidates
        )
        if (
            self.llm_service is None
            or not self.llm_service.live_api
            or not candidates
        ):
            return tuple(
                self.align_bidirectional(requirement_profile, code_profile, use_llm=False)
                for code_profile in code_profiles
            )

        parsed_by_code_id: dict[str, AlignmentEvidence] = {}
        for start in range(0, len(code_profiles), ALIGNMENT_BATCH_SIZE):
            chunk_candidates = candidates[start : start + ALIGNMENT_BATCH_SIZE]
            chunk_profiles = code_profiles[start : start + ALIGNMENT_BATCH_SIZE]
            expected_ids = {profile.code_id for profile in chunk_profiles}
            chunk_local_fallback = {
                code_profile.code_id: self.align_bidirectional(
                    requirement_profile, code_profile, use_llm=False
                )
                for code_profile in chunk_profiles
            }

            def _validate(
                payload: dict[str, Any],
                expected_ids: set[str] = expected_ids,
                chunk_profiles: tuple[CodeSemanticProfile, ...] = chunk_profiles,
                chunk_local_fallback: dict[str, AlignmentEvidence] = chunk_local_fallback,
            ) -> bool:
                # Strict mode: a candidate only counts as present if every
                # score field the schema requires parses to a valid [0,1]
                # number -- matching candidate-ID presence alone is not
                # enough, since _alignment_from_payload can otherwise
                # silently fill missing fields from the local fallback.
                strict_parsed = _batch_alignment_payload(
                    payload, requirement_profile, chunk_profiles, self.alpha,
                    chunk_local_fallback, strict=True,
                )
                return expected_ids.issubset(strict_parsed.keys())

            result = self.llm_service.complete_json(
                module="alignment_requirement_batch_chunked_b" + str(ALIGNMENT_BATCH_SIZE),
                prompt=self.prompt_builder.build_batch_prompt(
                    requirement_profile,
                    chunk_candidates,
                    chunk_profiles,
                ),
                requirement_id=requirement_profile.requirement_id,
                code_file=chunk_cache_suffix(
                    batch_cache_key, tuple(p.code_id for p in chunk_profiles)
                ),
                max_completion_tokens=BATCH_ALIGNMENT_MAX_COMPLETION_TOKENS,
                validate=_validate,
                max_attempts=ALIGNMENT_BATCH_MAX_ATTEMPTS,
            )
            # Re-parse in the same strict mode used to validate: the
            # evidence that enters predictions/metrics must be exactly
            # what was confirmed complete, never a laxer, fallback-blended
            # re-parse of the same payload.
            chunk_parsed = _batch_alignment_payload(
                result.payload,
                requirement_profile,
                chunk_profiles,
                self.alpha,
                chunk_local_fallback,
                strict=True,
            )
            for code_profile in chunk_profiles:
                evidence = chunk_parsed.get(code_profile.code_id)
                if evidence is None:
                    raise IncompleteBatchResponseError(
                        f"alignment chunk for requirement "
                        f"{requirement_profile.requirement_id} validated the "
                        f"candidate-ID set but produced no usable evidence for "
                        f"{code_profile.code_id}."
                    )
                parsed_by_code_id[code_profile.code_id] = evidence

        return tuple(
            parsed_by_code_id[code_profile.code_id] for code_profile in code_profiles
        )

    def build_code_profile(
        self,
        code_id: str,
        code_text: str = "",
        summary: CodeSummary | None = None,
        selected_context: str = "",
    ) -> CodeSemanticProfile:
        """Build a code semantic profile without LLM calls."""
        summary_functions = summary.key_functions if summary else ()
        summary_classes = summary.key_classes if summary else ()
        quality_elements = summary.quality_related_elements if summary else ()
        context = selected_context or summary.description if summary else ""
        source = "\n".join(item for item in (code_text, context) if item)
        return CodeSemanticProfile(
            code_id=code_id,
            functions=summary_functions,
            classes=summary_classes,
            variables=_extract_unique(_VARIABLE_PATTERN, source),
            quality_related_elements=quality_elements,
            control_flow=tuple(term for term in _CONTROL_TERMS if term in _tokens(source)),
            return_states=_extract_unique(_RETURN_PATTERN, source),
            semantic_keywords=_top_keywords(
                " ".join(
                    (
                        source,
                        " ".join(summary_functions),
                        " ".join(summary_classes),
                        " ".join(quality_elements),
                    )
                )
            ),
            selected_context=context,
        )

    def align_requirement_to_code(
        self,
        requirement_profile: RequirementProfile,
        code_profile: CodeSemanticProfile,
        *,
        use_llm: bool = True,
    ) -> DirectionalAlignment:
        """Assess whether code evidence covers the requirement profile."""
        if use_llm and self.llm_service is not None and self.llm_service.live_api:
            result = self.llm_service.complete_json(
                module="alignment_requirement_to_code",
                prompt=self.prompt_builder.build_requirement_to_code_prompt(
                    requirement_profile, code_profile
                ),
                requirement_id=requirement_profile.requirement_id,
                code_file=code_profile.code_id,
            )
            parsed = _directional_from_payload(
                result.payload, "requirement_to_code"
            )
            if parsed is not None:
                return parsed
        required_terms = self._requirement_terms(requirement_profile)
        code_terms = self._code_terms(code_profile)
        matched = tuple(sorted(required_terms & code_terms))
        missing = tuple(sorted(required_terms - code_terms))
        score = _coverage_score(required_terms, matched)
        return DirectionalAlignment(
            direction="requirement_to_code",
            score=score,
            coverage_label=_coverage_label(score),
            matched_elements=matched,
            missing_elements=missing,
            rationale=(
                "Local alignment compares type-aware requirement profile terms "
                "against DCAR code evidence."
            ),
        )

    def align_code_to_requirement(
        self,
        code_profile: CodeSemanticProfile,
        requirement_profile: RequirementProfile,
        *,
        use_llm: bool = True,
    ) -> DirectionalAlignment:
        """Assess whether code-side evidence supports the requirement."""
        if use_llm and self.llm_service is not None and self.llm_service.live_api:
            result = self.llm_service.complete_json(
                module="alignment_code_to_requirement",
                prompt=self.prompt_builder.build_code_to_requirement_prompt(
                    code_profile, requirement_profile
                ),
                requirement_id=requirement_profile.requirement_id,
                code_file=code_profile.code_id,
            )
            parsed = _directional_from_payload(
                result.payload, "code_to_requirement"
            )
            if parsed is not None:
                return parsed
        requirement_terms = self._requirement_terms(requirement_profile)
        code_terms = self._code_terms(code_profile)
        evidence_terms = self._salient_code_terms(code_profile)
        matched = tuple(sorted(evidence_terms & requirement_terms))
        unsupported = tuple(sorted(evidence_terms - requirement_terms))
        if not evidence_terms:
            score = 0.0
        else:
            direct_support = len(matched) / len(evidence_terms)
            broader_support = _coverage_score(requirement_terms, code_terms & requirement_terms)
            score = 0.6 * direct_support + 0.4 * broader_support
        return DirectionalAlignment(
            direction="code_to_requirement",
            score=max(0.0, min(1.0, score)),
            coverage_label=_coverage_label(score),
            matched_elements=matched,
            missing_elements=unsupported,
            rationale=(
                "Local reverse alignment checks whether salient code evidence "
                "supports the requirement profile."
            ),
        )

    def align_bidirectional(
        self,
        requirement_profile: RequirementProfile,
        code_profile: CodeSemanticProfile,
        *,
        use_llm: bool = True,
    ) -> AlignmentEvidence:
        """Combine requirement-to-code and code-to-requirement alignment."""
        requirement_to_code = self.align_requirement_to_code(
            requirement_profile, code_profile, use_llm=use_llm
        )
        code_to_requirement = self.align_code_to_requirement(
            code_profile, requirement_profile, use_llm=use_llm
        )
        functional_score: float | None = None
        quality_score: float | None = None
        if isinstance(requirement_profile, MixedRequirementProfile):
            functional_score = self._alignment_score_for_profile(
                requirement_profile.functional_profile,
                code_profile,
            )
            quality_score = self._alignment_score_for_profile(
                requirement_profile.quality_profile,
                code_profile,
            )
            score = self.mixed_fusion.combine(functional_score, quality_score)
        else:
            score = (
                self.alpha * requirement_to_code.score
                + (1.0 - self.alpha) * code_to_requirement.score
            )
        matched = tuple(
            sorted(
                set(requirement_to_code.matched_elements)
                | set(code_to_requirement.matched_elements)
            )
        )
        return AlignmentEvidence(
            requirement_id=requirement_profile.requirement_id,
            code_id=code_profile.code_id,
            requirement_type=requirement_profile.requirement_type,
            code_profile=code_profile,
            requirement_to_code_alignment=requirement_to_code,
            code_to_requirement_alignment=code_to_requirement,
            matched_elements=matched,
            alignment_score=max(0.0, min(1.0, score)),
            alignment_explanation=(
                f"R->C {requirement_to_code.coverage_label}; "
                f"C->R {code_to_requirement.coverage_label}; "
                f"matched: {', '.join(matched) if matched else 'none'}."
            ),
            alpha=self.alpha,
            functional_alignment_score=functional_score,
            quality_alignment_score=quality_score,
        )

    def _alignment_score_for_profile(
        self,
        requirement_profile: FRRequirementProfile | NFRRequirementProfile,
        code_profile: CodeSemanticProfile,
    ) -> float:
        requirement_to_code = self.align_requirement_to_code(
            requirement_profile, code_profile, use_llm=False
        )
        code_to_requirement = self.align_code_to_requirement(
            code_profile, requirement_profile, use_llm=False
        )
        return max(
            0.0,
            min(
                1.0,
                self.alpha * requirement_to_code.score
                + (1.0 - self.alpha) * code_to_requirement.score,
            ),
        )

    def _requirement_terms(self, profile: RequirementProfile) -> set[str]:
        if isinstance(profile, FRRequirementProfile):
            return _tokens(
                " ".join(
                    (
                        profile.intent,
                        profile.action,
                        profile.object,
                        profile.constraint,
                        " ".join(profile.entities),
                        profile.expected_behavior,
                    )
                )
            )
        if isinstance(profile, NFRRequirementProfile):
            return _tokens(
                " ".join(
                    (
                        profile.quality_attribute,
                        profile.quality_metric,
                        profile.scope,
                        profile.constraint,
                        profile.context,
                        profile.priority,
                        profile.evaluation_condition,
                        evidence_cue_text(profile.evidence_cues),
                    )
                )
            )
        if isinstance(profile, MixedRequirementProfile):
            return (
                self._requirement_terms(profile.functional_profile)
                | self._requirement_terms(profile.quality_profile)
            )
        return _tokens(profile.raw_text)

    @staticmethod
    def _code_terms(profile: CodeSemanticProfile) -> set[str]:
        return _tokens(
            " ".join(
                (
                    profile.code_id,
                    " ".join(profile.functions),
                    " ".join(profile.classes),
                    " ".join(profile.variables),
                    " ".join(profile.quality_related_elements),
                    " ".join(profile.control_flow),
                    " ".join(profile.return_states),
                    " ".join(profile.semantic_keywords),
                    profile.selected_context,
                )
            )
        )

    @staticmethod
    def _salient_code_terms(profile: CodeSemanticProfile) -> set[str]:
        salient = (
            set(profile.functions)
            | set(profile.classes)
            | set(profile.quality_related_elements)
            | set(profile.return_states)
            | set(profile.semantic_keywords[:8])
        )
        return {item.casefold() for item in salient if item}


CodeProfile = CodeSemanticProfile
AlignmentResult = DirectionalAlignment
BidirectionalAlignmentResult = AlignmentEvidence


def _tokens(value: str) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(value)}


def _extract_unique(pattern: re.Pattern[str], value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in pattern.findall(value) if item.strip()))[:20]


def _top_keywords(value: str, limit: int = 20) -> tuple[str, ...]:
    ignored = {
        "the",
        "and",
        "for",
        "with",
        "return",
        "class",
        "void",
        "int",
        "string",
        "boolean",
        "true",
        "false",
        "none",
    }
    tokens = [item for item in _tokens(value) if item not in ignored and len(item) > 2]
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return tuple(
        item for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    )


def _coverage_score(required_terms: set[str], matched_terms: tuple[str, ...] | set[str]) -> float:
    if not required_terms:
        return 0.0
    return len(set(matched_terms)) / len(required_terms)


def _coverage_label(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.4:
        return "partial"
    if score > 0.0:
        return "weak"
    return "none"


class AlignmentPromptBuilder:
    """Build bidirectional semantic-alignment prompts."""

    def build_batch_prompt(
        self,
        requirement_profile: RequirementProfile,
        candidates: tuple[FilteredCandidate, ...],
        code_profiles: tuple[CodeSemanticProfile, ...],
    ) -> str:
        candidate_blocks = []
        for candidate, code_profile in zip(candidates, code_profiles, strict=True):
            candidate_blocks.append(
                {
                    "code_id": candidate.code_id,
                    "retrieval_rank": candidate.rank,
                    "retrieval_score": candidate.retrieval_score,
                    "relevance_score": candidate.relevance_score,
                    "code_profile": code_profile,
                }
            )
        return f"""IST bidirectional semantic alignment batch.
Return only valid JSON.
Use concise phrases: max 3 matched_elements, max 3 missing_elements, rationale <= 12 words.
Requirement type: {requirement_profile.requirement_type}
Requirement profile: {requirement_profile}
Candidates: {candidate_blocks}
Schema:
{{
  "candidates": [
    {{
      "code_id": "candidate id",
      "requirement_to_code_alignment": {{
        "score": 0.0,
        "coverage_label": "strong|partial|weak|none",
        "matched_elements": ["terms or evidence that match"],
        "missing_elements": ["important requirement elements not covered"],
        "rationale": "brief alignment rationale"
      }},
      "code_to_requirement_alignment": {{
        "score": 0.0,
        "coverage_label": "strong|partial|weak|none",
        "matched_elements": ["code evidence that supports the requirement"],
        "missing_elements": ["code-side evidence not supported by requirement"],
        "rationale": "brief reverse-alignment rationale"
      }},
      "matched_elements": ["combined matched elements"],
      "functional_alignment_score": 0.0,
      "quality_alignment_score": 0.0,
      "alignment_score": 0.0,
      "alignment_explanation": "brief bidirectional explanation"
    }}
  ]
}}"""

    def build_requirement_to_code_prompt(
        self,
        requirement_profile: RequirementProfile,
        code_profile: CodeSemanticProfile,
    ) -> str:
        return f"""IST bidirectional alignment: Requirement -> Code.
Return only valid JSON.
Requirement type: {requirement_profile.requirement_type}
Requirement profile: {requirement_profile}
Code semantic profile: {code_profile}
Schema:
{{
  "score": 0.0,
  "coverage_label": "strong|partial|weak|none",
  "matched_elements": ["terms or evidence that match"],
  "missing_elements": ["important requirement elements not covered"],
  "rationale": "brief alignment rationale"
}}"""

    def build_code_to_requirement_prompt(
        self,
        code_profile: CodeSemanticProfile,
        requirement_profile: RequirementProfile,
    ) -> str:
        return f"""IST bidirectional alignment: Code -> Requirement.
Return only valid JSON.
Code semantic profile: {code_profile}
Requirement type: {requirement_profile.requirement_type}
Requirement profile: {requirement_profile}
Schema:
{{
  "score": 0.0,
  "coverage_label": "strong|partial|weak|none",
  "matched_elements": ["code evidence that supports the requirement"],
  "missing_elements": ["code-side evidence not supported by requirement"],
  "rationale": "brief reverse-alignment rationale"
}}"""


def _directional_from_payload(
    payload: dict[str, Any],
    direction: str,
    *,
    strict: bool = False,
) -> DirectionalAlignment | None:
    if not payload:
        return None
    if strict:
        score = _strict_unit_score(payload.get("score"))
        if score is None:
            return None
    else:
        try:
            score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
        except (TypeError, ValueError):
            score = 0.0
    return DirectionalAlignment(
        direction=direction,
        score=score,
        coverage_label=str(payload.get("coverage_label") or _coverage_label(score)),
        matched_elements=_tuple_of_strings(payload.get("matched_elements")),
        missing_elements=_tuple_of_strings(payload.get("missing_elements")),
        rationale=str(payload.get("rationale") or ""),
    )


def _batch_alignment_payload(
    payload: dict[str, Any],
    requirement_profile: RequirementProfile,
    code_profiles: tuple[CodeSemanticProfile, ...],
    alpha: float,
    fallback_by_code_id: dict[str, AlignmentEvidence],
    *,
    strict: bool = False,
) -> dict[str, AlignmentEvidence]:
    """Parse a batch alignment response into AlignmentEvidence per code_id.

    In strict mode (used by the chunked-batching validate/build path), a
    candidate is only included if every score field the prompt schema
    requires is present and a valid [0,1] number -- never silently filled
    in from `fallback_by_code_id`. Non-strict mode preserves the original
    single-call behavior (missing/invalid fields fall back to the local
    heuristic) for any other caller."""
    items = payload.get("candidates")
    if not isinstance(items, list):
        return {}
    profiles = {profile.code_id: profile for profile in code_profiles}
    parsed: dict[str, AlignmentEvidence] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        code_id = str(item.get("code_id") or "")
        code_profile = profiles.get(code_id)
        if code_profile is None:
            continue
        evidence = _alignment_from_payload(
            item,
            requirement_profile,
            code_profile,
            alpha,
            fallback_by_code_id[code_id],
            strict=strict,
        )
        if evidence is not None:
            parsed[code_id] = evidence
    return parsed


def _alignment_from_payload(
    payload: dict[str, Any],
    requirement_profile: RequirementProfile,
    code_profile: CodeSemanticProfile,
    alpha: float,
    fallback: AlignmentEvidence,
    *,
    strict: bool = False,
) -> AlignmentEvidence | None:
    r2c = _directional_from_payload(
        _dict_payload(payload.get("requirement_to_code_alignment")),
        "requirement_to_code",
        strict=strict,
    )
    c2r = _directional_from_payload(
        _dict_payload(payload.get("code_to_requirement_alignment")),
        "code_to_requirement",
        strict=strict,
    )
    if strict and (r2c is None or c2r is None):
        return None
    r2c = r2c or fallback.requirement_to_code_alignment
    c2r = c2r or fallback.code_to_requirement_alignment

    matched = _tuple_of_strings(payload.get("matched_elements"))
    if not matched:
        matched = tuple(sorted(set(r2c.matched_elements) | set(c2r.matched_elements)))

    if strict:
        functional_score = _strict_unit_score(payload.get("functional_alignment_score"))
        quality_score = _strict_unit_score(payload.get("quality_alignment_score"))
        if functional_score is None or quality_score is None:
            return None
        score = _strict_unit_score(payload.get("alignment_score"))
        if score is None:
            return None
    else:
        functional_score = _optional_score(
            payload.get("functional_alignment_score"),
            fallback.functional_alignment_score,
        )
        quality_score = _optional_score(
            payload.get("quality_alignment_score"),
            fallback.quality_alignment_score,
        )
        if isinstance(requirement_profile, MixedRequirementProfile):
            if functional_score is None:
                functional_score = fallback.functional_alignment_score
            if quality_score is None:
                quality_score = fallback.quality_alignment_score
        try:
            score = float(payload.get("alignment_score"))
        except (TypeError, ValueError):
            score = alpha * r2c.score + (1.0 - alpha) * c2r.score

    if (
        isinstance(requirement_profile, MixedRequirementProfile)
        and functional_score is not None
        and quality_score is not None
    ):
        score = DEFAULT_MIXED_FUSION.combine(functional_score, quality_score)
    score = max(0.0, min(1.0, score))
    return AlignmentEvidence(
        requirement_id=requirement_profile.requirement_id,
        code_id=code_profile.code_id,
        requirement_type=requirement_profile.requirement_type,
        code_profile=code_profile,
        requirement_to_code_alignment=r2c,
        code_to_requirement_alignment=c2r,
        matched_elements=matched,
        alignment_score=score,
        alignment_explanation=str(
            payload.get("alignment_explanation")
            or (
                f"R->C {r2c.coverage_label}; C->R {c2r.coverage_label}; "
                f"matched: {', '.join(matched) if matched else 'none'}."
            )
        ),
        alpha=alpha,
        functional_alignment_score=functional_score,
        quality_alignment_score=quality_score,
    )


def _dict_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_score(value: Any, fallback: float | None) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback


def _strict_unit_score(value: Any) -> float | None:
    """Return `value` as a float only if it is a genuine, present [0,1]
    number -- never a silently-clamped or defaulted score. Used to detect
    an incomplete batch response instead of quietly accepting it."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= score <= 1.0):
        return None
    return score


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, tuple):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()
