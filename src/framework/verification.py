"""Self-reflective verification for the IST framework."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .alignment import AlignmentEvidence, CodeSemanticProfile
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
BATCH_VERIFICATION_MAX_COMPLETION_TOKENS = 1600

# See alignment.py's ALIGNMENT_BATCH_SIZE for the rationale: batches larger
# than this reliably truncate under BATCH_VERIFICATION_MAX_COMPLETION_TOKENS
# and previously fell back silently to a non-LLM local score.
VERIFICATION_BATCH_SIZE = 3
VERIFICATION_BATCH_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class VerificationDimensionResult:
    """Score and rationale for one verification dimension."""

    score: float
    label: str
    rationale: str = ""
    supporting_terms: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()

    @property
    def placeholder(self) -> bool:
        """Verification is implemented locally in Stage 5e."""
        return False


@dataclass(frozen=True)
class VerificationEvidence:
    """Self-reflective verification evidence for one aligned candidate."""

    requirement_id: str
    code_id: str
    semantic_consistency: VerificationDimensionResult
    evidence_sufficiency: VerificationDimensionResult
    evidence_grounding: VerificationDimensionResult
    verification_score: float
    verification_explanation: str
    supported: bool
    functional_verification_score: float | None = None
    quality_verification_score: float | None = None

    @property
    def placeholder(self) -> bool:
        """Verification is implemented locally in Stage 5e."""
        return False


class SelfReflectiveVerifier:
    """Verify semantic consistency, evidence sufficiency, and grounding."""

    def __init__(
        self,
        consistency_weight: float = 0.33,
        sufficiency_weight: float = 0.33,
        grounding_weight: float = 0.34,
        support_threshold: float = 0.5,
        mixed_fusion: FixedMixedSignalFusion = DEFAULT_MIXED_FUSION,
        llm_service: FrameworkLLMService | None = None,
        prompt_builder: "VerificationPromptBuilder | None" = None,
    ):
        total = consistency_weight + sufficiency_weight + grounding_weight
        if total <= 0.0:
            raise ValueError("verification weights must sum to a positive value.")
        self.consistency_weight = consistency_weight / total
        self.sufficiency_weight = sufficiency_weight / total
        self.grounding_weight = grounding_weight / total
        self.support_threshold = support_threshold
        self.mixed_fusion = mixed_fusion
        self.llm_service = llm_service
        self.prompt_builder = prompt_builder or VerificationPromptBuilder()

    def verify(
        self,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
    ) -> VerificationEvidence:
        """Run all three self-reflective verification checks."""
        if self.llm_service is not None and self.llm_service.live_api:
            batched = self._verify_with_llm_batch(requirement_profile, alignment)
            if batched is not None:
                return batched
        consistency = self.check_semantic_consistency(
            requirement_profile, alignment, use_llm=False
        )
        sufficiency = self.check_evidence_sufficiency(
            requirement_profile, alignment, use_llm=False
        )
        grounding = self.check_evidence_grounding(
            requirement_profile, alignment, use_llm=False
        )
        return self._build_verification_evidence(
            requirement_profile,
            alignment,
            consistency,
            sufficiency,
            grounding,
        )

    def verify_batch(
        self,
        requirement_profile: RequirementProfile,
        alignments: tuple[AlignmentEvidence, ...],
        *,
        batch_cache_key: str = "",
    ) -> tuple[VerificationEvidence, ...]:
        """Verify all candidates for one requirement, chunked into
        deterministic batches of VERIFICATION_BATCH_SIZE candidates per LLM
        call. Each chunk's returned candidate-ID set is validated; an
        incomplete or unparseable response is retried (never silently
        replaced by a local fallback score), and if still incomplete after
        VERIFICATION_BATCH_MAX_ATTEMPTS, raises IncompleteBatchResponseError
        to hard-fail the whole run."""
        if not alignments:
            return ()
        if self.llm_service is None or not self.llm_service.live_api:
            return tuple(
                self._verify_locally(requirement_profile, alignment)
                for alignment in alignments
            )

        parsed_by_code_id: dict[str, VerificationEvidence] = {}
        for start in range(0, len(alignments), VERIFICATION_BATCH_SIZE):
            chunk = alignments[start : start + VERIFICATION_BATCH_SIZE]
            expected_ids = {alignment.code_id for alignment in chunk}
            chunk_local_fallback = {
                alignment.code_id: self._verify_locally(requirement_profile, alignment)
                for alignment in chunk
            }

            def _validate(
                payload: dict[str, Any],
                expected_ids: set[str] = expected_ids,
                chunk: tuple[AlignmentEvidence, ...] = chunk,
                chunk_local_fallback: dict[str, VerificationEvidence] = chunk_local_fallback,
            ) -> bool:
                # Strict mode: a candidate only counts as present if every
                # score field the schema requires parses to a valid [0,1]
                # number -- matching candidate-ID presence alone is not
                # enough, since _verification_from_payload can otherwise
                # silently fill missing fields from the local fallback.
                strict_parsed = _batch_verification_payload(
                    payload, requirement_profile, chunk, self, chunk_local_fallback,
                    strict=True,
                )
                return expected_ids.issubset(strict_parsed.keys())

            result = self.llm_service.complete_json(
                module="verification_requirement_batch_chunked_b" + str(VERIFICATION_BATCH_SIZE),
                prompt=self.prompt_builder.build_requirement_batch_prompt(
                    requirement_profile,
                    chunk,
                ),
                requirement_id=requirement_profile.requirement_id,
                code_file=chunk_cache_suffix(
                    batch_cache_key, tuple(a.code_id for a in chunk)
                ),
                max_completion_tokens=BATCH_VERIFICATION_MAX_COMPLETION_TOKENS,
                validate=_validate,
                max_attempts=VERIFICATION_BATCH_MAX_ATTEMPTS,
            )
            # Re-parse in the same strict mode used to validate: the
            # evidence that enters predictions/metrics must be exactly
            # what was confirmed complete, never a laxer, fallback-blended
            # re-parse of the same payload.
            chunk_parsed = _batch_verification_payload(
                result.payload,
                requirement_profile,
                chunk,
                self,
                chunk_local_fallback,
                strict=True,
            )
            for alignment in chunk:
                evidence = chunk_parsed.get(alignment.code_id)
                if evidence is None:
                    raise IncompleteBatchResponseError(
                        f"verification chunk for requirement "
                        f"{requirement_profile.requirement_id} validated the "
                        f"candidate-ID set but produced no usable evidence for "
                        f"{alignment.code_id}."
                    )
                parsed_by_code_id[alignment.code_id] = evidence

        return tuple(parsed_by_code_id[alignment.code_id] for alignment in alignments)

    def _verify_locally(
        self,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
    ) -> VerificationEvidence:
        consistency = self.check_semantic_consistency(
            requirement_profile, alignment, use_llm=False
        )
        sufficiency = self.check_evidence_sufficiency(
            requirement_profile, alignment, use_llm=False
        )
        grounding = self.check_evidence_grounding(
            requirement_profile, alignment, use_llm=False
        )
        return self._build_verification_evidence(
            requirement_profile,
            alignment,
            consistency,
            sufficiency,
            grounding,
        )

    def _verify_with_llm_batch(
        self,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
    ) -> VerificationEvidence | None:
        result = self.llm_service.complete_json(
            module="verification_batch",
            prompt=self.prompt_builder.build_batch_prompt(
                requirement_profile, alignment
            ),
            requirement_id=requirement_profile.requirement_id,
            code_file=alignment.code_id,
        )
        consistency = _dimension_from_payload(
            _dict_payload(result.payload.get("semantic_consistency"))
        )
        sufficiency = _dimension_from_payload(
            _dict_payload(result.payload.get("evidence_sufficiency"))
        )
        grounding = _dimension_from_payload(
            _dict_payload(result.payload.get("evidence_grounding"))
        )
        if consistency is None or sufficiency is None or grounding is None:
            return None
        return self._build_verification_evidence(
            requirement_profile,
            alignment,
            consistency,
            sufficiency,
            grounding,
        )

    def _build_verification_evidence(
        self,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
        consistency: VerificationDimensionResult,
        sufficiency: VerificationDimensionResult,
        grounding: VerificationDimensionResult,
    ) -> VerificationEvidence:
        functional_score: float | None = None
        quality_score: float | None = None
        if isinstance(requirement_profile, MixedRequirementProfile):
            functional_score = self._local_verification_score_for_profile(
                requirement_profile.functional_profile,
                alignment,
            )
            quality_score = self._local_verification_score_for_profile(
                requirement_profile.quality_profile,
                alignment,
            )
            score = self.mixed_fusion.combine(functional_score, quality_score)
        else:
            score = self._weighted_score(consistency, sufficiency, grounding)
        score = max(0.0, min(1.0, score))
        return VerificationEvidence(
            requirement_id=requirement_profile.requirement_id,
            code_id=alignment.code_id,
            semantic_consistency=consistency,
            evidence_sufficiency=sufficiency,
            evidence_grounding=grounding,
            verification_score=score,
            verification_explanation=(
                f"Consistency={consistency.label}; "
                f"Sufficiency={sufficiency.label}; "
                f"Grounding={grounding.label}."
            ),
            supported=score >= self.support_threshold,
            functional_verification_score=functional_score,
            quality_verification_score=quality_score,
        )

    def _weighted_score(
        self,
        consistency: VerificationDimensionResult,
        sufficiency: VerificationDimensionResult,
        grounding: VerificationDimensionResult,
    ) -> float:
        return (
            self.consistency_weight * consistency.score
            + self.sufficiency_weight * sufficiency.score
            + self.grounding_weight * grounding.score
        )

    def _local_verification_score_for_profile(
        self,
        requirement_profile: FRRequirementProfile | NFRRequirementProfile,
        alignment: AlignmentEvidence,
    ) -> float:
        consistency = self.check_semantic_consistency(
            requirement_profile, alignment, use_llm=False
        )
        sufficiency = self.check_evidence_sufficiency(
            requirement_profile, alignment, use_llm=False
        )
        grounding = self.check_evidence_grounding(
            requirement_profile, alignment, use_llm=False
        )
        return max(0.0, min(1.0, self._weighted_score(consistency, sufficiency, grounding)))

    def check_semantic_consistency(
        self,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
        *,
        use_llm: bool = True,
    ) -> VerificationDimensionResult:
        """Check whether requirement meaning agrees with alignment evidence."""
        if use_llm and self.llm_service is not None and self.llm_service.live_api:
            result = self.llm_service.complete_json(
                module="verification_semantic_consistency",
                prompt=self.prompt_builder.build_semantic_consistency_prompt(
                    requirement_profile, alignment
                ),
                requirement_id=requirement_profile.requirement_id,
                code_file=alignment.code_id,
            )
            parsed = _dimension_from_payload(result.payload)
            if parsed is not None:
                return parsed
        requirement_terms = _requirement_terms(requirement_profile)
        matched_terms = set(alignment.matched_elements)
        directional_average = (
            alignment.requirement_to_code_alignment.score
            + alignment.code_to_requirement_alignment.score
        ) / 2.0
        coverage = _coverage(requirement_terms, matched_terms)
        score = 0.55 * directional_average + 0.45 * coverage
        missing = tuple(sorted(requirement_terms - matched_terms))
        return VerificationDimensionResult(
            score=max(0.0, min(1.0, score)),
            label=_label(score),
            rationale=(
                "Checks whether matched alignment elements cover the "
                "type-aware requirement profile."
            ),
            supporting_terms=tuple(sorted(matched_terms & requirement_terms)),
            missing_terms=missing,
        )

    def check_evidence_sufficiency(
        self,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
        *,
        use_llm: bool = True,
    ) -> VerificationDimensionResult:
        """Check whether retrieved code evidence is enough to support the link."""
        if use_llm and self.llm_service is not None and self.llm_service.live_api:
            result = self.llm_service.complete_json(
                module="verification_evidence_sufficiency",
                prompt=self.prompt_builder.build_evidence_sufficiency_prompt(
                    requirement_profile, alignment
                ),
                requirement_id=requirement_profile.requirement_id,
                code_file=alignment.code_id,
            )
            parsed = _dimension_from_payload(result.payload)
            if parsed is not None:
                return parsed
        code_profile = alignment.code_profile
        evidence_units = _evidence_units(code_profile)
        requirement_terms = _requirement_terms(requirement_profile)
        support_terms = _code_terms(code_profile) & requirement_terms
        evidence_density = min(1.0, evidence_units / 6.0)
        support_coverage = _coverage(requirement_terms, support_terms)
        score = 0.45 * evidence_density + 0.35 * support_coverage + 0.20 * alignment.alignment_score
        return VerificationDimensionResult(
            score=max(0.0, min(1.0, score)),
            label=_label(score),
            rationale=(
                "Checks whether DCAR provided enough functions, classes, "
                "quality cues, control flow, return states, or context."
            ),
            supporting_terms=tuple(sorted(support_terms)),
            missing_terms=tuple(sorted(requirement_terms - support_terms)),
        )

    def check_evidence_grounding(
        self,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
        *,
        use_llm: bool = True,
    ) -> VerificationDimensionResult:
        """Check whether alignment claims are grounded in retrieved evidence."""
        if use_llm and self.llm_service is not None and self.llm_service.live_api:
            result = self.llm_service.complete_json(
                module="verification_evidence_grounding",
                prompt=self.prompt_builder.build_evidence_grounding_prompt(
                    requirement_profile, alignment
                ),
                requirement_id=requirement_profile.requirement_id,
                code_file=alignment.code_id,
            )
            parsed = _dimension_from_payload(result.payload)
            if parsed is not None:
                return parsed
        code_terms = _code_terms(alignment.code_profile)
        matched_terms = set(alignment.matched_elements)
        grounded = matched_terms & code_terms
        ungrounded = matched_terms - code_terms
        score = _coverage(matched_terms, grounded)
        if not matched_terms and alignment.alignment_score > 0.0:
            score = 0.0
        return VerificationDimensionResult(
            score=max(0.0, min(1.0, score)),
            label=_label(score),
            rationale=(
                "Checks whether matched alignment terms appear in the retrieved "
                "code profile or selected evidence context."
            ),
            supporting_terms=tuple(sorted(grounded)),
            missing_terms=tuple(sorted(ungrounded)),
        )


VerificationResult = VerificationEvidence


def _requirement_terms(profile: RequirementProfile) -> set[str]:
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
            _requirement_terms(profile.functional_profile)
            | _requirement_terms(profile.quality_profile)
        )
    return _tokens(profile.raw_text)


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


def _tokens(value: str) -> set[str]:
    return {item.casefold() for item in _TOKEN.findall(value)}


def _coverage(expected: set[str], observed: set[str]) -> float:
    if not expected:
        return 0.0
    return len(expected & observed) / len(expected)


def _evidence_units(profile: CodeSemanticProfile) -> int:
    return sum(
        bool(items)
        for items in (
            profile.functions,
            profile.classes,
            profile.variables,
            profile.quality_related_elements,
            profile.control_flow,
            profile.return_states,
            profile.semantic_keywords,
            profile.selected_context,
        )
    )


def _label(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.5:
        return "moderate"
    if score > 0.0:
        return "weak"
    return "none"


class VerificationPromptBuilder:
    """Build self-reflective verification prompts."""

    def build_requirement_batch_prompt(
        self,
        requirement_profile: RequirementProfile,
        alignments: tuple[AlignmentEvidence, ...],
    ) -> str:
        items = [
            {
                "code_id": alignment.code_id,
                "alignment_score": alignment.alignment_score,
                "matched_elements": alignment.matched_elements,
                "alignment_explanation": alignment.alignment_explanation,
                "requirement_to_code_alignment": (
                    alignment.requirement_to_code_alignment
                ),
                "code_to_requirement_alignment": (
                    alignment.code_to_requirement_alignment
                ),
            }
            for alignment in alignments
        ]
        return f"""IST self-reflective verification batch.
Evaluate semantic consistency, evidence sufficiency, and evidence grounding for each candidate.
Return only valid JSON.
Use concise phrases: max 3 supporting_terms, max 3 missing_terms, rationale <= 12 words.
Requirement profile: {requirement_profile}
Alignment evidence by candidate: {items}
Schema:
{{
  "candidates": [
    {{
      "code_id": "candidate id",
      "semantic_consistency": {{
        "score": 0.0,
        "label": "strong|moderate|weak|none",
        "rationale": "brief verification rationale",
        "supporting_terms": ["grounded supporting terms"],
        "missing_terms": ["missing or unsupported terms"]
      }},
      "evidence_sufficiency": {{
        "score": 0.0,
        "label": "strong|moderate|weak|none",
        "rationale": "brief verification rationale",
        "supporting_terms": ["grounded supporting terms"],
        "missing_terms": ["missing or unsupported terms"]
      }},
      "evidence_grounding": {{
        "score": 0.0,
        "label": "strong|moderate|weak|none",
        "rationale": "brief verification rationale",
        "supporting_terms": ["grounded supporting terms"],
        "missing_terms": ["missing or unsupported terms"]
      }},
      "functional_verification_score": 0.0,
      "quality_verification_score": 0.0,
      "verification_score": 0.0,
      "verification_explanation": "brief combined verification explanation",
      "supported": false
    }}
  ]
}}"""

    def build_batch_prompt(
        self,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
    ) -> str:
        return f"""IST self-reflective verification.
Evaluate semantic consistency, evidence sufficiency, and evidence grounding.
Return only valid JSON.
Requirement profile: {requirement_profile}
Alignment evidence: {alignment}
Schema:
{{
  "semantic_consistency": {{
    "score": 0.0,
    "label": "strong|moderate|weak|none",
    "rationale": "brief verification rationale",
    "supporting_terms": ["grounded supporting terms"],
    "missing_terms": ["missing or unsupported terms"]
  }},
  "evidence_sufficiency": {{
    "score": 0.0,
    "label": "strong|moderate|weak|none",
    "rationale": "brief verification rationale",
    "supporting_terms": ["grounded supporting terms"],
    "missing_terms": ["missing or unsupported terms"]
  }},
  "evidence_grounding": {{
    "score": 0.0,
    "label": "strong|moderate|weak|none",
    "rationale": "brief verification rationale",
    "supporting_terms": ["grounded supporting terms"],
    "missing_terms": ["missing or unsupported terms"]
  }}
}}"""

    def build_semantic_consistency_prompt(
        self,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
    ) -> str:
        return self._build_prompt(
            "semantic consistency",
            "whether the alignment evidence is semantically consistent with the requirement profile",
            requirement_profile,
            alignment,
        )

    def build_evidence_sufficiency_prompt(
        self,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
    ) -> str:
        return self._build_prompt(
            "evidence sufficiency",
            "whether the retrieved code evidence is sufficient to support the requirement",
            requirement_profile,
            alignment,
        )

    def build_evidence_grounding_prompt(
        self,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
    ) -> str:
        return self._build_prompt(
            "evidence grounding",
            "whether the alignment conclusion is grounded in the supplied code evidence",
            requirement_profile,
            alignment,
        )

    @staticmethod
    def _build_prompt(
        dimension: str,
        task: str,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
    ) -> str:
        return f"""IST self-reflective verification: {dimension}.
Check {task}. Return only valid JSON.
Requirement profile: {requirement_profile}
Alignment evidence: {alignment}
Schema:
{{
  "score": 0.0,
  "label": "strong|moderate|weak|none",
  "rationale": "brief verification rationale",
  "supporting_terms": ["grounded supporting terms"],
  "missing_terms": ["missing or unsupported terms"]
}}"""


def _dimension_from_payload(
    payload: dict[str, Any],
    *,
    strict: bool = False,
) -> VerificationDimensionResult | None:
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
    return VerificationDimensionResult(
        score=score,
        label=str(payload.get("label") or _label(score)),
        rationale=str(payload.get("rationale") or ""),
        supporting_terms=_tuple_of_strings(payload.get("supporting_terms")),
        missing_terms=_tuple_of_strings(payload.get("missing_terms")),
    )


def _batch_verification_payload(
    payload: dict[str, Any],
    requirement_profile: RequirementProfile,
    alignments: tuple[AlignmentEvidence, ...],
    verifier: SelfReflectiveVerifier,
    fallback_by_code_id: dict[str, VerificationEvidence],
    *,
    strict: bool = False,
) -> dict[str, VerificationEvidence]:
    """Parse a batch verification response into VerificationEvidence per
    code_id. In strict mode (used by the chunked-batching validate/build
    path), a candidate is only included if every score field the prompt
    schema requires is present and a valid [0,1] number -- never silently
    filled in from `fallback_by_code_id`. Non-strict mode preserves the
    original single-call behavior for any other caller."""
    items = payload.get("candidates")
    if not isinstance(items, list):
        return {}
    alignment_by_code_id = {alignment.code_id: alignment for alignment in alignments}
    parsed: dict[str, VerificationEvidence] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        code_id = str(item.get("code_id") or "")
        alignment = alignment_by_code_id.get(code_id)
        if alignment is None:
            continue
        evidence = _verification_from_payload(
            item,
            requirement_profile,
            alignment,
            verifier,
            fallback_by_code_id[code_id],
            strict=strict,
        )
        if evidence is not None:
            parsed[code_id] = evidence
    return parsed


def _verification_from_payload(
    payload: dict[str, Any],
    requirement_profile: RequirementProfile,
    alignment: AlignmentEvidence,
    verifier: SelfReflectiveVerifier,
    fallback: VerificationEvidence,
    *,
    strict: bool = False,
) -> VerificationEvidence | None:
    consistency = _dimension_from_payload(
        _dict_payload(payload.get("semantic_consistency")), strict=strict
    )
    sufficiency = _dimension_from_payload(
        _dict_payload(payload.get("evidence_sufficiency")), strict=strict
    )
    grounding = _dimension_from_payload(
        _dict_payload(payload.get("evidence_grounding")), strict=strict
    )
    if strict and (consistency is None or sufficiency is None or grounding is None):
        return None
    consistency = consistency or fallback.semantic_consistency
    sufficiency = sufficiency or fallback.evidence_sufficiency
    grounding = grounding or fallback.evidence_grounding

    evidence = verifier._build_verification_evidence(
        requirement_profile,
        alignment,
        consistency,
        sufficiency,
        grounding,
    )

    if strict:
        functional_score = _strict_unit_score(payload.get("functional_verification_score"))
        quality_score = _strict_unit_score(payload.get("quality_verification_score"))
        if functional_score is None or quality_score is None:
            return None
        score = _strict_unit_score(payload.get("verification_score"))
        if score is None:
            return None
        supported = _strict_bool(payload.get("supported"))
        if supported is None:
            return None
    else:
        functional_score = _optional_score(
            payload.get("functional_verification_score"),
            evidence.functional_verification_score,
        )
        quality_score = _optional_score(
            payload.get("quality_verification_score"),
            evidence.quality_verification_score,
        )
        try:
            score = float(payload.get("verification_score"))
        except (TypeError, ValueError):
            score = evidence.verification_score
        supported = _bool_value(payload.get("supported"), evidence.supported)

    if (
        isinstance(requirement_profile, MixedRequirementProfile)
        and functional_score is not None
        and quality_score is not None
    ):
        score = verifier.mixed_fusion.combine(functional_score, quality_score)
    return VerificationEvidence(
        requirement_id=evidence.requirement_id,
        code_id=evidence.code_id,
        semantic_consistency=evidence.semantic_consistency,
        evidence_sufficiency=evidence.evidence_sufficiency,
        evidence_grounding=evidence.evidence_grounding,
        verification_score=max(0.0, min(1.0, score)),
        verification_explanation=str(
            payload.get("verification_explanation")
            or evidence.verification_explanation
        ),
        supported=supported,
        functional_verification_score=functional_score,
        quality_verification_score=quality_score,
    )


def _dict_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _strict_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


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
