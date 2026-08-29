"""w/o Bi-Alignment ablation: only requirement-to-code alignment.

Isolated experiment for the 1:1 pair-classification track only (see
scripts/run_pair_classification.py's ABLATIONS table). Does not modify
alignment.py's BidirectionalSemanticAligner/AlignmentPromptBuilder (used
unchanged by v1/v2/v3 and the formal top-k Proposed Framework runs) --
this module only imports plain, already-shared helpers from it
(IncompleteBatchResponseError, chunk_cache_suffix, ALIGNMENT_BATCH_SIZE,
BATCH_ALIGNMENT_MAX_COMPLETION_TOKENS) and adds new classes alongside it.

Code-to-requirement alignment is genuinely REMOVED, not just discarded
after the fact: the prompt schema never asks the model for it, so no
reasoning about "does the code support the requirement" is elicited or
paid for. `alignment_score` is always set to the requirement-to-code
score directly in code (never trusted from a model-returned blended
"alignment_score" field, and never a function of `alpha`), so the
invariant holds even if a future prompt edit accidentally re-invites a
blended score from the model.

Batching/retry/strict-validation semantics mirror
BidirectionalSemanticAligner.align_requirement_batch exactly (same
ALIGNMENT_BATCH_SIZE chunk size, same ALIGNMENT_BATCH_MAX_ATTEMPTS retries,
same hard-fail-via-IncompleteBatchResponseError-never-silent-fallback
policy) -- collapsed to a single alignment direction.

For Mixed requirements, this ablation does not compute separate
functional_alignment_score/quality_alignment_score sub-scores (those were
originally produced by the same bidirectional alpha-blend this ablation
removes): `alignment_score` is the requirement-to-code coverage of the
Mixed profile's pooled functional+quality terms, uniformly for FR/NFR/Mixed
alike, matching the ablation's literal definition ("uses only
requirement-to-code alignment") without inventing an unspecified
Mixed-only sub-scoring scheme.
"""

from __future__ import annotations

from typing import Any

from .alignment import (
    ALIGNMENT_BATCH_MAX_ATTEMPTS,
    ALIGNMENT_BATCH_SIZE,
    BATCH_ALIGNMENT_MAX_COMPLETION_TOKENS,
    AlignmentEvidence,
    BidirectionalSemanticAligner,
    CodeSemanticProfile,
    DirectionalAlignment,
)
from .dependency_context_prompts import DEPENDENCY_CONTEXT_INSTRUCTIONS, _dependency_blocks
from .evidence_anchored_prompts import ALIGNMENT_ANCHOR_RUBRIC
from .llm_support import FrameworkLLMService, IncompleteBatchResponseError, chunk_cache_suffix
from .understanding import RequirementProfile
from src.retrieval.dcar import DCAREvidenceContext
from src.retrieval.filters import FilteredCandidate
from src.retrieval.summarizer import CodeSummary

REMOVED_CODE_TO_REQUIREMENT_ALIGNMENT = DirectionalAlignment(
    direction="code_to_requirement",
    score=0.0,
    coverage_label="none",
    rationale=(
        "Removed (w/o Bi-Alignment ablation): code-to-requirement "
        "alignment is not computed."
    ),
)


class UnidirectionalAlignmentPromptBuilder:
    """Requirement-to-code alignment only, with v3's dependency-context
    block (this ablation keeps evidence anchoring and dependency context;
    it removes only the code-to-requirement direction)."""

    def __init__(
        self,
        dependency_graph: dict[str, set[str]],
        code_summaries: dict[str, CodeSummary],
        max_dependencies_shown: int = 3,
    ):
        self.dependency_graph = dependency_graph
        self.code_summaries = code_summaries
        self.max_dependencies_shown = max_dependencies_shown

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
                    "dependency_context": _dependency_blocks(
                        candidate.code_id, self.dependency_graph,
                        self.code_summaries, self.max_dependencies_shown,
                    ),
                }
            )
        return f"""IST alignment batch (w/o Bi-Alignment ablation: requirement-to-code only).
Return only valid JSON.
Use concise phrases: max 3 matched_elements, max 3 missing_elements, rationale <= 12 words.
{ALIGNMENT_ANCHOR_RUBRIC}
{DEPENDENCY_CONTEXT_INSTRUCTIONS}
Do NOT assess whether the code supports/justifies the requirement (code-to-requirement) --
only whether the code covers the requirement (requirement-to-code).
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
        "matched_elements": ["requirement-specific terms or evidence that match -- name the specific entity/field/rule/API, or the specific dependency file, not just that overlap exists"],
        "missing_elements": ["important requirement elements not covered"],
        "rationale": "brief alignment rationale, must state whether the match is requirement-specific, dependency-based (name the dependency), or only generic overlap"
      }}
    }}
  ]
}}"""


class RequirementToCodeOnlyAligner:
    """Duck-types BidirectionalSemanticAligner's `.align()`/`.build_code_profile()`
    signatures so it can be passed directly as run_framework(aligner=...)."""

    def __init__(
        self,
        llm_service: FrameworkLLMService | None = None,
        prompt_builder: UnidirectionalAlignmentPromptBuilder | None = None,
    ):
        self.llm_service = llm_service
        self.prompt_builder = prompt_builder
        # Reused only for build_code_profile()/align_requirement_to_code()'s
        # own non-LLM heuristics -- never for its bidirectional combination
        # or its code_to_requirement direction.
        self._base = BidirectionalSemanticAligner()

    def build_code_profile(self, *args: Any, **kwargs: Any) -> CodeSemanticProfile:
        return self._base.build_code_profile(*args, **kwargs)

    def align(
        self,
        requirement_profile: RequirementProfile,
        evidence_context: DCAREvidenceContext,
        batch_cache_key: str = "",
    ) -> tuple[AlignmentEvidence, ...]:
        candidates = tuple(evidence_context.candidates)
        code_profiles = tuple(
            self._base.build_code_profile(
                candidate.code_id,
                summary=candidate.summary,
                selected_context=candidate.selected_context,
            )
            for candidate in candidates
        )
        if self.llm_service is None or not self.llm_service.live_api or not candidates:
            return tuple(
                self._align_locally(requirement_profile, code_profile)
                for code_profile in code_profiles
            )

        parsed_by_code_id: dict[str, AlignmentEvidence] = {}
        for start in range(0, len(code_profiles), ALIGNMENT_BATCH_SIZE):
            chunk_candidates = candidates[start : start + ALIGNMENT_BATCH_SIZE]
            chunk_profiles = code_profiles[start : start + ALIGNMENT_BATCH_SIZE]
            expected_ids = {profile.code_id for profile in chunk_profiles}
            chunk_local_fallback = {
                code_profile.code_id: self._align_locally(requirement_profile, code_profile)
                for code_profile in chunk_profiles
            }

            def _validate(
                payload: dict[str, Any],
                expected_ids: set[str] = expected_ids,
                chunk_profiles: tuple[CodeSemanticProfile, ...] = chunk_profiles,
                chunk_local_fallback: dict[str, AlignmentEvidence] = chunk_local_fallback,
            ) -> bool:
                strict_parsed = self._batch_payload(
                    payload, requirement_profile, chunk_profiles,
                    chunk_local_fallback, strict=True,
                )
                return expected_ids.issubset(strict_parsed.keys())

            result = self.llm_service.complete_json(
                module="alignment_r2c_only_batch_chunked_b" + str(ALIGNMENT_BATCH_SIZE),
                prompt=self.prompt_builder.build_batch_prompt(
                    requirement_profile, chunk_candidates, chunk_profiles,
                ),
                requirement_id=requirement_profile.requirement_id,
                code_file=chunk_cache_suffix(
                    batch_cache_key, tuple(p.code_id for p in chunk_profiles)
                ),
                max_completion_tokens=BATCH_ALIGNMENT_MAX_COMPLETION_TOKENS,
                validate=_validate,
                max_attempts=ALIGNMENT_BATCH_MAX_ATTEMPTS,
            )
            chunk_parsed = self._batch_payload(
                result.payload, requirement_profile, chunk_profiles,
                chunk_local_fallback, strict=True,
            )
            for code_profile in chunk_profiles:
                evidence = chunk_parsed.get(code_profile.code_id)
                if evidence is None:
                    raise IncompleteBatchResponseError(
                        f"w/o Bi-Alignment chunk for requirement "
                        f"{requirement_profile.requirement_id} validated the "
                        f"candidate-ID set but produced no usable evidence for "
                        f"{code_profile.code_id}."
                    )
                parsed_by_code_id[code_profile.code_id] = evidence

        return tuple(
            parsed_by_code_id[code_profile.code_id] for code_profile in code_profiles
        )

    def _align_locally(
        self,
        requirement_profile: RequirementProfile,
        code_profile: CodeSemanticProfile,
    ) -> AlignmentEvidence:
        r2c = self._base.align_requirement_to_code(
            requirement_profile, code_profile, use_llm=False
        )
        return AlignmentEvidence(
            requirement_id=requirement_profile.requirement_id,
            code_id=code_profile.code_id,
            requirement_type=requirement_profile.requirement_type,
            code_profile=code_profile,
            requirement_to_code_alignment=r2c,
            code_to_requirement_alignment=REMOVED_CODE_TO_REQUIREMENT_ALIGNMENT,
            matched_elements=r2c.matched_elements,
            alignment_score=r2c.score,
            alignment_explanation=(
                f"R->C only (w/o Bi-Alignment): {r2c.coverage_label}."
            ),
            alpha=1.0,
        )

    def _batch_payload(
        self,
        payload: dict[str, Any],
        requirement_profile: RequirementProfile,
        code_profiles: tuple[CodeSemanticProfile, ...],
        fallback_by_code_id: dict[str, AlignmentEvidence],
        *,
        strict: bool = False,
    ) -> dict[str, AlignmentEvidence]:
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
            evidence = self._from_payload(
                item, requirement_profile, code_profile,
                fallback_by_code_id[code_id], strict=strict,
            )
            if evidence is not None:
                parsed[code_id] = evidence
        return parsed

    def _from_payload(
        self,
        payload: dict[str, Any],
        requirement_profile: RequirementProfile,
        code_profile: CodeSemanticProfile,
        fallback: AlignmentEvidence,
        *,
        strict: bool = False,
    ) -> AlignmentEvidence | None:
        r2c = _directional_from_payload(
            _dict_payload(payload.get("requirement_to_code_alignment")),
            strict=strict,
        )
        if strict and r2c is None:
            return None
        r2c = r2c or fallback.requirement_to_code_alignment
        return AlignmentEvidence(
            requirement_id=requirement_profile.requirement_id,
            code_id=code_profile.code_id,
            requirement_type=requirement_profile.requirement_type,
            code_profile=code_profile,
            requirement_to_code_alignment=r2c,
            code_to_requirement_alignment=REMOVED_CODE_TO_REQUIREMENT_ALIGNMENT,
            matched_elements=r2c.matched_elements,
            # Always the requirement-to-code score directly -- never a
            # model-provided blend, never a function of alpha.
            alignment_score=r2c.score,
            alignment_explanation=(
                f"R->C only (w/o Bi-Alignment): {r2c.coverage_label}."
            ),
            alpha=1.0,
        )


def _dict_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _directional_from_payload(
    payload: dict[str, Any],
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
        direction="requirement_to_code",
        score=score,
        coverage_label=str(payload.get("coverage_label") or _coverage_label(score)),
        matched_elements=_tuple_of_strings(payload.get("matched_elements")),
        missing_elements=_tuple_of_strings(payload.get("missing_elements")),
        rationale=str(payload.get("rationale") or ""),
    )


def _coverage_label(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.4:
        return "partial"
    if score > 0.0:
        return "weak"
    return "none"


def _strict_unit_score(value: Any) -> float | None:
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
