"""w/o Type Understanding ablation: a genuinely type-neutral pipeline.

Isolated experiment for the 1:1 pair-classification track only (see
scripts/run_pair_classification.py's ABLATIONS table). Does not modify
understanding.py's TypeAwareRequirementUnderstanding/
TypeAwareUnderstandingPromptBuilder, alignment.py's/verification.py's
FROZEN AlignmentPromptBuilder/VerificationPromptBuilder or DCAR's FR/NFR/
Mixed SemanticFilter classes (all used unchanged by v1/v2/v3 and the
formal top-k Proposed Framework runs).

Fixed version (superseding an earlier, LEAKY implementation that reused
FRRequirementProfile as a container -- see the audit that found it): this
module now defines its OWN, genuinely distinct GenericRequirementProfile
(a sibling of FR/NFR/Mixed, never an instance of any of them), and its OWN
alignment/verification prompt builders that never print
"Requirement type: ..." and never interpolate the profile object's repr
(whose dataclass-generated str() would include requirement_type) -- only
the explicit summary/key_terms/expected_behavior fields ever reach a
prompt. DCAR's SemanticFilter gained a real, separate generic_filter
branch (src/retrieval/filters.py's GenericSemanticFilter, additive, not
touching the FR/NFR/Mixed branches) so this profile is scored through a
type-neutral path, not FR's; DCAR's own retrieval_budget_for() already had
a pre-existing generic fallback (multiplier=1, strategy="generic") that
this profile now correctly reaches instead of always taking the FR path.

Cache version bump: GENERIC_UNDERSTANDING_MODULE changed from
"understanding_generic" to "understanding_type_neutral" specifically so
this fixed implementation can never accidentally reuse the earlier, leaky
version's cached responses -- every cache identity here is guaranteed
fresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dependency_context_prompts import DEPENDENCY_CONTEXT_INSTRUCTIONS, _dependency_blocks
from .evidence_anchored_prompts import ALIGNMENT_ANCHOR_RUBRIC, VERIFICATION_ANCHOR_RUBRIC
from .llm_support import FrameworkLLMService
from .understanding import TypeAwareRequirementProfile
from src.retrieval.filters import FilteredCandidate
from src.retrieval.summarizer import CodeSummary

GENERIC_UNDERSTANDING_MODULE = "understanding_type_neutral"

# Deliberately NOT one of "FR"/"NFR"/"Mixed" -- never equal to any gold
# type label, and (per this module's design) never interpolated into any
# prompt anyway. Kept only so the field has a harmless, clearly-neutral
# value if ever inspected/logged.
GENERIC_REQUIREMENT_TYPE_MARKER = "Generic"


@dataclass(frozen=True)
class GenericRequirementProfile(TypeAwareRequirementProfile):
    """Type-neutral requirement profile. NOT FRRequirementProfile,
    NFRRequirementProfile, or MixedRequirementProfile -- isinstance checks
    against those three (DCAR's SemanticFilter.score_relevance,
    alignment.py's/verification.py's _requirement_terms) all correctly
    miss this class, so it never takes any type-specific code path.

    _requirement_terms()'s catch-all (`return _tokens(profile.raw_text)`)
    already handles any unrecognized profile type without modification,
    and DCAR's retrieval_budget_for()/analyze_complexity() already fall
    back to a generic/non-Mixed path for any non-FR/NFR/Mixed profile --
    only SemanticFilter.score_relevance() needed a new branch (added in
    src/retrieval/filters.py as GenericSemanticFilter), since it had no
    fallback and raised TypeError otherwise."""

    summary: str = ""
    key_terms: tuple[str, ...] = ()
    expected_behavior: str = ""


def _profile_text_block(profile: GenericRequirementProfile) -> str:
    """Explicit, controlled textual representation of a generic profile --
    built field-by-field, NEVER via str(profile)/repr(profile) (whose
    dataclass-generated form would include requirement_id/requirement_type/
    prompt/placeholder). Only summary/key_terms/expected_behavior -- the
    three fields this ablation is defined to keep -- ever reach a prompt."""
    lines = [f"Summary: {profile.summary}"]
    if profile.key_terms:
        lines.append("Key terms: " + ", ".join(profile.key_terms))
    lines.append(f"Expected behavior: {profile.expected_behavior}")
    return "\n".join(lines)


class GenericUnderstandingPromptBuilder:
    """Build one uniform prompt regardless of requirement type."""

    def build_prompt(self, requirement_text: str) -> str:
        return f"""IST generic requirement understanding (w/o Type Understanding ablation, type-neutral).
Return only valid JSON. Do not tailor the answer to FR, NFR, or Mixed -- use
the same generic questions for every requirement.
Requirement: {requirement_text}
Schema:
{{
  "summary": "concise, type-agnostic restatement of what this requirement asks for",
  "key_terms": ["important domain terms, entities, or constraints mentioned"],
  "expected_behavior": "observable behavior or property expected from the system/code"
}}"""


class GenericRequirementUnderstanding:
    """Drop-in replacement for TypeAwareRequirementUnderstanding: same
    constructor/`.understand()` signature, so it can be passed directly as
    scripts/run_proposed_framework.py's run_framework(understanding=...).

    `.understand()` accepts `requirement_type` for signature compatibility
    but never stores, branches on, or forwards it anywhere -- the returned
    profile's requirement_type is always GENERIC_REQUIREMENT_TYPE_MARKER,
    never the true gold FR/NFR/Mixed label."""

    CACHE_MODULE = GENERIC_UNDERSTANDING_MODULE

    def __init__(
        self,
        prompt_builder: GenericUnderstandingPromptBuilder | None = None,
        llm_service: FrameworkLLMService | None = None,
    ):
        self.prompt_builder = prompt_builder or GenericUnderstandingPromptBuilder()
        self.llm_service = llm_service

    def understand(
        self,
        requirement_text: str,
        requirement_type: str,  # noqa: ARG002 -- intentionally ignored, see class docstring
        requirement_id: str = "",
    ) -> GenericRequirementProfile:
        prompt = self.prompt_builder.build_prompt(requirement_text)
        payload: dict[str, Any] = {}
        if self.llm_service is not None and self.llm_service.live_api:
            result = self.llm_service.complete_json(
                module=self.CACHE_MODULE,
                prompt=prompt,
                requirement_id=requirement_id,
            )
            payload = result.payload
        text = requirement_text.strip()
        summary = str(payload.get("summary") or text)
        key_terms = _tuple_of_strings(payload.get("key_terms"))
        expected_behavior = str(payload.get("expected_behavior") or text)
        raw_text = " ".join(
            part for part in (summary, " ".join(key_terms), expected_behavior) if part
        )
        return GenericRequirementProfile(
            requirement_id=requirement_id,
            requirement_type=GENERIC_REQUIREMENT_TYPE_MARKER,
            raw_text=raw_text or text,
            prompt=prompt,
            placeholder=False,
            summary=summary,
            key_terms=key_terms,
            expected_behavior=expected_behavior,
        )


class TypeNeutralAlignmentPromptBuilder:
    """w/o Type Understanding (fixed): same evidence-anchored +
    dependency-context mechanism as v3's DependencyContextAlignmentPromptBuilder
    (this ablation keeps "the rest of v3" unchanged), but NEVER prints
    "Requirement type: ..." and NEVER interpolates the whole profile object
    (whose repr would include requirement_type) -- only
    _profile_text_block()'s explicit summary/key_terms/expected_behavior
    text reaches the prompt."""

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
        requirement_profile: GenericRequirementProfile,
        candidates: tuple[FilteredCandidate, ...],
        code_profiles: tuple[Any, ...],
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
        return f"""IST bidirectional semantic alignment batch (w/o Type Understanding ablation: type-neutral, v3 + dependency_context).
Return only valid JSON.
Use concise phrases: max 3 matched_elements, max 3 missing_elements, rationale <= 12 words.
{ALIGNMENT_ANCHOR_RUBRIC}
{DEPENDENCY_CONTEXT_INSTRUCTIONS}
Requirement profile:
{_profile_text_block(requirement_profile)}
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
      }},
      "code_to_requirement_alignment": {{
        "score": 0.0,
        "coverage_label": "strong|partial|weak|none",
        "matched_elements": ["code evidence that supports the requirement -- name the specific entity/field/rule/API, or the specific dependency file, not just that overlap exists"],
        "missing_elements": ["code-side evidence not supported by requirement"],
        "rationale": "brief reverse-alignment rationale, must state whether the match is requirement-specific, dependency-based (name the dependency), or only generic overlap"
      }},
      "matched_elements": ["combined matched elements"],
      "functional_alignment_score": 0.0,
      "quality_alignment_score": 0.0,
      "alignment_score": 0.0,
      "alignment_explanation": "brief bidirectional explanation, must state whether the match is requirement-specific, dependency-based (name the dependency), or only generic overlap"
    }}
  ]
}}"""


class TypeNeutralVerificationPromptBuilder:
    """w/o Type Understanding (fixed): same evidence-anchored +
    dependency-context verification mechanism as v3's
    DependencyContextVerificationPromptBuilder, but never prints the
    requirement type or the profile's repr -- see
    TypeNeutralAlignmentPromptBuilder's docstring for the same rationale."""

    def __init__(
        self,
        dependency_graph: dict[str, set[str]],
        code_summaries: dict[str, CodeSummary],
        max_dependencies_shown: int = 3,
    ):
        self.dependency_graph = dependency_graph
        self.code_summaries = code_summaries
        self.max_dependencies_shown = max_dependencies_shown

    def build_requirement_batch_prompt(
        self,
        requirement_profile: GenericRequirementProfile,
        alignments: tuple[Any, ...],
    ) -> str:
        items = [
            {
                "code_id": alignment.code_id,
                "alignment_score": alignment.alignment_score,
                "matched_elements": alignment.matched_elements,
                "alignment_explanation": alignment.alignment_explanation,
                "requirement_to_code_alignment": alignment.requirement_to_code_alignment,
                "code_to_requirement_alignment": alignment.code_to_requirement_alignment,
                "dependency_context": _dependency_blocks(
                    alignment.code_id, self.dependency_graph,
                    self.code_summaries, self.max_dependencies_shown,
                ),
            }
            for alignment in alignments
        ]
        return f"""IST self-reflective verification batch (w/o Type Understanding ablation: type-neutral, v3 + dependency_context).
Evaluate semantic consistency, evidence sufficiency, and evidence grounding for each candidate.
Return only valid JSON.
Use concise phrases: max 3 supporting_terms, max 3 missing_terms, rationale <= 12 words.
{VERIFICATION_ANCHOR_RUBRIC}
{DEPENDENCY_CONTEXT_INSTRUCTIONS}
Requirement profile:
{_profile_text_block(requirement_profile)}
Alignment evidence by candidate: {items}
Schema:
{{
  "candidates": [
    {{
      "code_id": "candidate id",
      "semantic_consistency": {{
        "score": 0.0,
        "label": "strong|moderate|weak|none",
        "rationale": "brief verification rationale, must state whether supporting terms are requirement-specific, dependency-based (name the dependency), or only generic overlap",
        "supporting_terms": ["grounded, requirement-specific or named-dependency supporting terms -- not generic/stopword overlap"],
        "missing_terms": ["missing or unsupported terms"]
      }},
      "evidence_sufficiency": {{
        "score": 0.0,
        "label": "strong|moderate|weak|none",
        "rationale": "brief verification rationale, must state whether supporting terms are requirement-specific, dependency-based (name the dependency), or only generic overlap",
        "supporting_terms": ["grounded, requirement-specific or named-dependency supporting terms -- not generic/stopword overlap"],
        "missing_terms": ["missing or unsupported terms"]
      }},
      "evidence_grounding": {{
        "score": 0.0,
        "label": "strong|moderate|weak|none",
        "rationale": "brief verification rationale, must state whether supporting terms are requirement-specific, dependency-based (name the dependency), or only generic overlap",
        "supporting_terms": ["grounded, requirement-specific or named-dependency supporting terms -- not generic/stopword overlap"],
        "missing_terms": ["missing or unsupported terms"]
      }},
      "functional_verification_score": 0.0,
      "quality_verification_score": 0.0,
      "verification_score": 0.0,
      "verification_explanation": "brief combined verification explanation, must state whether evidence is requirement-specific, dependency-based (name the dependency), or only generic overlap",
      "supported": false
    }}
  ]
}}"""


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, tuple):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()
