"""Evidence-anchored (v2) alignment/verification prompt variant.

Isolated experiment, not part of the frozen end-to-end Proposed Framework:
does not modify AlignmentPromptBuilder/VerificationPromptBuilder (the
frozen prompts every formal top-k Proposed Framework run uses), DCAR, any
dataset, any global threshold, or configs/final_framework_config.yaml.
Drop-in replacements (identical method signatures and JSON response
schema, so the existing _batch_alignment_payload/_batch_verification_payload
parsers work unchanged) for BidirectionalSemanticAligner(prompt_builder=...)
/ SelfReflectiveVerifier(prompt_builder=...).

Motivated by an error analysis of v1: v1's highest-scoring false
positives are driven almost entirely by generic
lexical overlap (stopwords, generic domain nouns) with no requirement to
show the evidence is requirement-SPECIFIC, while the verification stage
still labels these "strong" grounding. v2 adds explicit numeric score
anchors and requires requirement-specific entities/fields/business rules/
methods/API calls or invocation evidence for a high score; generic
functional or lexical similarity alone must not score high.
"""

from __future__ import annotations

from typing import Any

from .alignment import AlignmentEvidence, CodeSemanticProfile
from .understanding import RequirementProfile
from src.retrieval.filters import FilteredCandidate

ALIGNMENT_ANCHOR_RUBRIC = """Score anchors (apply strictly):
- 0.8-1.0: The code contains requirement-SPECIFIC evidence -- a named
  entity, field, business rule, method/API, or function call that is
  unique to THIS requirement (not something that would equally match many
  other requirement-code pairs in this project).
- 0.4-0.7: Some requirement-specific evidence is present, but partial or
  missing key elements the requirement calls for.
- 0.0-0.3: Only generic or common terms match (stopwords, generic domain
  nouns like "data"/"database"/"handler"/"manager", or terms that would
  match many unrelated requirement-code pairs in this project). Generic
  functional or lexical similarity alone MUST NOT score above 0.3, even if
  many words overlap."""

VERIFICATION_ANCHOR_RUBRIC = """Score anchors (apply strictly):
- 0.8-1.0: Requirement-specific evidence (a named entity, field, business
  rule, or method/API call unique to this requirement) is present in the
  code and directly supports the link.
- 0.4-0.7: Some requirement-specific evidence is present but incomplete or
  only indirectly supports the link.
- 0.0-0.3: Only generic overlap (stopwords, generic domain nouns, or terms
  common across many requirement-code pairs in this project). This must
  NOT be scored as strong grounding or sufficiency even if several such
  generic terms match."""


class EvidenceAnchoredAlignmentPromptBuilder:
    """v2 alignment prompt: same JSON schema as AlignmentPromptBuilder,
    with explicit evidence-specificity score anchors added."""

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
        return f"""IST bidirectional semantic alignment batch (evidence-anchored v2).
Return only valid JSON.
Use concise phrases: max 3 matched_elements, max 3 missing_elements, rationale <= 12 words.
{ALIGNMENT_ANCHOR_RUBRIC}
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
        "matched_elements": ["requirement-specific terms or evidence that match -- name the specific entity/field/rule/API, not just that overlap exists"],
        "missing_elements": ["important requirement elements not covered"],
        "rationale": "brief alignment rationale, must state whether the match is requirement-specific or only generic overlap"
      }},
      "code_to_requirement_alignment": {{
        "score": 0.0,
        "coverage_label": "strong|partial|weak|none",
        "matched_elements": ["code evidence that supports the requirement -- name the specific entity/field/rule/API, not just that overlap exists"],
        "missing_elements": ["code-side evidence not supported by requirement"],
        "rationale": "brief reverse-alignment rationale, must state whether the match is requirement-specific or only generic overlap"
      }},
      "matched_elements": ["combined matched elements"],
      "functional_alignment_score": 0.0,
      "quality_alignment_score": 0.0,
      "alignment_score": 0.0,
      "alignment_explanation": "brief bidirectional explanation, must state whether the match is requirement-specific or only generic overlap"
    }}
  ]
}}"""


class EvidenceAnchoredVerificationPromptBuilder:
    """v2 verification prompt: same JSON schema as VerificationPromptBuilder,
    with explicit evidence-specificity score anchors added."""

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
        return f"""IST self-reflective verification batch (evidence-anchored v2).
Evaluate semantic consistency, evidence sufficiency, and evidence grounding for each candidate.
Return only valid JSON.
Use concise phrases: max 3 supporting_terms, max 3 missing_terms, rationale <= 12 words.
{VERIFICATION_ANCHOR_RUBRIC}
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
        "rationale": "brief verification rationale, must state whether supporting terms are requirement-specific or only generic overlap",
        "supporting_terms": ["grounded, requirement-specific supporting terms -- not generic/stopword overlap"],
        "missing_terms": ["missing or unsupported terms"]
      }},
      "evidence_sufficiency": {{
        "score": 0.0,
        "label": "strong|moderate|weak|none",
        "rationale": "brief verification rationale, must state whether supporting terms are requirement-specific or only generic overlap",
        "supporting_terms": ["grounded, requirement-specific supporting terms -- not generic/stopword overlap"],
        "missing_terms": ["missing or unsupported terms"]
      }},
      "evidence_grounding": {{
        "score": 0.0,
        "label": "strong|moderate|weak|none",
        "rationale": "brief verification rationale, must state whether supporting terms are requirement-specific or only generic overlap",
        "supporting_terms": ["grounded, requirement-specific supporting terms -- not generic/stopword overlap"],
        "missing_terms": ["missing or unsupported terms"]
      }},
      "functional_verification_score": 0.0,
      "quality_verification_score": 0.0,
      "verification_score": 0.0,
      "verification_explanation": "brief combined verification explanation, must state whether evidence is requirement-specific or only generic overlap",
      "supported": false
    }}
  ]
}}"""
