"""evidence_anchored_v3 = evidence_anchored_v2 + dependency_context.

Isolated experiment: does not modify v1 (AlignmentPromptBuilder/
VerificationPromptBuilder), v2 (EvidenceAnchoredAlignmentPromptBuilder/
EvidenceAnchoredVerificationPromptBuilder), DCAR, any raw Data/ file, or
configs/final_framework_config.yaml. Drop-in replacements (identical
method signatures and JSON response schema to v1/v2, so the existing
_batch_alignment_payload/_batch_verification_payload parsers work
unchanged) for BidirectionalSemanticAligner(prompt_builder=...) /
SelfReflectiveVerifier(prompt_builder=...).

Adds a per-candidate "dependency_context" block (up to
max_dependencies_shown neighboring files this candidate has a verifiable
import/class-usage (Java) or function-call (C) dependency on -- see
src/framework/dependency_context.py) plus an explicit instruction that a
candidate not directly covering the requirement, but calling/depending on
a listed file that clearly does, counts as valid indirect evidence -- to
recover real links whose surface text alone doesn't match the requirement
(a "false negative" pattern, e.g. Alignment=none/Verification=none
despite being a genuine gold link)."""

from __future__ import annotations

from .alignment import AlignmentEvidence, CodeSemanticProfile
from .evidence_anchored_prompts import ALIGNMENT_ANCHOR_RUBRIC, VERIFICATION_ANCHOR_RUBRIC
from .understanding import RequirementProfile
from src.retrieval.filters import FilteredCandidate
from src.retrieval.summarizer import CodeSummary

DEPENDENCY_CONTEXT_INSTRUCTIONS = """Dependency context (apply strictly):
Each candidate may list a small number of OTHER files it has a verifiable
import/class-usage (Java) or function-call (C) dependency on, with a
one-line description of each. This is real, machine-extracted structural
context, not a hint about the correct answer.
- If the candidate's OWN code does not directly cover the requirement,
  but it calls/depends on a listed dependency file whose description
  clearly implements the requirement, that counts as valid INDIRECT
  evidence -- name the specific dependency file you relied on in your
  rationale/explanation.
- A dependency listed with no clear relevance to the requirement is not
  evidence of anything; do not inflate a score just because a dependency
  list is present.
- Indirect (dependency-based) evidence should be labeled as such in the
  rationale -- do not present it as if it were direct evidence."""


def _dependency_blocks(
    code_id: str,
    dependency_graph: dict[str, set[str]],
    code_summaries: dict[str, CodeSummary],
    max_dependencies_shown: int,
) -> list[dict[str, str]]:
    neighbor_ids = sorted(dependency_graph.get(code_id, set()))[:max_dependencies_shown]
    blocks = []
    for neighbor_id in neighbor_ids:
        summary = code_summaries.get(neighbor_id)
        blocks.append(
            {
                "code_id": neighbor_id,
                "description": summary.description if summary else "",
            }
        )
    return blocks


class DependencyContextAlignmentPromptBuilder:
    """v3 alignment prompt: v2's schema and score anchors, plus a
    per-candidate dependency_context block."""

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
        return f"""IST bidirectional semantic alignment batch (evidence-anchored v3: v2 + dependency_context).
Return only valid JSON.
Use concise phrases: max 3 matched_elements, max 3 missing_elements, rationale <= 12 words.
{ALIGNMENT_ANCHOR_RUBRIC}
{DEPENDENCY_CONTEXT_INSTRUCTIONS}
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


class DependencyContextVerificationPromptBuilder:
    """v3 verification prompt: v2's schema and score anchors, plus a
    per-candidate dependency_context block. Verification only ever
    receives this batch's AlignmentEvidence (not the original candidates),
    so the dependency graph/summaries are looked up again here by code_id
    rather than reusing the alignment builder's per-call data."""

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
                "dependency_context": _dependency_blocks(
                    alignment.code_id, self.dependency_graph,
                    self.code_summaries, self.max_dependencies_shown,
                ),
            }
            for alignment in alignments
        ]
        return f"""IST self-reflective verification batch (evidence-anchored v3: v2 + dependency_context).
Evaluate semantic consistency, evidence sufficiency, and evidence grounding for each candidate.
Return only valid JSON.
Use concise phrases: max 3 supporting_terms, max 3 missing_terms, rationale <= 12 words.
{VERIFICATION_ANCHOR_RUBRIC}
{DEPENDENCY_CONTEXT_INSTRUCTIONS}
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
