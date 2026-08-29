"""w/o Verification ablation: skip self-reflective verification entirely.

Isolated experiment for the 1:1 pair-classification track only (see
scripts/run_pair_classification.py's ABLATIONS table). Does not modify
verification.py's SelfReflectiveVerifier/VerificationPromptBuilder (used
unchanged by v1/v2/v3 and the formal top-k Proposed Framework runs).

NullVerifier.verify_batch() never calls self.llm_service (there is none --
it does not even accept one) and never touches the LLM cache: it returns a
neutral VerificationEvidence(verification_score=1.0) for every alignment.
Two independent guarantees follow from that:

1. No verification-category LLM call is ever made, live or cached
   (verifiable directly: NullVerifier holds no client/cache reference at
   all).
2. TypeAwareTraceabilityPipeline.predict_pair's formula
   `alignment_score * (lambda + (1-lambda) * verification_score)` collapses
   to exactly `alignment_score` for ANY lambda, since
   lambda + (1-lambda)*1 == 1 algebraically.

scripts/run_pair_classification.py's ablation runner additionally forces
Final_Score = Alignment_Score explicitly after run_framework() returns (and
skips the frozen scoring-variant B recombination applied to the other
ablations), so the "use alignment score as the final score" requirement
does not rely on point (2) alone.
"""

from __future__ import annotations

from .alignment import AlignmentEvidence
from .understanding import RequirementProfile
from .verification import VerificationDimensionResult, VerificationEvidence

_SKIPPED_DIMENSION = VerificationDimensionResult(
    score=1.0,
    label="skipped",
    rationale="Verification skipped (w/o Verification ablation).",
)


class NullVerifier:
    """Duck-types SelfReflectiveVerifier's verify_batch() signature so it
    can be passed directly as run_framework(verifier=...) / TypeAware
    TraceabilityPipeline(verifier=...) without any change to either."""

    def verify_batch(
        self,
        requirement_profile: RequirementProfile,
        alignments: tuple[AlignmentEvidence, ...],
        *,
        batch_cache_key: str = "",
    ) -> tuple[VerificationEvidence, ...]:
        return tuple(
            VerificationEvidence(
                requirement_id=requirement_profile.requirement_id,
                code_id=alignment.code_id,
                semantic_consistency=_SKIPPED_DIMENSION,
                evidence_sufficiency=_SKIPPED_DIMENSION,
                evidence_grounding=_SKIPPED_DIMENSION,
                verification_score=1.0,
                verification_explanation=(
                    "Verification skipped (w/o Verification ablation); "
                    "final score uses the alignment score directly."
                ),
                supported=True,
            )
            for alignment in alignments
        )
