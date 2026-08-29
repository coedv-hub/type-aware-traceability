"""End-to-end IST Proposed Framework pipeline skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from src.retrieval.dcar import DCAREvidenceContext, DynamicContextAdaptiveRetriever
from src.retrieval.summarizer import CachedCodeSummarizer

from .alignment import AlignmentEvidence, BidirectionalSemanticAligner
from .llm_support import FrameworkLLMService
from .understanding import RequirementProfile, TypeAwareRequirementUnderstanding
from .verification import SelfReflectiveVerifier, VerificationEvidence


@dataclass(frozen=True)
class FrameworkPrediction:
    """Final decision for one requirement-code pair."""

    requirement_id: str
    code_id: str
    requirement_type: str
    final_score: float
    alignment_score: float
    verification_score: float
    prediction: str
    explanation: str
    alignment: AlignmentEvidence
    verification: VerificationEvidence

    @property
    def score(self) -> float:
        """Backward-compatible score alias."""
        return self.final_score


@dataclass(frozen=True)
class RequirementRunResult:
    """Pipeline output for one requirement."""

    requirement_profile: RequirementProfile
    evidence_context: DCAREvidenceContext
    alignment_evidence: tuple[AlignmentEvidence, ...]
    verification_evidence: tuple[VerificationEvidence, ...]
    predictions: tuple[FrameworkPrediction, ...]
    placeholder: bool = True


@dataclass(frozen=True)
class FrameworkConfig:
    """Configuration values for the framework skeleton."""

    decision_threshold: float = 0.2
    verification_lambda: float = 0.5


class TypeAwareTraceabilityPipeline:
    """IST pipeline: Understanding -> DCAR -> Alignment -> Verification -> Decision."""

    def __init__(
        self,
        understanding: TypeAwareRequirementUnderstanding | None = None,
        retriever: DynamicContextAdaptiveRetriever | None = None,
        aligner: BidirectionalSemanticAligner | None = None,
        verifier: SelfReflectiveVerifier | None = None,
        config: FrameworkConfig | None = None,
        llm_service: FrameworkLLMService | None = None,
    ):
        self.llm_service = llm_service
        self.understanding = understanding or TypeAwareRequirementUnderstanding(
            llm_service=llm_service
        )
        self.retriever = retriever or DynamicContextAdaptiveRetriever(
            summarizer=CachedCodeSummarizer(llm_service=llm_service)
        )
        self.aligner = aligner or BidirectionalSemanticAligner(
            llm_service=llm_service
        )
        self.verifier = verifier or SelfReflectiveVerifier(
            llm_service=llm_service
        )
        self.config = config or FrameworkConfig()

    def run_requirement(
        self,
        requirement_id: str,
        requirement_text: str,
        requirement_type: str,
        code_corpus: dict[str, str],
        ranked_candidates: dict[str, tuple[int, float]] | None = None,
        batch_cache_key: str = "",
    ) -> RequirementRunResult:
        """Run the skeleton pipeline for one requirement."""
        requirement_profile = self.understanding.understand(
            requirement_text=requirement_text,
            requirement_type=requirement_type,
            requirement_id=requirement_id,
        )
        evidence_context = self.retriever.retrieve(
            requirement_profile=requirement_profile,
            code_corpus=code_corpus,
            ranked_candidates=ranked_candidates,
        )
        alignment_evidence = self.aligner.align(
            requirement_profile=requirement_profile,
            evidence_context=evidence_context,
            batch_cache_key=batch_cache_key,
        )
        verification_evidence = self.verifier.verify_batch(
            requirement_profile,
            alignment_evidence,
            batch_cache_key=batch_cache_key,
        )
        predictions = tuple(
            self.predict_pair(
                requirement_profile=requirement_profile,
                alignment=evidence,
                verification=verification,
            )
            for evidence, verification in zip(
                alignment_evidence, verification_evidence, strict=True
            )
        )
        return RequirementRunResult(
            requirement_profile=requirement_profile,
            evidence_context=evidence_context,
            alignment_evidence=alignment_evidence,
            verification_evidence=verification_evidence,
            predictions=predictions,
            placeholder=True,
        )

    def predict_pair(
        self,
        requirement_profile: RequirementProfile,
        alignment: AlignmentEvidence,
        verification: VerificationEvidence,
    ) -> FrameworkPrediction:
        """Combine alignment and verification evidence into a final decision."""
        final_score = alignment.alignment_score * (
            self.config.verification_lambda
            + (1.0 - self.config.verification_lambda)
            * verification.verification_score
        )
        prediction = (
            "Yes" if final_score >= self.config.decision_threshold else "No"
        )
        return FrameworkPrediction(
            requirement_id=requirement_profile.requirement_id,
            code_id=alignment.code_id,
            requirement_type=requirement_profile.requirement_type,
            final_score=final_score,
            alignment_score=alignment.alignment_score,
            verification_score=verification.verification_score,
            prediction=prediction,
            explanation=(
                f"Final score combines alignment ({alignment.alignment_score:.3f}) "
                f"and verification ({verification.verification_score:.3f}); "
                f"threshold={self.config.decision_threshold:.3f}."
            ),
            alignment=alignment,
            verification=verification,
        )
