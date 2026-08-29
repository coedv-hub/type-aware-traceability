"""Proposed framework components for Type-aware traceability."""

from .alignment import (
    AlignmentEvidence,
    AlignmentPromptBuilder,
    AlignmentResult,
    BidirectionalAlignmentResult,
    BidirectionalSemanticAligner,
    CodeSemanticProfile,
    CodeProfile,
    DirectionalAlignment,
)
from .llm_support import FrameworkLLMResult, FrameworkLLMService
from .fusion import DEFAULT_MIXED_FUSION, FixedMixedSignalFusion
from .pipeline import (
    FrameworkConfig,
    FrameworkPrediction,
    RequirementRunResult,
    TypeAwareTraceabilityPipeline,
)
from .understanding import (
    FRRequirementProfile,
    FRProfile,
    MixedRequirementProfile,
    MixedProfile,
    NFRRequirementProfile,
    NFRProfile,
    RequirementProfile,
    RequirementType,
    TypeAwareRequirementProfile,
    TypeAwareRequirementUnderstanding,
    TypeAwareUnderstandingPromptBuilder,
)
from .verification import (
    SelfReflectiveVerifier,
    VerificationDimensionResult,
    VerificationEvidence,
    VerificationPromptBuilder,
    VerificationResult,
)

__all__ = [
    "AlignmentResult",
    "AlignmentEvidence",
    "AlignmentPromptBuilder",
    "BidirectionalAlignmentResult",
    "BidirectionalSemanticAligner",
    "CodeSemanticProfile",
    "CodeProfile",
    "DirectionalAlignment",
    "FRRequirementProfile",
    "FRProfile",
    "FixedMixedSignalFusion",
    "FrameworkConfig",
    "FrameworkLLMResult",
    "FrameworkLLMService",
    "FrameworkPrediction",
    "MixedRequirementProfile",
    "MixedProfile",
    "NFRRequirementProfile",
    "NFRProfile",
    "RequirementProfile",
    "RequirementRunResult",
    "RequirementType",
    "SelfReflectiveVerifier",
    "TypeAwareRequirementProfile",
    "TypeAwareRequirementUnderstanding",
    "TypeAwareUnderstandingPromptBuilder",
    "TypeAwareTraceabilityPipeline",
    "DEFAULT_MIXED_FUSION",
    "VerificationDimensionResult",
    "VerificationEvidence",
    "VerificationPromptBuilder",
    "VerificationResult",
]
