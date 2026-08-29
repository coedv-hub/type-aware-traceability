"""Baseline methods used by the IST experiments."""

from .ir import BM25Baseline, IRBaseline, LSIBaseline, TFIDFBaseline, create_baseline
from .llm import (
    DirectPromptingBaseline,
    GenericRAGLLMBaseline,
    GenericRetriever,
    LLMBaseline,
    PairResult,
    Prediction,
    create_llm_baseline,
    parse_prediction,
)
from .neural import (
    CacheStats,
    CodeBERTBaseline,
    EmbeddingCache,
    NeuralBaseline,
    SentenceBERTBaseline,
    create_neural_baseline,
    select_device,
)

__all__ = [
    "BM25Baseline",
    "CacheStats",
    "CodeBERTBaseline",
    "DirectPromptingBaseline",
    "EmbeddingCache",
    "GenericRAGLLMBaseline",
    "GenericRetriever",
    "IRBaseline",
    "LLMBaseline",
    "LSIBaseline",
    "NeuralBaseline",
    "PairResult",
    "Prediction",
    "SentenceBERTBaseline",
    "TFIDFBaseline",
    "create_baseline",
    "create_llm_baseline",
    "create_neural_baseline",
    "parse_prediction",
    "select_device",
]
