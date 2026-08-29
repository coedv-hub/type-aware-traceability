"""Direct Prompting and Generic RAG-LLM baselines for Stage 4a."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from .ir import TFIDFBaseline
from ..data.loader import TraceDataset
from ..llm.base_client import LLMClient, LLMResponse
from ..llm.cache import CacheIdentity, LLMCache
from ..llm.prompt_templates import direct_prompt, rag_prompt


@dataclass(frozen=True)
class Prediction:
    prediction: str
    confidence: float | None
    explanation: str
    parse_error: str = ""


@dataclass(frozen=True)
class PairResult:
    prediction: Prediction
    response: LLMResponse
    prompt: str
    cache_hit: bool
    cache_write_verified: bool


def parse_prediction(text: str) -> Prediction:
    """Parse the promised JSON while preserving malformed provider output."""
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        payload = json.loads(candidate)
        raw_prediction = str(payload["prediction"]).strip().casefold()
        lookup = {"yes": "Yes", "no": "No", "unknown": "Unknown"}
        if raw_prediction not in lookup:
            raise ValueError("prediction must be Yes or No")
        confidence_value = payload.get("confidence")
        confidence = (
            None if confidence_value is None else float(confidence_value)
        )
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return Prediction(
            prediction=lookup[raw_prediction],
            confidence=confidence,
            explanation=str(payload.get("explanation") or "").strip(),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return Prediction(
            prediction="Unknown",
            confidence=None,
            explanation=candidate[:1000],
            parse_error=f"{type(exc).__name__}: {exc}",
        )


class GenericRetriever:
    """Simple TF-IDF retriever that ranks every candidate code artifact."""

    def __init__(self, dataset: TraceDataset):
        self.dataset = dataset
        self.code_ids = list(dataset.code_files)
        self.baseline = TFIDFBaseline()
        self.baseline.fit(
            self.code_ids,
            [dataset.code_files[code_id].text for code_id in self.code_ids],
        )

    def retrieve(self, requirement_text: str) -> dict[str, tuple[int, float]]:
        scores: np.ndarray = self.baseline.score_all([requirement_text])[0]
        ranked = sorted(
            zip(self.code_ids, scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
        return {
            code_id: (rank, float(score))
            for rank, (code_id, score) in enumerate(ranked, start=1)
        }


class LLMBaseline:
    name = "base"

    def __init__(
        self,
        client: LLMClient,
        cache: LLMCache,
        dataset: TraceDataset,
        estimate_only: bool,
        candidates: dict[str, dict[str, tuple[int, float]]] | None = None,
    ):
        self.client = client
        self.cache = cache
        self.dataset = dataset
        self.estimate_only = estimate_only
        self.candidates = candidates

    def candidates_for(self, requirement_id: str) -> dict[str, tuple[int, float]]:
        if self.candidates is None:
            return {
                code_id: (rank, 0.0)
                for rank, code_id in enumerate(self.dataset.code_files, start=1)
            }
        return self.candidates[requirement_id]

    def build_prompts(self, requirement_id: str) -> dict[str, str]:
        raise NotImplementedError

    def run_pair(
        self, requirement_id: str, code_file: str, prompt: str
    ) -> PairResult:
        identity = CacheIdentity(
            model=self.client.model,
            prompt=prompt,
            dataset=self.dataset.name,
            requirement_id=requirement_id,
            code_file=code_file,
            provider=self.client.provider,
        )
        # A dry run's placeholder is never a substitute for a real
        # response, in either direction: it must not count as a hit (that
        # would silently mask a pair that still needs a real call), and a
        # dry run must never persist it to the shared, real cache (that
        # would leave every future run -- dry or live -- that shares this
        # identity permanently misled by a fake answer). allow_estimated is
        # therefore always False here, independent of self.estimate_only.
        cached = self.cache.get(identity, allow_estimated=False)
        if cached is not None:
            return PairResult(
                prediction=parse_prediction(cached.text),
                response=cached,
                prompt=prompt,
                cache_hit=True,
                cache_write_verified=True,
            )

        if self.estimate_only:
            response = self.client.estimated_response(prompt)
            return PairResult(
                prediction=parse_prediction(response.text),
                response=response,
                prompt=prompt,
                cache_hit=False,
                # Nothing was written -- a dry run has nothing to verify,
                # which is not the same as a failed write.
                cache_write_verified=True,
            )

        response = self.client.complete(prompt)
        self.cache.put(identity, response)
        verified = self.cache.get(identity, allow_estimated=False)
        return PairResult(
            prediction=parse_prediction(response.text),
            response=response,
            prompt=prompt,
            cache_hit=False,
            cache_write_verified=verified == response,
        )


class DirectPromptingBaseline(LLMBaseline):
    name = "direct"

    def build_prompts(self, requirement_id: str) -> dict[str, str]:
        requirement = self.dataset.requirements[requirement_id]
        return {
            code_id: direct_prompt(
                requirement.text, code_id, self.dataset.code_files[code_id].text
            )
            for code_id in self.candidates_for(requirement_id)
        }


class GenericRAGLLMBaseline(LLMBaseline):
    name = "rag"

    def __init__(
        self,
        client: LLMClient,
        cache: LLMCache,
        dataset: TraceDataset,
        estimate_only: bool,
        candidates: dict[str, dict[str, tuple[int, float]]] | None = None,
    ):
        super().__init__(client, cache, dataset, estimate_only, candidates)
        self.retriever = GenericRetriever(dataset) if candidates is None else None

    def build_prompts(self, requirement_id: str) -> dict[str, str]:
        requirement = self.dataset.requirements[requirement_id]
        retrieved = (
            self.retriever.retrieve(requirement.text)
            if self.retriever is not None
            else self.candidates_for(requirement_id)
        )
        return {
            code_id: rag_prompt(
                requirement=requirement.text,
                code_file=code_id,
                code=self.dataset.code_files[code_id].text,
                retrieval_rank=retrieved[code_id][0],
                retrieval_score=retrieved[code_id][1],
            )
            for code_id in retrieved
        }


def create_llm_baseline(
    method: str,
    client: LLMClient,
    cache: LLMCache,
    dataset: TraceDataset,
    estimate_only: bool,
    candidates: dict[str, dict[str, tuple[int, float]]] | None = None,
) -> LLMBaseline:
    if method == "direct":
        return DirectPromptingBaseline(
            client, cache, dataset, estimate_only, candidates
        )
    if method == "rag":
        return GenericRAGLLMBaseline(
            client, cache, dataset, estimate_only, candidates
        )
    raise ValueError(f"Unknown LLM baseline: {method}")
