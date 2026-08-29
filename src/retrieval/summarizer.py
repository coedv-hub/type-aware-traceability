"""Cached code summarization interfaces for DCAR."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from src.framework.evidence_cues import EVIDENCE_CUE_TERMS
from src.framework.llm_support import FrameworkLLMService


_FUNCTION_PATTERN = re.compile(
    r"\b(?:def|function|void|int|boolean|bool|String|char|float|double)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_CLASS_PATTERN = re.compile(
    r"\b(?:class|interface|enum|struct)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_API_CALL_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_QUALITY_TERMS = (
    "auth",
    "encrypt",
    "permission",
    "validate",
    "timeout",
    "cache",
    "retry",
    "exception",
    "error",
    "log",
    "lock",
    "thread",
    "performance",
    "security",
    *EVIDENCE_CUE_TERMS,
)
SUMMARY_CACHE_MODULE = "dcar_summarizer"
SUMMARY_CACHE_PROMPT = "persistent_code_summary_v1"
SUMMARY_CACHE_REQUIREMENT_ID = "code_summary"


@dataclass(frozen=True)
class CodeSummary:
    """Reusable code summary described in the IST methodology."""

    code_id: str
    name: str = ""
    description: str = ""
    key_functions: tuple[str, ...] = ()
    key_classes: tuple[str, ...] = ()
    api_calls: tuple[str, ...] = ()
    quality_related_elements: tuple[str, ...] = ()
    cached_summary: bool = False

class CachedCodeSummarizer:
    """Build and cache code summaries with per-code-file reuse."""

    def __init__(
        self,
        llm_service: FrameworkLLMService | None = None,
        prompt_builder: "CodeSummaryPromptBuilder | None" = None,
    ):
        self._cache: dict[str, CodeSummary] = {}
        self.llm_service = llm_service
        self.prompt_builder = prompt_builder or CodeSummaryPromptBuilder()
        self.summary_memory_hits = 0
        self.summary_persistent_cache_hits = 0
        self.summary_cache_misses = 0
        self.summary_api_calls = 0
        self.summary_local_generations = 0

    def summarize_code(self, code_id: str, code_text: str) -> CodeSummary:
        """Return a cached summary for one code artifact."""
        if code_id in self._cache:
            self.summary_memory_hits += 1
            return replace(self._cache[code_id], cached_summary=True)

        payload: dict[str, Any] = {}
        cached_summary = False
        if self.llm_service is not None and self.llm_service.live_api:
            api_calls_before = self._api_call_count()
            result = self.llm_service.complete_json(
                module=SUMMARY_CACHE_MODULE,
                prompt=self.prompt_builder.build_summary_prompt(code_id, code_text),
                requirement_id=SUMMARY_CACHE_REQUIREMENT_ID,
                code_file=code_id,
                cache_prompt=SUMMARY_CACHE_PROMPT,
                cache_requirement_id=SUMMARY_CACHE_REQUIREMENT_ID,
            )
            payload = result.payload
            cached_summary = result.cache_hit
            if result.cache_hit:
                self.summary_persistent_cache_hits += 1
            else:
                self.summary_cache_misses += 1
            self.summary_api_calls += max(0, self._api_call_count() - api_calls_before)
        else:
            self.summary_cache_misses += 1
            self.summary_local_generations += 1

        functions = _tuple_or_extract(payload.get("key_functions"), _FUNCTION_PATTERN, code_text)
        classes = _tuple_or_extract(payload.get("key_classes"), _CLASS_PATTERN, code_text)
        api_calls = _tuple_or_extract(payload.get("api_calls"), _API_CALL_PATTERN, code_text)
        api_calls = tuple(item for item in api_calls if item not in set(functions))[:20]
        quality_elements = _tuple_or_terms(payload.get("quality_related_elements"), code_text)
        name = str(payload.get("name") or (classes[0] if classes else functions[0] if functions else code_id))
        description = str(
            payload.get("description")
            or self._local_description(code_id, functions, classes)
        )
        summary = CodeSummary(
            code_id=code_id,
            name=name,
            description=description,
            key_functions=functions,
            key_classes=classes,
            api_calls=api_calls,
            quality_related_elements=quality_elements,
            cached_summary=cached_summary,
        )
        self._cache[code_id] = summary
        return summary

    def summarize_corpus(
        self, code_corpus: dict[str, str]
    ) -> dict[str, CodeSummary]:
        """Summarize all code artifacts using the local cache."""
        return {
            code_id: self.summarize_code(code_id, code_text)
            for code_id, code_text in code_corpus.items()
        }

    def stats(self) -> dict[str, int]:
        """Return runtime cache statistics for efficiency analysis."""
        summary_cache_hits = (
            self.summary_memory_hits + self.summary_persistent_cache_hits
        )
        return {
            "summary_cache_hits": summary_cache_hits,
            "summary_memory_hits": self.summary_memory_hits,
            "summary_persistent_cache_hits": self.summary_persistent_cache_hits,
            "summary_cache_misses": self.summary_cache_misses,
            "summary_api_calls": self.summary_api_calls,
            "summary_reused": summary_cache_hits,
            "summary_local_generations": self.summary_local_generations,
        }

    def _api_call_count(self) -> int:
        if self.llm_service is None:
            return 0
        return int(getattr(self.llm_service.client.statistics, "api_calls", 0))

    @staticmethod
    def _extract_unique(pattern: re.Pattern[str], text: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(pattern.findall(text)))[:20]

    @staticmethod
    def _quality_elements(text: str) -> tuple[str, ...]:
        lowered = text.casefold()
        return tuple(term for term in _QUALITY_TERMS if term in lowered)

    @staticmethod
    def _local_description(
        code_id: str,
        functions: tuple[str, ...],
        classes: tuple[str, ...],
    ) -> str:
        parts = [code_id]
        if classes:
            parts.append("classes: " + ", ".join(classes[:3]))
        if functions:
            parts.append("functions: " + ", ".join(functions[:5]))
        return " | ".join(parts)


class CodeSummaryPromptBuilder:
    """Build DCAR cached code-summarization prompts."""

    def build_summary_prompt(self, code_id: str, code_text: str) -> str:
        return f"""DCAR cached code summary. Return only valid JSON.
Code file: {code_id}
Code:
{code_text[:4500]}

Schema:
{{
  "name": "main file, class, module, or function name",
  "description": "brief behavior summary",
  "key_functions": ["important functions or methods"],
  "key_classes": ["important classes or types"],
  "api_calls": ["important calls or dependencies"],
  "quality_related_elements": ["security, performance, reliability, validation, error handling, cache, retry, logging, or other quality-related evidence"]
}}"""


def _tuple_or_extract(
    value: Any,
    pattern: re.Pattern[str],
    text: str,
) -> tuple[str, ...]:
    if isinstance(value, list):
        items = tuple(str(item).strip() for item in value if str(item).strip())
        if items:
            return items[:20]
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return tuple(dict.fromkeys(pattern.findall(text)))[:20]


def _tuple_or_terms(value: Any, text: str) -> tuple[str, ...]:
    if isinstance(value, list):
        items = tuple(str(item).strip() for item in value if str(item).strip())
        if items:
            return items[:20]
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return CachedCodeSummarizer._quality_elements(text)
