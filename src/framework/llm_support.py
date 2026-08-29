"""Shared LLM/cache bridge for the proposed framework."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.llm.base_client import LLMClient, LLMResponse
from src.llm.cache import CacheIdentity, LLMCache
from src.llm.token_statistics import TokenRecord, estimate_cost


class IncompleteBatchResponseError(RuntimeError):
    """A validated batch call never produced a complete response.

    Raised by complete_json() when `validate` is given and every attempt
    (up to `max_attempts`) either fails to parse or fails validate().
    Callers must treat this as a hard failure of the whole run -- never
    substitute a locally-computed fallback score for the missing
    candidates, since that is exactly the silent-fallback bug this
    mechanism exists to eliminate."""


class RunBudgetExceededError(RuntimeError):
    """The run's real-API-call budget (FrameworkLLMService.max_api_calls)
    was reached. Raised BEFORE the next real call is made -- the call
    itself never happens, so this is not an accounting-after-the-fact
    check. Callers must treat this as a hard failure of the whole run,
    same as IncompleteBatchResponseError: no predictions are written."""


@dataclass(frozen=True)
class FrameworkLLMResult:
    """One framework LLM completion with parsed JSON and accounting data."""

    response: LLMResponse
    payload: dict[str, Any]
    cache_hit: bool
    parse_error: str = ""


class FrameworkLLMService:
    """Reuse Stage 4 LLM client, cache, and token accounting."""

    def __init__(
        self,
        client: LLMClient,
        cache: LLMCache,
        dataset: str,
        *,
        live_api: bool,
        input_price: float = 0.0,
        output_price: float = 0.0,
        records: list[TokenRecord] | None = None,
        max_api_calls: int | None = None,
    ):
        self.client = client
        self.cache = cache
        self.dataset = dataset
        self.live_api = live_api
        self.input_price = input_price
        self.output_price = output_price
        self.records = records if records is not None else []
        self.cache_hits = 0
        self.cache_misses = 0
        self.parse_errors = 0
        self.empty_responses = 0
        self.max_api_calls = max_api_calls
        self.real_api_calls = 0

    def complete_json(
        self,
        *,
        module: str,
        prompt: str,
        requirement_id: str,
        code_file: str = "",
        cache_prompt: str | None = None,
        cache_requirement_id: str | None = None,
        max_completion_tokens: int | None = None,
        validate: Callable[[dict[str, Any]], bool] | None = None,
        max_attempts: int = 1,
    ) -> FrameworkLLMResult:
        """Read from cache or call the configured client only in live-api mode.

        When `validate` is given, a response (cached or fresh) is only
        accepted -- and only ever written to the cache -- once it parses
        cleanly and validate(payload) returns True. An invalid response is
        never cached, so a retry can never be poisoned by a prior bad
        attempt; if every one of `max_attempts` attempts is invalid, raises
        IncompleteBatchResponseError instead of returning a partial or
        locally-substituted result. `validate=None` (the default) preserves
        the original single-attempt, always-cache behavior exactly, for
        every non-batched caller (understanding, code summarization,
        single-candidate alignment/verification checks)."""
        identity_prompt = prompt if cache_prompt is None else cache_prompt
        identity_requirement_id = (
            requirement_id
            if cache_requirement_id is None
            else cache_requirement_id
        )
        identity = CacheIdentity(
            model=self.client.model,
            prompt=f"[{module}]\n{identity_prompt}",
            dataset=self.dataset,
            requirement_id=identity_requirement_id,
            code_file=code_file or module,
            provider=self.client.provider,
        )
        attempts = max(1, max_attempts)
        last_error = ""
        for attempt in range(attempts):
            cached = (
                self.cache.get(identity, allow_estimated=not self.live_api)
                if attempt == 0
                else None
            )
            cache_hit = cached is not None
            if not cache_hit:
                if (
                    self.max_api_calls is not None
                    and self.real_api_calls >= self.max_api_calls
                ):
                    raise RunBudgetExceededError(
                        f"Real API call budget exhausted "
                        f"({self.real_api_calls}/{self.max_api_calls}); refusing "
                        f"the next call for {module} (requirement "
                        f"{identity_requirement_id}, code_file="
                        f"{code_file or module}). No predictions were written "
                        f"for this run."
                    )
                response = self._complete_with_limit(prompt, max_completion_tokens)
                self.real_api_calls += 1
            else:
                response = cached
            payload, parse_error = self._parse_json(response.text)
            valid = validate is None or (not parse_error and validate(payload))

            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            self._record(
                module=module,
                requirement_id=requirement_id,
                code_file=code_file,
                response=response,
                cache_hit=cache_hit,
                parse_error=parse_error,
            )

            if valid:
                if not cache_hit:
                    self.cache.put(identity, response)
                return FrameworkLLMResult(
                    response=response,
                    payload=payload,
                    cache_hit=cache_hit,
                    parse_error=parse_error,
                )
            last_error = parse_error or "validate() rejected response"

        raise IncompleteBatchResponseError(
            f"{module} batch for requirement {identity_requirement_id} "
            f"(code_file={code_file or module}) failed validation after "
            f"{attempts} attempt(s): {last_error}"
        )

    def _record(
        self,
        *,
        module: str,
        requirement_id: str,
        code_file: str,
        response: LLMResponse,
        cache_hit: bool,
        parse_error: str,
    ) -> None:
        self.parse_errors += int(bool(parse_error))
        self.empty_responses += int(not response.text.strip())
        self.records.append(
            TokenRecord(
                dataset=self.dataset,
                baseline=f"Proposed Framework::{module}",
                provider=self.client.provider,
                model=self.client.model,
                requirement_id=requirement_id,
                code_file=code_file or module,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                estimated_cost_usd=estimate_cost(
                    response.prompt_tokens,
                    response.completion_tokens,
                    self.input_price,
                    self.output_price,
                ),
                token_source=(
                    "provider_usage"
                    if not response.estimated
                    else "estimated"
                    if self.live_api
                    else "no_api_local"
                ),
                cache_hit=cache_hit,
            )
        )

    def _complete_with_limit(
        self,
        prompt: str,
        max_completion_tokens: int | None,
    ) -> LLMResponse:
        if max_completion_tokens is None:
            return (
                self.client.complete(prompt)
                if self.live_api
                else self.client.estimated_response(prompt)
            )
        original_limit = self.client.max_completion_tokens
        self.client.max_completion_tokens = max_completion_tokens
        try:
            return (
                self.client.complete(prompt)
                if self.live_api
                else self.client.estimated_response(prompt)
            )
        finally:
            self.client.max_completion_tokens = original_limit

    @staticmethod
    def _parse_json(text: str) -> tuple[dict[str, Any], str]:
        candidate = text.strip()
        if not candidate:
            return {}, "empty response"

        attempts = _json_candidates(candidate)
        errors: list[str] = []
        for item in attempts:
            try:
                payload = json.loads(item)
            except json.JSONDecodeError as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            if isinstance(payload, dict):
                return payload, ""
            if isinstance(payload, list):
                return {"items": payload, "candidates": payload}, ""
            return {}, "JSON payload is not an object or array"

        partial = _recover_partial_candidates(candidate)
        if partial:
            return {"candidates": partial}, ""

        return {}, errors[0] if errors else "No JSON object or array found"


def chunk_cache_suffix(batch_cache_key: str, code_ids: list[str] | tuple[str, ...]) -> str:
    """Cache-key suffix that pins a chunked batch call to its exact chunk
    content, so a different chunking of the same requirement (different B,
    different candidate set) can never reuse another chunk's cached
    response."""
    prefix = batch_cache_key or "batch"
    return f"{prefix}::chunk::{','.join(code_ids)}"


def _json_candidates(text: str) -> list[str]:
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(item.strip() for item in fenced if item.strip())
    candidates.append(text)

    object_text = _balanced_json_slice(text, "{", "}")
    if object_text:
        candidates.append(object_text)
    array_text = _balanced_json_slice(text, "[", "]")
    if array_text:
        candidates.append(array_text)

    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def _balanced_json_slice(text: str, opening: str, closing: str) -> str:
    start = text.find(opening)
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _recover_partial_candidates(text: str) -> list[dict[str, Any]]:
    marker = '"candidates"'
    marker_index = text.find(marker)
    search_start = marker_index if marker_index >= 0 else 0
    array_start = text.find("[", search_start)
    if array_start < 0:
        return []

    items: list[dict[str, Any]] = []
    index = array_start + 1
    while index < len(text):
        object_start = text.find("{", index)
        if object_start < 0:
            break
        object_text = _balanced_json_slice(text[object_start:], "{", "}")
        if not object_text:
            break
        try:
            payload = json.loads(object_text)
        except json.JSONDecodeError:
            index = object_start + 1
            continue
        if isinstance(payload, dict):
            items.append(payload)
        index = object_start + len(object_text)
    return items
