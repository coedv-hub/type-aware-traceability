"""Common interface for OpenAI-compatible LLM providers."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import sleep
from typing import Any

from .rate_limiter import RateLimiter


@dataclass(frozen=True)
class LLMResponse:
    """Normalized provider response used by all LLM baselines."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    provider: str
    raw_id: str = ""
    estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LLMResponse":
        return cls(**payload)


@dataclass
class ClientStatistics:
    api_calls: int = 0
    http_attempts: int = 0
    retries: int = 0
    http_429_count: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ProviderHTTPError(RuntimeError):
    def __init__(
        self,
        provider: str,
        status_code: int,
        details: str,
        retry_after: float | None,
    ):
        super().__init__(
            f"{provider} API returned HTTP {status_code}: {details[:500]}"
        )
        self.status_code = status_code
        self.retry_after = retry_after


class LLMClient(ABC):
    """Minimal provider-neutral completion client."""

    provider = "base"

    def __init__(
        self,
        model: str,
        api_key_env: str,
        base_url: str,
        timeout_seconds: int = 120,
        max_completion_tokens: int = 300,
        chars_per_token: float = 3.5,
        max_retries: int = 8,
        backoff_base_seconds: float = 2.0,
        backoff_max_seconds: float = 60.0,
        rate_limiter: RateLimiter | None = None,
    ):
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_completion_tokens = max_completion_tokens
        self.chars_per_token = chars_per_token
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.rate_limiter = rate_limiter or RateLimiter()
        self.statistics = ClientStatistics()

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "").strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def estimate_tokens(self, text: str) -> int:
        """Conservative tokenizer-free estimate for offline planning."""
        return max(1, math.ceil(len(text) / self.chars_per_token))

    def estimated_response(self, prompt: str) -> LLMResponse:
        prompt_tokens = self.estimate_tokens(prompt)
        completion_tokens = self.max_completion_tokens
        return LLMResponse(
            text=json.dumps(
                {
                    "prediction": "Unknown",
                    "confidence": None,
                    "explanation": "Dry-run estimate; no API call was made.",
                }
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=self.model,
            provider=self.provider,
            estimated=True,
        )

    @staticmethod
    def _retry_after_seconds(headers: Any) -> float | None:
        if headers is None:
            return None
        value = headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(value))
                now = (
                    parsedate_to_datetime(headers.get("Date"))
                    if headers.get("Date")
                    else datetime.now(timezone.utc)
                )
                return max(0.0, (retry_at - now).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    def _post_once(
        self, prompt: str, extra_body: dict[str, Any] | None
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_completion_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        body.update(extra_body or {})
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        self.statistics.http_attempts += 1
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise ProviderHTTPError(
                self.provider,
                exc.code,
                details,
                self._retry_after_seconds(exc.headers),
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{self.provider} API request failed: {exc}") from exc

        try:
            usage = payload.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(
                usage.get("total_tokens") or prompt_tokens + completion_tokens
            )
            response = LLMResponse(
                text=payload["choices"][0]["message"]["content"],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=str(payload.get("model") or self.model),
                provider=self.provider,
                raw_id=str(payload.get("id") or ""),
            )
            return response
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Unexpected {self.provider} response schema."
            ) from exc

    def _post_chat_completion(
        self, prompt: str, extra_body: dict[str, Any] | None = None
    ) -> LLMResponse:
        if not self.is_configured:
            raise RuntimeError(
                f"{self.api_key_env} is not configured; refusing to call the API."
            )
        reserved_tokens = (
            self.estimate_tokens(prompt) + self.max_completion_tokens
        )
        retry_number = 0
        while True:
            self.rate_limiter.acquire(reserved_tokens)
            try:
                response = self._post_once(prompt, extra_body)
                self.statistics.api_calls += 1
                return response
            except ProviderHTTPError as exc:
                if exc.status_code != 429:
                    self.statistics.failures += 1
                    raise
                self.statistics.http_429_count += 1
                if retry_number >= self.max_retries:
                    self.statistics.failures += 1
                    raise RuntimeError(
                        f"{exc} Retry limit ({self.max_retries}) exhausted."
                    ) from exc
                delay = (
                    exc.retry_after
                    if exc.retry_after is not None
                    else min(
                        self.backoff_max_seconds,
                        self.backoff_base_seconds * (2**retry_number),
                    )
                )
                retry_number += 1
                self.statistics.retries += 1
                print(
                    f"HTTP 429: retry {retry_number}/{self.max_retries} "
                    f"after {delay:.1f}s "
                    f"(429 count={self.statistics.http_429_count})."
                )
                sleep(delay)
            except RuntimeError:
                self.statistics.failures += 1
                raise

    @abstractmethod
    def complete(self, prompt: str) -> LLMResponse:
        """Return one normalized completion."""
