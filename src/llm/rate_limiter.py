"""Conservative process-wide RPM/TPM limiter for LLM API calls."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    """Reserve request and token capacity in a rolling one-minute window."""

    WINDOW_SECONDS = 60.0

    def __init__(
        self,
        requests_per_minute: int = 10,
        tokens_per_minute: int = 20_000,
        min_interval_seconds: float = 3.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        logger: Callable[[str], None] = print,
    ):
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")
        if tokens_per_minute < 1:
            raise ValueError("tokens_per_minute must be at least 1")
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds cannot be negative")
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.min_interval_seconds = min_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._logger = logger
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._last_request: float | None = None
        self._lock = threading.Lock()

    def _discard_expired(self, now: float) -> None:
        cutoff = now - self.WINDOW_SECONDS
        while self._requests and self._requests[0] <= cutoff:
            self._requests.popleft()
        while self._tokens and self._tokens[0][0] <= cutoff:
            self._tokens.popleft()

    def _wait_required(self, now: float, tokens: int) -> float:
        waits = [0.0]
        if self._last_request is not None:
            waits.append(
                self.min_interval_seconds - (now - self._last_request)
            )
        if len(self._requests) >= self.requests_per_minute:
            waits.append(
                self.WINDOW_SECONDS - (now - self._requests[0])
            )
        used_tokens = sum(amount for _, amount in self._tokens)
        if used_tokens + tokens > self.tokens_per_minute and self._tokens:
            tokens_to_expire = used_tokens + tokens - self.tokens_per_minute
            expired = 0
            for timestamp, amount in self._tokens:
                expired += amount
                if expired >= tokens_to_expire:
                    waits.append(
                        self.WINDOW_SECONDS - (now - timestamp)
                    )
                    break
        return max(waits)

    def acquire(self, estimated_tokens: int) -> None:
        """Block until capacity is available, then reserve it immediately."""
        tokens = min(
            max(1, int(estimated_tokens)), self.tokens_per_minute
        )
        with self._lock:
            while True:
                now = self._clock()
                self._discard_expired(now)
                wait_seconds = self._wait_required(now, tokens)
                if wait_seconds <= 0:
                    self._requests.append(now)
                    self._tokens.append((now, tokens))
                    self._last_request = now
                    return
                self._logger(
                    "Rate limiter: sleeping "
                    f"{wait_seconds:.1f}s (RPM={self.requests_per_minute}, "
                    f"TPM={self.tokens_per_minute})."
                )
                self._sleep(wait_seconds)
