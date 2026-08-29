"""DeepSeek OpenAI-compatible Chat Completions client.

Analogous ``LLMClient`` implementation for DeepSeek, selected via
``--provider deepseek``. Reads its key only from the ``DEEPSEEK_API_KEY``
environment variable and shares the same retry/rate-limiting base class as
``openai_client.py``. ``extra_body={"thinking": {"type": "disabled"}}``
disables DeepSeek's reasoning-trace output so responses stay comparable in
shape/cost to the OpenAI baseline calls.
"""

from __future__ import annotations

from .base_client import LLMClient, LLMResponse


class DeepSeekClient(LLMClient):
    provider = "deepseek"

    def complete(self, prompt: str) -> LLMResponse:
        return self._post_chat_completion(
            prompt, extra_body={"thinking": {"type": "disabled"}}
        )
