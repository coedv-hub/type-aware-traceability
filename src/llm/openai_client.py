"""OpenAI Chat Completions client.

``LLMClient`` implementation for OpenAI, used by every Stage 4/5 code path
that needs a live LLM call: the Direct Prompting / Generic RAG-LLM
baselines and the Proposed Framework's Understanding, Alignment, and
Verification stages. The API key is read only from the ``OPENAI_API_KEY``
environment variable (see ``base_client.LLMClient``); no key is ever read
from or written to a config file. Retry/backoff, rate limiting, and token
accounting are implemented once in ``base_client.py`` and shared with
``deepseek_client.py`` so provider choice does not change any calling code
elsewhere in the pipeline.
"""

from __future__ import annotations

from .base_client import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    provider = "openai"

    def complete(self, prompt: str) -> LLMResponse:
        return self._post_chat_completion(prompt)
