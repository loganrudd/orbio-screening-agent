"""Provider-agnostic LLM client wrapper.

Thin layer over the Anthropic Claude client. Owns retries, timeouts, and
structured-output (tool-use) calls. Keep provider specifics in here so the rest of the
system is provider-agnostic.
"""

from __future__ import annotations

import abc
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(abc.ABC):
    """Interface so tests can mock the LLM (see testing.md — never assert on real
    model output strings; mock here)."""

    @abc.abstractmethod
    async def extract_structured(
        self,
        *,
        system: str,
        messages: list[dict],
        schema: Type[T],
    ) -> T:
        """Call the model in structured-output mode and return a validated `schema`
        instance. Raises on validation failure (caller treats as not-collected)."""
        ...

    @abc.abstractmethod
    async def respond(self, *, system: str, messages: list[dict]) -> str:
        """Free-form assistant turn (the agent's next question/prompt)."""
        ...


class ClaudeClient(LLMClient):
    """Anthropic Claude implementation. API-key auth via env (ANTHROPIC_API_KEY)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",  # confirm current model id in Explore
        *,
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        # TODO(execute): init anthropic.AsyncAnthropic() (key from env).

    async def extract_structured(
        self, *, system: str, messages: list[dict], schema: Type[T]
    ) -> T:
        # TODO(execute): use tool-use / structured output with a tool whose input
        #   schema is `schema.model_json_schema()`; validate the tool input via
        #   schema.model_validate(...); retries w/ backoff via the concurrency limiter.
        raise NotImplementedError

    async def respond(self, *, system: str, messages: list[dict]) -> str:
        # TODO(execute): standard messages call; return text.
        raise NotImplementedError
