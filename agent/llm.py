"""Provider-agnostic LLM client wrapper.

Thin layer over the Anthropic Claude client. Owns retries, timeouts, and
structured-output (tool-use) calls. Keep provider specifics in here so the rest of the
system is provider-agnostic.
"""

from __future__ import annotations

import abc
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from .concurrency import ConcurrencyLimiter

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
    """Anthropic Claude implementation. API-key auth via env (ANTHROPIC_API_KEY).

    Structured extraction uses tool use: the schema becomes the tool's input_schema,
    tool_choice='any' forces the model to fill it, and we validate the returned input
    via Pydantic. This is behind the LLMClient interface so tests can mock it.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        limiter: Optional[ConcurrencyLimiter] = None,
    ) -> None:
        import anthropic

        self._model = model
        self._limiter = limiter or ConcurrencyLimiter()
        self._client = anthropic.AsyncAnthropic(
            max_retries=max_retries,
            timeout=timeout_s,
        )

    async def extract_structured(
        self, *, system: str, messages: list[dict], schema: Type[T]
    ) -> T:
        tool = {
            "name": "record_extraction",
            "description": (
                "Record the fields extracted from the candidate's message. "
                "Only populate fields the candidate explicitly mentioned."
            ),
            "input_schema": schema.model_json_schema(),
        }

        async def _call() -> T:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "any"},
            )
            for block in response.content:
                if block.type == "tool_use":
                    return schema.model_validate(block.input)
            raise ValueError("No tool_use block in extraction response")

        return await self._limiter.run(_call)

    async def respond(self, *, system: str, messages: list[dict]) -> str:
        async def _call() -> str:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=system,
                messages=messages,
            )
            return response.content[0].text

        return await self._limiter.run(_call)
