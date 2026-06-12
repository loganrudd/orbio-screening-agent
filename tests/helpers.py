"""Shared test helpers: MockLLM and field/record builders.

Import from tests that need them; not a pytest conftest.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional, Type, TypeVar

from agent.llm import LLMClient
from agent.schemas import (
    Confidence,
    ExtractedField,
    FieldFlag,
    Provenance,
    ScreeningRecord,
    TurnExtraction,
)

T = TypeVar("T")


class MockLLM(LLMClient):
    """Deterministic LLM stand-in for unit tests.

    Usage:
        # Always return a fixed TurnExtraction
        llm = MockLLM(extraction=TurnExtraction(...))

        # Return different proposals per call (consumed in order)
        llm = MockLLM(extractions=[TurnExtraction(...), TurnExtraction(...)])

        # Fixed free-form reply
        llm = MockLLM(reply="Got it, what position?")

        # Simulate extraction failure
        llm = MockLLM(raise_on_extract=True)
    """

    def __init__(
        self,
        *,
        extraction: Optional[TurnExtraction] = None,
        extractions: Optional[list[TurnExtraction]] = None,
        reply: str = "Okay, noted.",
        replies: Optional[list[str]] = None,
        raise_on_extract: bool = False,
    ) -> None:
        self._extraction = extraction
        self._extractions: deque[TurnExtraction] = deque(extractions or [])
        self._reply = reply
        self._replies: deque[str] = deque(replies or [])
        self._raise_on_extract = raise_on_extract

    async def extract_structured(
        self, *, system: str, messages: list[dict], schema: Type[T]
    ) -> T:
        if self._raise_on_extract:
            raise RuntimeError("simulated extraction failure")
        if self._extractions:
            return self._extractions.popleft()  # type: ignore[return-value]
        if self._extraction is not None:
            return self._extraction  # type: ignore[return-value]
        return TurnExtraction()  # type: ignore[return-value]  # empty proposal

    async def respond(self, *, system: str, messages: list[dict]) -> str:
        if self._replies:
            return self._replies.popleft()
        return self._reply


# --------------------------------------------------------------------------- builders


def make_provenance(
    turn_index: int = 0,
    source_text: str = "test source",
    audio_start_s: Optional[float] = None,
    audio_end_s: Optional[float] = None,
) -> Provenance:
    return Provenance(
        turn_index=turn_index,
        source_text=source_text,
        audio_start_s=audio_start_s,
        audio_end_s=audio_end_s,
    )


def make_confidence(
    validated: bool = True,
    explicitly_stated: bool = True,
    stt_confidence: Optional[float] = None,
) -> Confidence:
    return Confidence(
        validated=validated,
        explicitly_stated=explicitly_stated,
        stt_confidence=stt_confidence,
    )


def make_field(
    value: Any,
    *,
    validated: bool = True,
    explicitly_stated: bool = True,
    stt_confidence: Optional[float] = None,
    flag: FieldFlag = FieldFlag.CONFIRMED,
    turn_index: int = 0,
    source_text: str = "test source",
    conflicting_values: Optional[list[Any]] = None,
) -> ExtractedField:
    confidence = make_confidence(
        validated=validated,
        explicitly_stated=explicitly_stated,
        stt_confidence=stt_confidence,
    )
    return ExtractedField(
        value=value,
        confidence=confidence,
        flag=flag,
        provenance=[make_provenance(turn_index=turn_index, source_text=source_text)],
        conflicting_values=conflicting_values or [],
    )


def make_record(**fields: Optional[ExtractedField]) -> ScreeningRecord:
    """Build a ScreeningRecord with only the specified fields set (rest None)."""
    return ScreeningRecord(**fields)
