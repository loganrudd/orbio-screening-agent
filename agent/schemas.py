"""Pydantic schemas for the restaurant screening agent.

Defines the collected field set, provenance, flagging, and confidence types that the
rest of the system depends on. This module is the single source of truth for the
shape of a screening record.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field


# --------------------------------------------------------------------------- enums


class Position(str, Enum):
    SERVER = "server"
    LINE_COOK = "line_cook"
    HOST = "host"
    SHIFT_MANAGER = "shift_manager"
    OTHER = "other"


class Shift(str, Enum):
    WEEKDAY_DAY = "weekday_day"
    WEEKDAY_EVENING = "weekday_evening"
    WEEKEND_DAY = "weekend_day"
    WEEKEND_EVENING = "weekend_evening"


class FieldFlag(str, Enum):
    """Reviewer-facing flag. See .claude/rules/output-contract.md."""

    CONFIRMED = "confirmed"
    NEEDS_REVIEW = "needs_review"
    MISSING = "missing"
    CONFLICTING = "conflicting"


class ConversationState(str, Enum):
    GREETING = "greeting"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    SUMMARY = "summary"


# --------------------------------------------------------- provenance / confidence


class Provenance(BaseModel):
    """Where an extracted value came from (first-class concept #1: attribution)."""

    turn_index: int
    # Voice only: STT timestamp span (seconds) of the candidate speech for this value.
    audio_start_s: Optional[float] = None
    audio_end_s: Optional[float] = None
    # The raw candidate utterance fragment this value was grounded in.
    source_text: Optional[str] = None


class Confidence(BaseModel):
    """Rule-derived confidence (NOT model self-report). See extraction.md.

    Score formula (docs/architecture/decisions.md #3):
      0.9  validated & explicitly_stated
      0.6  validated & inferred (not explicitly_stated)
      0.3  not validated
      ×stt_confidence when voice input (0..1)
    CONFIRMED threshold: score >= 0.8
    """

    validated: bool
    explicitly_stated: bool  # False => inferred
    stt_confidence: Optional[float] = None  # voice only, 0..1

    @computed_field
    @property
    def score(self) -> float:
        if not self.validated:
            base = 0.3
        elif self.explicitly_stated:
            base = 0.9
        else:
            base = 0.6
        if self.stt_confidence is not None:
            return base * self.stt_confidence
        return base


# --------------------------------------------------------------- per-field wrapper


class ExtractedField(BaseModel):
    """A single collected field with everything the reviewer needs."""

    value: Any  # concrete type depends on the field; validated per-field below
    confidence: Confidence
    flag: FieldFlag
    provenance: list[Provenance] = Field(default_factory=list)
    # For CONFLICTING: alternative values seen across turns (kept, not overwritten).
    conflicting_values: list[Any] = Field(default_factory=list)


# ----------------------------------------------------------------- the record


class ScreeningRecord(BaseModel):
    """Canonical extracted record for a restaurant screening conversation.

    Each field is optional at the type level because it may be MISSING; the flag
    captures the real status. Validation of value formats happens in the validators
    and in extraction.py before a field is marked CONFIRMED.
    """

    candidate_name: Optional[ExtractedField] = None
    position_applied_for: Optional[ExtractedField] = None
    years_experience: Optional[ExtractedField] = None
    relevant_skills: Optional[ExtractedField] = None
    availability: Optional[ExtractedField] = None
    earliest_start_date: Optional[ExtractedField] = None
    work_authorization: Optional[ExtractedField] = None
    location_preference: Optional[ExtractedField] = None

    @staticmethod
    def required_fields() -> list[str]:
        return [
            "candidate_name",
            "position_applied_for",
            "years_experience",
            "relevant_skills",
            "availability",
            "earliest_start_date",
            "work_authorization",
        ]


# ----------------------------------------------- raw-value validators (helpers)
# These validate the *value* a field should hold once extracted, used by
# extraction.py to decide validated=True/False. Kept as standalone helpers so they
# can be unit-tested without constructing the full wrapper.


def validate_years_experience(value: object) -> bool:
    """0 <= years <= 60 (sane upper bound)."""
    try:
        n = int(value)  # type: ignore[arg-type]
        return 0 <= n <= 60
    except (TypeError, ValueError):
        return False


def validate_position(value: object) -> bool:
    """Maps to a Position enum member."""
    if not isinstance(value, str):
        return False
    try:
        Position(value)
        return True
    except ValueError:
        return False


def validate_availability(value: object) -> bool:
    """Non-empty list of valid Shift members."""
    if not isinstance(value, list) or len(value) == 0:
        return False
    try:
        for item in value:
            Shift(item)
        return True
    except ValueError:
        return False


def validate_start_date(value: object) -> Optional[str]:
    """Normalize to ISO date string or 'immediate' sentinel; None if invalid.

    Returns str (not date) because 'immediate' cannot be represented as a date object.
    CONFIRMED threshold still applies — the string value is stored as-is.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in ("immediate", "asap", "now", "right away", "immediately", "today"):
        return "immediate"
    try:
        parsed = date.fromisoformat(value.strip())
        return parsed.isoformat()
    except ValueError:
        return None


# ----------------------------------------------------------------- LLM proposal model
# This is the ONLY schema the LLM fills. ScreeningRecord carries rule-derived
# confidence/flag/provenance — those are computed downstream, never model-reported.
# The proposal → ExtractedField mapping happens in extraction.py.


class FieldProposal(BaseModel):
    """Base for a per-field LLM proposal."""

    explicitly_stated: bool = True
    source_text: str = ""  # verbatim quote from candidate speech grounding this value


class StringProposal(FieldProposal):
    value: str


class IntProposal(FieldProposal):
    value: int


class StringListProposal(FieldProposal):
    value: list[str]


class BoolProposal(FieldProposal):
    value: bool


class TurnExtraction(BaseModel):
    """Structured output the LLM fills after each candidate turn.

    Only fields mentioned in the current turn should be populated. Anything the
    candidate did NOT say MUST be left as None — never guess, never infer without
    a direct source. source_text must be a verbatim quote, not a paraphrase.
    """

    candidate_name: Optional[StringProposal] = None
    position_applied_for: Optional[StringProposal] = None
    years_experience: Optional[IntProposal] = None
    relevant_skills: Optional[StringListProposal] = None
    availability: Optional[StringListProposal] = None
    earliest_start_date: Optional[StringProposal] = None
    work_authorization: Optional[BoolProposal] = None
    location_preference: Optional[StringProposal] = None
