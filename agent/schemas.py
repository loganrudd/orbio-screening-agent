"""Pydantic schemas for the restaurant screening agent.

Defines the collected field set, provenance, flagging, and confidence types that the
rest of the system depends on. This module is the single source of truth for the
shape of a screening record.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


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
    """Rule-derived confidence (NOT model self-report). See extraction.md."""

    validated: bool
    explicitly_stated: bool  # False => inferred
    stt_confidence: Optional[float] = None  # voice only, 0..1
    score: float = Field(ge=0.0, le=1.0)

    # TODO(execute): implement the deterministic combination of the inputs into `score`
    #   e.g. base on validated & explicitly_stated, attenuate by stt_confidence.


# --------------------------------------------------------------- per-field wrapper


class ExtractedField(BaseModel):
    """A single collected field with everything the reviewer needs."""

    value: object  # concrete type depends on the field; validated per-field below
    confidence: Confidence
    flag: FieldFlag
    provenance: list[Provenance] = Field(default_factory=list)
    # For CONFLICTING: alternative values seen across turns (kept, not overwritten).
    conflicting_values: list[object] = Field(default_factory=list)


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
    # TODO(execute)
    raise NotImplementedError


def validate_position(value: object) -> bool:
    """Maps to a Position enum member."""
    # TODO(execute)
    raise NotImplementedError


def validate_availability(value: object) -> bool:
    """Non-empty list of valid Shift members."""
    # TODO(execute)
    raise NotImplementedError


def validate_start_date(value: object) -> Optional[date]:
    """Normalize to an ISO date if possible; None if unparseable."""
    # TODO(execute)
    raise NotImplementedError
