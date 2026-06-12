"""Reviewer-facing output: confidence computation, flag assignment, rendering, summary.

The output is built for a human reviewer deciding whether to advance a candidate.
See output-contract.md. Confidence is rule-derived (extraction.md), never model
self-reported.
"""

from __future__ import annotations

from .schemas import Confidence, ExtractedField, FieldFlag, ScreeningRecord


def compute_confidence(
    *,
    validated: bool,
    explicitly_stated: bool,
    stt_confidence: float | None = None,
) -> Confidence:
    """Deterministic combination of the confidence inputs into a 0..1 score.

    Rough intent (finalize in execute):
      - validated & explicitly_stated => high
      - inferred (not explicitly_stated) => capped lower
      - not validated => low
      - if voice, attenuate by stt_confidence
    """
    # TODO(execute)
    raise NotImplementedError


def assign_flag(field: ExtractedField, *, reprompt_capped: bool) -> FieldFlag:
    """Map a field's state to a reviewer flag.

      - CONFIRMED: validated, explicitly stated, high confidence
      - NEEDS_REVIEW: low confidence, inferred, or accepted after re-prompt cap
      - CONFLICTING: contradictory values across turns
      - MISSING: handled at the record level (no field present)
    """
    # TODO(execute)
    raise NotImplementedError


def render_reviewer_table(record: ScreeningRecord) -> str:
    """Render a formatted terminal table with visual flag states
    (✓ confirmed / ⚠ needs_review / ✗ missing / ! conflicting).

    A clean CLI table is sufficient — no web UI required (output-contract.md).
    """
    # TODO(execute)
    raise NotImplementedError


def build_summary(record: ScreeningRecord) -> str:
    """Structured synopsis: what to trust, what to verify, short natural-language
    summary. This is the SUMMARY-state artifact."""
    # TODO(execute)
    raise NotImplementedError
