"""Incremental structured extraction with source attribution.

Given the latest candidate turn and the current known state, extract/correct fields
only (not the whole record each turn). Every recorded field carries provenance.
Anti-over-inference: prefer leaving a field MISSING over guessing it (a guess is a
false positive — the thing the eval harness penalizes). See extraction.md.
"""

from __future__ import annotations

from typing import Optional

from .llm import LLMClient
from .schemas import ExtractedField, Provenance, ScreeningRecord
from .storage import Turn


class Extractor:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def extract_turn(
        self,
        *,
        record: ScreeningRecord,
        latest_turn: Turn,
        turn_index: int,
        language: str,
    ) -> ScreeningRecord:
        """Return an updated record after merging anything new/corrected from
        `latest_turn`. Must:
          - pass current known state so the model fills gaps/corrections only
          - attach Provenance (turn_index + STT span if voice) to each new value
          - reject values not grounded in the candidate's actual utterance
          - detect contradictions against existing values (flag CONFLICTING upstream)
        """
        # TODO(execute):
        #   1. build extraction prompt with current state + latest turn
        #   2. call self._llm.extract_structured(...)
        #   3. validate, attach provenance, merge into a copy of `record`
        #   4. return merged record (flag/confidence assigned in output.py)
        raise NotImplementedError

    @staticmethod
    def _build_provenance(turn: Turn, turn_index: int, source_text: str) -> Provenance:
        return Provenance(
            turn_index=turn_index,
            audio_start_s=turn.audio_start_s,
            audio_end_s=turn.audio_end_s,
            source_text=source_text,
        )

    @staticmethod
    def _merge_field(
        existing: Optional[ExtractedField], incoming: ExtractedField
    ) -> ExtractedField:
        """Merge a newly extracted field with any existing one, preserving conflict
        information rather than silently overwriting. See output-contract.md."""
        # TODO(execute)
        raise NotImplementedError
