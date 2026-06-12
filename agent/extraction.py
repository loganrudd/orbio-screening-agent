"""Incremental structured extraction with source attribution.

Given the latest candidate turn and the current known state, extract/correct fields
only (not the whole record each turn). Every recorded field carries provenance.
Anti-over-inference: prefer leaving a field MISSING over guessing it (a guess is a
false positive — the thing the eval harness penalizes). See extraction.md.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Optional

from .llm import LLMClient
from .schemas import (
    ExtractedField,
    FieldFlag,
    Provenance,
    ScreeningRecord,
    TurnExtraction,
    validate_availability,
    validate_position,
    validate_start_date,
    validate_years_experience,
)
from .storage import Turn, WordTiming

_EXTRACTION_SYSTEM_TEMPLATE = """\
You are a structured data extraction assistant for a restaurant job screening system.

Today's date: {today}

Extract candidate information from their CURRENT message only. RULES:
1. Only populate fields explicitly mentioned in this turn's message.
2. Set explicitly_stated=true if the candidate directly stated the value.
3. Set explicitly_stated=false ONLY for clear, specific inferences (rare — prefer null).
4. Set source_text to the EXACT verbatim quote from the candidate's message.
5. Leave fields null if not mentioned — NEVER guess, infer without direct evidence, or hallucinate.
6. Normalize position to: server, line_cook, host, shift_manager, or other
7. Normalize availability to one or more of: weekday_day, weekday_evening, weekend_day, weekend_evening
8. Normalize start date: ISO format YYYY-MM-DD, or "immediate" for ASAP/now answers
   Use today's date above to resolve relative dates ("next Monday", "in two weeks", etc.)
9. work_authorization: true if clearly authorized to work, false if clearly not

Already collected (only re-populate if the candidate is correcting a value):
{current_state}"""


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
        """Return an updated record after merging anything new/corrected from `latest_turn`."""
        # Import here to avoid module-level circular dependency concern
        from .output import assign_flag, compute_confidence

        system = _EXTRACTION_SYSTEM_TEMPLATE.format(
            today=datetime.date.today().isoformat(),
            current_state=_format_current_state(record),
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f'Candidate message (turn {turn_index}): "{latest_turn.content}"\n\n'
                    "Extract only new or corrected information from this message."
                ),
            }
        ]

        try:
            proposal: TurnExtraction = await self._llm.extract_structured(
                system=system,
                messages=messages,
                schema=TurnExtraction,
            )
        except Exception:
            # Extraction failure treated as empty proposal — no fields updated.
            return record

        updates: dict[str, ExtractedField] = {}

        field_proposals = [
            ("candidate_name", proposal.candidate_name),
            ("position_applied_for", proposal.position_applied_for),
            ("years_experience", proposal.years_experience),
            ("relevant_skills", proposal.relevant_skills),
            ("availability", proposal.availability),
            ("earliest_start_date", proposal.earliest_start_date),
            ("work_authorization", proposal.work_authorization),
            ("location_preference", proposal.location_preference),
        ]

        for field_name, field_proposal in field_proposals:
            if field_proposal is None:
                continue  # LLM left this field null — nothing to merge

            source_text = field_proposal.source_text
            # Anti-over-inference: require a non-empty, grounded source_text
            if not source_text or not source_text.strip():
                continue

            value, validated = _process_field(field_name, field_proposal.value)
            if value is None:
                continue  # Field-specific normalization failed — skip

            # Per-field word-aligned attribution: align source_text to STT word timings.
            # Falls back to utterance-level span when words absent or alignment fails.
            audio_start: Optional[float] = None
            audio_end: Optional[float] = None
            stt_conf: Optional[float] = None

            if latest_turn.words:
                aligned = _align_span(source_text, latest_turn.words)
                if aligned is not None:
                    audio_start, audio_end, stt_conf = aligned
                else:
                    audio_start = latest_turn.audio_start_s
                    audio_end = latest_turn.audio_end_s
                    stt_conf = latest_turn.stt_confidence
            elif latest_turn.audio_start_s is not None:
                audio_start = latest_turn.audio_start_s
                audio_end = latest_turn.audio_end_s
                stt_conf = latest_turn.stt_confidence

            provenance = Provenance(
                turn_index=turn_index,
                audio_start_s=audio_start,
                audio_end_s=audio_end,
                source_text=source_text,
            )
            confidence = compute_confidence(
                validated=validated,
                explicitly_stated=field_proposal.explicitly_stated,
                stt_confidence=stt_conf,
            )

            # Build incoming field (placeholder flag, then finalize)
            incoming_tmp = ExtractedField(
                value=value,
                confidence=confidence,
                flag=FieldFlag.NEEDS_REVIEW,
                provenance=[provenance],
            )
            incoming = incoming_tmp.model_copy(
                update={"flag": assign_flag(incoming_tmp, reprompt_capped=False)}
            )

            existing: Optional[ExtractedField] = getattr(record, field_name)
            updates[field_name] = self._merge_field(existing, incoming)

        return record.model_copy(update=updates) if updates else record

    @staticmethod
    def _merge_field(
        existing: Optional[ExtractedField], incoming: ExtractedField
    ) -> ExtractedField:
        """Merge a newly extracted field with any existing one.

        On contradiction (different non-null values), record both in conflicting_values
        and flag CONFLICTING rather than silently overwriting.
        """
        if existing is None:
            return incoming

        if _values_differ(existing.value, incoming.value):
            # Contradiction detected — preserve both values and flag for reviewer
            return ExtractedField(
                value=incoming.value,
                confidence=incoming.confidence,
                flag=FieldFlag.CONFLICTING,
                provenance=existing.provenance + incoming.provenance,
                conflicting_values=list(existing.conflicting_values) + [existing.value],
            )

        # Same value — keep better confidence, accumulate provenance
        if incoming.confidence.score >= existing.confidence.score:
            return ExtractedField(
                value=incoming.value,
                confidence=incoming.confidence,
                flag=incoming.flag,
                provenance=existing.provenance + incoming.provenance,
                conflicting_values=existing.conflicting_values,
            )
        return ExtractedField(
            value=existing.value,
            confidence=existing.confidence,
            flag=existing.flag,
            provenance=existing.provenance + incoming.provenance,
            conflicting_values=existing.conflicting_values,
        )


# --------------------------------------------------------------------------- helpers


def _process_field(field_name: str, raw_value: Any) -> tuple[Any, bool]:
    """Return (normalized_value, is_valid). Returns (None, False) to signal skip."""
    if field_name == "candidate_name":
        v = str(raw_value).strip() if raw_value is not None else ""
        return (v or None, bool(v))

    if field_name == "position_applied_for":
        valid = validate_position(raw_value)
        return (str(raw_value) if isinstance(raw_value, str) else None, valid)

    if field_name == "years_experience":
        try:
            v = int(raw_value)
            return (v, 0 <= v <= 60)
        except (TypeError, ValueError):
            return (None, False)

    if field_name == "relevant_skills":
        if not isinstance(raw_value, list):
            return (None, False)
        skills = [str(s).strip() for s in raw_value if str(s).strip()]
        return (skills or None, len(skills) > 0)

    if field_name == "availability":
        valid = validate_availability(raw_value)
        normalized = raw_value if isinstance(raw_value, list) else None
        return (normalized, valid)

    if field_name == "earliest_start_date":
        normalized = validate_start_date(raw_value)
        return (normalized, normalized is not None)

    if field_name == "work_authorization":
        if isinstance(raw_value, bool):
            return (raw_value, True)
        return (None, False)

    if field_name == "location_preference":
        v = str(raw_value).strip() if raw_value is not None else ""
        return (v or None, bool(v))

    return (raw_value, raw_value is not None)


def _values_differ(a: Any, b: Any) -> bool:
    """True when two extracted values are meaningfully different."""
    if a == b:
        return False
    if isinstance(a, list) and isinstance(b, list):
        return sorted(str(x) for x in a) != sorted(str(x) for x in b)
    return True


def _align_span(
    source_text: str, words: list[WordTiming]
) -> Optional[tuple[float, float, float]]:
    """Align source_text quote to a contiguous window in STT word timings.

    Returns (start_s, end_s, min_confidence) or None when alignment is too
    uncertain. Normalizes both sides (lowercase, punctuation stripped) and
    finds the highest-scoring contiguous window. Returns None when the best
    window matches fewer than half the query tokens.
    """

    def _tok(s: str) -> list[str]:
        return [t for t in re.sub(r"[^\w\s]", "", s.lower()).split() if t]

    query = _tok(source_text)
    if not query or not words:
        return None

    # Flatten per-word tokens with their word index (handles "I'd" → ["i", "d"])
    flat: list[tuple[str, int]] = []
    for wi, w in enumerate(words):
        for tok in _tok(w.word):
            flat.append((tok, wi))

    q_len = len(query)
    if len(flat) < q_len:
        return None

    best_score = 0
    best_start_wi = -1
    best_end_wi = -1

    for i in range(len(flat) - q_len + 1):
        score = sum(1 for j in range(q_len) if flat[i + j][0] == query[j])
        if score > best_score:
            best_score = score
            best_start_wi = flat[i][1]
            best_end_wi = flat[i + q_len - 1][1]

    # Require at least half of query tokens to match (rounded up)
    if best_score < (q_len + 1) // 2 or best_start_wi < 0 or best_end_wi < best_start_wi:
        return None

    span = words[best_start_wi : best_end_wi + 1]
    return (span[0].start_s, span[-1].end_s, min(w.confidence for w in span))


def _format_current_state(record: ScreeningRecord) -> str:
    parts = []
    for fname in ScreeningRecord.required_fields() + ["location_preference"]:
        ef: Optional[ExtractedField] = getattr(record, fname)
        if ef is not None:
            parts.append(f"  {fname}: {ef.value!r} (flag: {ef.flag.value})")
        else:
            parts.append(f"  {fname}: (not yet collected)")
    return "\n".join(parts)
