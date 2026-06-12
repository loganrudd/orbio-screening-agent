"""CLI smoke tests: scripted conversation through the engine to SUMMARY via MockLLM.

Uses JsonFileStore(tmp_path) — no network, no credentials.
Verifies that the engine reaches done=True with a persisted record.
"""

import datetime

import pytest

from agent.conversation import ConversationEngine
from agent.extraction import Extractor
from agent.schemas import (
    BoolProposal,
    ConversationState,
    IntProposal,
    ScreeningRecord,
    StringListProposal,
    StringProposal,
    TurnExtraction,
)
from agent.storage import JsonFileStore, Turn

from helpers import MockLLM


def _turn(content: str) -> Turn:
    return Turn(
        role="candidate",
        content=content,
        ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def _make_engine(llm: MockLLM, store: JsonFileStore) -> ConversationEngine:
    extractor = Extractor(llm=llm)
    return ConversationEngine(store=store, llm=llm, extractor=extractor)


def _all_fields_extraction() -> TurnExtraction:
    return TurnExtraction(
        candidate_name=StringProposal(value="Maria Gonzalez", source_text="Maria Gonzalez", explicitly_stated=True),
        position_applied_for=StringProposal(value="line_cook", source_text="line cook", explicitly_stated=True),
        years_experience=IntProposal(value=4, source_text="four years", explicitly_stated=True),
        relevant_skills=StringListProposal(value=["grill station"], source_text="grill station", explicitly_stated=True),
        availability=StringListProposal(
            value=["weekday_evening", "weekend_day", "weekend_evening"],
            source_text="weekday evenings and weekends",
            explicitly_stated=True,
        ),
        earliest_start_date=StringProposal(value="2026-06-23", source_text="June 23rd 2026", explicitly_stated=True),
        work_authorization=BoolProposal(value=True, source_text="yes I'm authorized", explicitly_stated=True),
    )


class TestCLISmoke:
    async def test_full_screening_reaches_done(self, tmp_path):
        """Happy-path: one turn fills all fields, confirmation turn → done=True."""
        llm = MockLLM(
            extractions=[_all_fields_extraction(), TurnExtraction()],
            replies=["What's your name?", "Great, here's a summary!"],
        )
        store = JsonFileStore(str(tmp_path))
        engine = _make_engine(llm, store)

        conv_id, greeting = await engine.start("en")
        assert greeting.done is False

        # Turn 1: provide all fields
        reply1 = await engine.handle_turn(
            conv_id,
            "I'm Maria Gonzalez, line cook, 4 years, grill station, weekday evenings and weekends, June 23rd 2026, yes",
            _turn("I'm Maria Gonzalez, line cook, 4 years, grill station, weekday evenings and weekends, June 23rd 2026, yes"),
        )
        assert reply1.done is False  # → CONFIRMING

        # Turn 2: confirm
        reply2 = await engine.handle_turn(conv_id, "Yes that's correct", _turn("Yes that's correct"))
        assert reply2.done is True  # → SUMMARY

    async def test_record_persisted_after_completion(self, tmp_path):
        """The conversation snapshot is saved with a non-None record after SUMMARY."""
        llm = MockLLM(
            extractions=[_all_fields_extraction(), TurnExtraction()],
            replies=["Collecting...", "Summary done!"],
        )
        store = JsonFileStore(str(tmp_path))
        engine = _make_engine(llm, store)

        conv_id, _ = await engine.start("en")
        await engine.handle_turn(conv_id, "All fields", _turn("All fields"))
        await engine.handle_turn(conv_id, "Confirmed", _turn("Confirmed"))

        snap = store.load(conv_id)
        assert snap is not None
        assert snap.state == ConversationState.SUMMARY
        assert snap.record is not None
        assert snap.record.candidate_name is not None
        assert snap.record.candidate_name.value == "Maria Gonzalez"

    async def test_summary_reviewer_output_contains_table_icons(self, tmp_path):
        """SUMMARY state: reviewer table (with icons) is in reviewer_output, not spoken text."""
        llm = MockLLM(
            extractions=[_all_fields_extraction(), TurnExtraction()],
            reply="Standard reply",
        )
        store = JsonFileStore(str(tmp_path))
        engine = _make_engine(llm, store)

        conv_id, _ = await engine.start("en")
        await engine.handle_turn(conv_id, "All fields", _turn("All fields"))
        summary_reply = await engine.handle_turn(conv_id, "Confirmed", _turn("Confirmed"))

        # Reviewer panel is backend-only — in reviewer_output, not in the spoken text
        assert summary_reply.reviewer_output is not None
        assert "✓" in summary_reply.reviewer_output or "⚠" in summary_reply.reviewer_output
        # Spoken text should NOT contain reviewer metadata
        assert "✓" not in summary_reply.text
        assert "Conf" not in summary_reply.text

    async def test_spanish_greeting(self, tmp_path):
        """Starting with lang=es uses the Spanish greeting."""
        llm = MockLLM(reply="¿Cuál es tu nombre?")
        store = JsonFileStore(str(tmp_path))
        engine = _make_engine(llm, store)

        _, greeting = await engine.start("es")
        # Spanish greeting should mention the agent's purpose in Spanish
        assert "¡Hola" in greeting.text or "Hola" in greeting.text

    async def test_transcript_includes_all_turns(self, tmp_path):
        """The persisted transcript has both agent and candidate turns."""
        llm = MockLLM(
            extractions=[_all_fields_extraction(), TurnExtraction()],
            reply="Noted",
        )
        store = JsonFileStore(str(tmp_path))
        engine = _make_engine(llm, store)

        conv_id, _ = await engine.start("en")
        await engine.handle_turn(conv_id, "Turn 1", _turn("Turn 1"))

        snap = store.load(conv_id)
        roles = [t.role for t in snap.transcript]
        assert "agent" in roles
        assert "candidate" in roles
