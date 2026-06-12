"""Unit tests for the conversation state machine and turn loop.

The LLM and storage are mocked or use JsonFileStore(tmp_path).
Tests cover: state transitions, outstanding-field logic, re-prompt cap,
conflicting-value detection across turns, and completion.
"""

import datetime

import pytest

from agent.conversation import MAX_REPROMPTS_PER_FIELD, ConversationEngine
from agent.extraction import Extractor
from agent.schemas import (
    ConversationState,
    FieldFlag,
    ScreeningRecord,
    StringProposal,
    IntProposal,
    StringListProposal,
    BoolProposal,
    TurnExtraction,
)
from agent.storage import ConversationSnapshot, JsonFileStore, Turn

from helpers import MockLLM, make_field, make_record


# ─────────────────────────────────────── helpers ──────────────────────────────

def _turn(content: str) -> Turn:
    return Turn(
        role="candidate",
        content=content,
        ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def _make_engine(llm: MockLLM, store: JsonFileStore) -> ConversationEngine:
    extractor = Extractor(llm=llm)
    return ConversationEngine(store=store, llm=llm, extractor=extractor)


def _full_record() -> ScreeningRecord:
    """A record with all required fields filled (drives COLLECTING → CONFIRMING)."""
    return ScreeningRecord(
        candidate_name=make_field("Maria"),
        position_applied_for=make_field("server"),
        years_experience=make_field(4),
        relevant_skills=make_field(["grill"]),
        availability=make_field(["weekday_evening"]),
        earliest_start_date=make_field("2026-06-23"),
        work_authorization=make_field(True),
    )


# ─────────────────────────── _outstanding_fields ──────────────────────────────

class TestOutstandingFields:
    def _engine(self, tmp_path):
        llm = MockLLM()
        store = JsonFileStore(str(tmp_path))
        return ConversationEngine(store=store, llm=llm, extractor=Extractor(llm=llm))

    def _snapshot(self, tmp_path, record=None, reprompt_counts=None):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        snap.state = ConversationState.COLLECTING
        snap.record = record or ScreeningRecord()
        snap.reprompt_counts = reprompt_counts or {}
        return snap

    def test_all_required_fields_missing_all_outstanding(self, tmp_path):
        engine = self._engine(tmp_path)
        snap = self._snapshot(tmp_path)
        outstanding = engine._outstanding_fields(snap)
        assert outstanding == ScreeningRecord.required_fields()

    def test_collected_field_excluded(self, tmp_path):
        engine = self._engine(tmp_path)
        snap = self._snapshot(tmp_path, record=ScreeningRecord(candidate_name=make_field("Maria")))
        outstanding = engine._outstanding_fields(snap)
        assert "candidate_name" not in outstanding

    def test_reprompt_capped_field_excluded(self, tmp_path):
        engine = self._engine(tmp_path)
        counts = {"years_experience": MAX_REPROMPTS_PER_FIELD + 1}
        snap = self._snapshot(tmp_path, reprompt_counts=counts)
        outstanding = engine._outstanding_fields(snap)
        assert "years_experience" not in outstanding

    def test_field_at_cap_boundary_excluded(self, tmp_path):
        engine = self._engine(tmp_path)
        # count == MAX+1 => excluded; count == MAX => still outstanding
        counts = {"years_experience": MAX_REPROMPTS_PER_FIELD + 1}
        snap = self._snapshot(tmp_path, reprompt_counts=counts)
        outstanding = engine._outstanding_fields(snap)
        assert "years_experience" not in outstanding

    def test_all_required_collected_empty_outstanding(self, tmp_path):
        engine = self._engine(tmp_path)
        snap = self._snapshot(tmp_path, record=_full_record())
        outstanding = engine._outstanding_fields(snap)
        assert outstanding == []

    def test_location_preference_never_outstanding(self, tmp_path):
        engine = self._engine(tmp_path)
        snap = self._snapshot(tmp_path)
        outstanding = engine._outstanding_fields(snap)
        assert "location_preference" not in outstanding


# ─────────────────────────────────── _next_state ──────────────────────────────

class TestNextState:
    def _engine(self, tmp_path):
        llm = MockLLM()
        store = JsonFileStore(str(tmp_path))
        return ConversationEngine(store=store, llm=llm, extractor=Extractor(llm=llm))

    def _snap(self, tmp_path, state):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        snap.state = state
        snap.record = ScreeningRecord()
        return snap

    def test_collecting_with_outstanding_stays_collecting(self, tmp_path):
        engine = self._engine(tmp_path)
        snap = self._snap(tmp_path, ConversationState.COLLECTING)
        result = engine._next_state(snap, ["candidate_name"])
        assert result == ConversationState.COLLECTING

    def test_collecting_no_outstanding_advances_to_confirming(self, tmp_path):
        engine = self._engine(tmp_path)
        snap = self._snap(tmp_path, ConversationState.COLLECTING)
        result = engine._next_state(snap, [])
        assert result == ConversationState.CONFIRMING

    def test_confirming_advances_to_summary(self, tmp_path):
        engine = self._engine(tmp_path)
        snap = self._snap(tmp_path, ConversationState.CONFIRMING)
        result = engine._next_state(snap, [])
        assert result == ConversationState.SUMMARY

    def test_summary_stays_summary(self, tmp_path):
        engine = self._engine(tmp_path)
        snap = self._snap(tmp_path, ConversationState.SUMMARY)
        result = engine._next_state(snap, [])
        assert result == ConversationState.SUMMARY


# ─────────────────────────── handle_turn — full flow ──────────────────────────

class TestHandleTurnFlow:
    async def test_greeting_transitions_to_collecting(self, tmp_path):
        llm = MockLLM(extraction=TurnExtraction(), reply="What position?")
        engine = _make_engine(llm, JsonFileStore(str(tmp_path)))
        conv_id, _ = await engine.start("en")

        reply = await engine.handle_turn(conv_id, "Hi, I'm Maria", _turn("Hi, I'm Maria"))

        snap = JsonFileStore(str(tmp_path)).load(conv_id)
        assert snap.state == ConversationState.COLLECTING
        assert reply.done is False

    async def test_all_fields_provided_reaches_done(self, tmp_path):
        """Scripted conversation: extraction fills all fields then confirms → done=True."""
        # All required fields in one extraction call
        full_extraction = TurnExtraction(
            candidate_name=StringProposal(value="Maria", source_text="Maria", explicitly_stated=True),
            position_applied_for=StringProposal(value="server", source_text="server", explicitly_stated=True),
            years_experience=IntProposal(value=4, source_text="4 years", explicitly_stated=True),
            relevant_skills=StringListProposal(value=["grill"], source_text="grill", explicitly_stated=True),
            availability=StringListProposal(value=["weekday_evening"], source_text="weekday evenings", explicitly_stated=True),
            earliest_start_date=StringProposal(value="2026-06-23", source_text="June 23", explicitly_stated=True),
            work_authorization=BoolProposal(value=True, source_text="yes authorized", explicitly_stated=True),
        )
        # Turn 1: extract all fields → COLLECTING→CONFIRMING
        # Turn 2: confirmation → SUMMARY (done)
        llm = MockLLM(
            extractions=[full_extraction, TurnExtraction()],
            replies=["Great, confirming...", "Thank you!"],
        )
        engine = _make_engine(llm, JsonFileStore(str(tmp_path)))
        conv_id, _ = await engine.start("en")

        # Turn 1 — provide all fields
        reply1 = await engine.handle_turn(
            conv_id,
            "I'm Maria, server, 4 years, grill, weekday evenings, June 23, yes authorized",
            _turn("I'm Maria, server, 4 years, grill, weekday evenings, June 23, yes authorized"),
        )
        assert reply1.done is False

        # Turn 2 — confirm
        reply2 = await engine.handle_turn(conv_id, "Looks correct", _turn("Looks correct"))
        assert reply2.done is True

    async def test_missing_field_stays_missing_after_cap(self, tmp_path):
        """A never-stated required field ends up missing, not invented."""
        # Empty extraction every call (nothing extracted)
        llm = MockLLM(extraction=TurnExtraction(), reply="What's your experience?")
        engine = _make_engine(llm, JsonFileStore(str(tmp_path)))
        conv_id, _ = await engine.start("en")

        # Drive enough turns to exhaust the reprompt cap for years_experience
        for i in range(MAX_REPROMPTS_PER_FIELD + 3):
            await engine.handle_turn(conv_id, "I'm not sure", _turn("I'm not sure"))

        snap = JsonFileStore(str(tmp_path)).load(conv_id)
        # The field must remain None — not invented
        if snap.record:
            assert snap.record.years_experience is None

    async def test_reprompt_counts_persisted(self, tmp_path):
        llm = MockLLM(extraction=TurnExtraction(), reply="Next question?")
        engine = _make_engine(llm, JsonFileStore(str(tmp_path)))
        conv_id, _ = await engine.start("en")

        await engine.handle_turn(conv_id, "something", _turn("something"))

        snap = JsonFileStore(str(tmp_path)).load(conv_id)
        # At least one reprompt count should have been incremented
        assert any(v > 0 for v in snap.reprompt_counts.values())

    async def test_unknown_conversation_raises(self, tmp_path):
        llm = MockLLM()
        engine = _make_engine(llm, JsonFileStore(str(tmp_path)))
        with pytest.raises(ValueError, match="Unknown conversation"):
            await engine.handle_turn("nonexistent-id", "hi", _turn("hi"))

    async def test_conflict_detected_across_turns(self, tmp_path):
        """Candidate gives position=server in turn 1, position=host in turn 2 → CONFLICTING."""
        extraction1 = TurnExtraction(
            position_applied_for=StringProposal(value="server", source_text="server", explicitly_stated=True),
        )
        extraction2 = TurnExtraction(
            position_applied_for=StringProposal(value="host", source_text="actually host", explicitly_stated=True),
        )
        llm = MockLLM(
            extractions=[extraction1, extraction2, TurnExtraction()],
            reply="Got it",
        )
        engine = _make_engine(llm, JsonFileStore(str(tmp_path)))
        conv_id, _ = await engine.start("en")

        await engine.handle_turn(conv_id, "I want server", _turn("I want server"))
        await engine.handle_turn(conv_id, "Actually host", _turn("Actually host"))

        snap = JsonFileStore(str(tmp_path)).load(conv_id)
        assert snap.record is not None
        pos = snap.record.position_applied_for
        assert pos is not None
        assert pos.flag == FieldFlag.CONFLICTING

    async def test_start_creates_persisted_snapshot(self, tmp_path):
        llm = MockLLM()
        store = JsonFileStore(str(tmp_path))
        engine = _make_engine(llm, store)
        conv_id, reply = await engine.start("en")

        snap = store.load(conv_id)
        assert snap is not None
        assert snap.state == ConversationState.GREETING
        assert reply.done is False
        assert len(reply.text) > 0


# ────────────────────────── multilingual / language detection ─────────────────


@pytest.mark.asyncio
class TestLanguageDetection:
    async def test_explicit_en_preserved(self, tmp_path):
        """Explicit --lang en with auto_detect=False must never switch language."""
        llm = MockLLM(reply="Got it.")
        store = JsonFileStore(str(tmp_path))
        engine = _make_engine(llm, store)
        conv_id, _ = await engine.start("en", auto_detect=False)

        # First turn is Spanish — should NOT trigger switch
        await engine.handle_turn(conv_id, "Me llamo Sofia", _turn("Me llamo Sofia"))

        snap = store.load(conv_id)
        assert snap.language == "en"

    async def test_auto_detect_switches_to_es_on_first_turn(self, tmp_path):
        """auto_detect=True must switch language when ES is detected on the first turn."""
        llm = MockLLM(reply="Entendido.")
        store = JsonFileStore(str(tmp_path))
        engine = _make_engine(llm, store)
        conv_id, _ = await engine.start("en", auto_detect=True)

        # First substantive turn in Spanish
        await engine.handle_turn(
            conv_id,
            "Me llamo Sofia y quiero trabajar como mesera",
            _turn("Me llamo Sofia y quiero trabajar como mesera"),
        )

        snap = store.load(conv_id)
        assert snap.language == "es"

    async def test_auto_detect_keeps_en_for_english_turn(self, tmp_path):
        """auto_detect=True must not switch to ES when the first turn is English."""
        llm = MockLLM(reply="Got it.")
        store = JsonFileStore(str(tmp_path))
        engine = _make_engine(llm, store)
        conv_id, _ = await engine.start("en", auto_detect=True)

        await engine.handle_turn(
            conv_id,
            "My name is Jane and I want to apply for a server position",
            _turn("My name is Jane and I want to apply for a server position"),
        )

        snap = store.load(conv_id)
        assert snap.language == "en"

    async def test_detection_only_runs_once(self, tmp_path):
        """Language must not flip on subsequent turns after the first."""
        llm = MockLLM(reply="Got it.")
        store = JsonFileStore(str(tmp_path))
        engine = _make_engine(llm, store)
        conv_id, _ = await engine.start("en", auto_detect=True)

        # First turn: ES → switches to "es"
        await engine.handle_turn(conv_id, "Me llamo Sofia", _turn("Me llamo Sofia"))
        snap_after_first = store.load(conv_id)
        assert snap_after_first.language == "es"
        assert snap_after_first.state == ConversationState.COLLECTING

        # Second turn in English: language must remain "es"
        await engine.handle_turn(conv_id, "I want to apply", _turn("I want to apply"))
        snap_after_second = store.load(conv_id)
        assert snap_after_second.language == "es"
