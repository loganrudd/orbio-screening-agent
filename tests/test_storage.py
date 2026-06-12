"""Unit tests for JsonFileStore: round-trip persistence, atomic write, and snapshot integrity."""

import json
import os

import pytest

from agent.schemas import ConversationState, ScreeningRecord
from agent.storage import ConversationSnapshot, JsonFileStore, Turn

from helpers import make_field, make_record


# ─────────────────────────── new_conversation ─────────────────────────────────

class TestNewConversation:
    def test_creates_snapshot_file(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".json"

    def test_snapshot_starts_in_greeting_state(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        assert snap.state == ConversationState.GREETING

    def test_snapshot_has_unique_id(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap1 = store.new_conversation("en")
        snap2 = store.new_conversation("en")
        assert snap1.conversation_id != snap2.conversation_id

    def test_language_preserved(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("es")
        assert snap.language == "es"

    def test_auto_detect_default_false(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        assert snap.auto_detect is False

    def test_auto_detect_true_preserved(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en", auto_detect=True)
        assert snap.auto_detect is True


# ─────────────────────────────── save / load ──────────────────────────────────

class TestSaveLoad:
    def test_load_returns_none_for_missing_id(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        assert store.load("nonexistent") is None

    def test_basic_round_trip(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        snap.state = ConversationState.COLLECTING
        snap.reprompt_counts["candidate_name"] = 1
        store.save(snap)

        loaded = store.load(snap.conversation_id)
        assert loaded is not None
        assert loaded.state == ConversationState.COLLECTING
        assert loaded.reprompt_counts["candidate_name"] == 1

    def test_transcript_round_trip(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        snap.transcript.append(Turn(role="candidate", content="Hi", ts="2026-01-01T00:00:00+00:00"))
        store.save(snap)

        loaded = store.load(snap.conversation_id)
        assert len(loaded.transcript) == 1
        assert loaded.transcript[0].content == "Hi"
        assert loaded.transcript[0].role == "candidate"

    def test_record_with_fields_round_trips(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        snap.record = make_record(
            candidate_name=make_field("Maria Gonzalez"),
            years_experience=make_field(4),
            work_authorization=make_field(True),
        )
        store.save(snap)

        loaded = store.load(snap.conversation_id)
        assert loaded.record is not None
        assert loaded.record.candidate_name is not None
        assert loaded.record.candidate_name.value == "Maria Gonzalez"
        assert loaded.record.years_experience.value == 4
        assert loaded.record.work_authorization.value is True

    def test_confidence_score_computed_field_survives_round_trip(self, tmp_path):
        """Confidence.score is a @computed_field; serialization must not break it."""
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        snap.record = make_record(candidate_name=make_field("Maria", validated=True, explicitly_stated=True))
        store.save(snap)

        loaded = store.load(snap.conversation_id)
        assert loaded.record.candidate_name.confidence.score == pytest.approx(0.9)

    def test_reprompt_counts_round_trip(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        snap.reprompt_counts = {"years_experience": 2, "availability": 1}
        store.save(snap)

        loaded = store.load(snap.conversation_id)
        assert loaded.reprompt_counts == {"years_experience": 2, "availability": 1}

    def test_auto_detect_round_trip(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en", auto_detect=True)
        store.save(snap)
        loaded = store.load(snap.conversation_id)
        assert loaded.auto_detect is True

    def test_summary_round_trip(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        snap.summary = "Candidate looks strong."
        store.save(snap)

        loaded = store.load(snap.conversation_id)
        assert loaded.summary == "Candidate looks strong."

    def test_audio_timestamps_round_trip(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        snap.transcript.append(
            Turn(role="candidate", content="Hi", ts="2026-01-01T00:00:00+00:00",
                 audio_start_s=1.5, audio_end_s=2.3)
        )
        store.save(snap)

        loaded = store.load(snap.conversation_id)
        turn = loaded.transcript[0]
        assert turn.audio_start_s == pytest.approx(1.5)
        assert turn.audio_end_s == pytest.approx(2.3)


# ─────────────────────────── atomic write ─────────────────────────────────────

class TestAtomicWrite:
    def test_file_is_valid_json_after_save(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        snap.state = ConversationState.COLLECTING
        store.save(snap)

        path = tmp_path / f"{snap.conversation_id}.json"
        with open(path) as f:
            data = json.load(f)
        assert data["conversation_id"] == snap.conversation_id

    def test_no_tmp_files_left_after_save(self, tmp_path):
        store = JsonFileStore(str(tmp_path))
        snap = store.new_conversation("en")
        store.save(snap)

        tmp_files = [f for f in tmp_path.iterdir() if f.suffix == ".tmp"]
        assert tmp_files == []
