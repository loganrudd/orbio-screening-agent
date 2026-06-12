"""Unit tests for confidence computation, flag assignment, and rendering.

All pure functions — no LLM, no filesystem.
"""

import pytest

from agent.output import assign_flag, build_summary, compute_confidence, render_candidate_confirmation, render_reviewer_table
from agent.schemas import FieldFlag, ScreeningRecord

from helpers import make_field, make_record


# ──────────────────────────────────── compute_confidence ──────────────────────

class TestComputeConfidence:
    def test_validated_and_explicitly_stated(self):
        c = compute_confidence(validated=True, explicitly_stated=True)
        assert c.score == pytest.approx(0.9)

    def test_validated_inferred(self):
        c = compute_confidence(validated=True, explicitly_stated=False)
        assert c.score == pytest.approx(0.6)

    def test_not_validated(self):
        c = compute_confidence(validated=False, explicitly_stated=True)
        assert c.score == pytest.approx(0.3)

    def test_not_validated_inferred(self):
        c = compute_confidence(validated=False, explicitly_stated=False)
        assert c.score == pytest.approx(0.3)

    def test_stt_confidence_multiplied(self):
        c = compute_confidence(validated=True, explicitly_stated=True, stt_confidence=0.8)
        assert c.score == pytest.approx(0.9 * 0.8)

    def test_stt_confidence_on_inferred(self):
        c = compute_confidence(validated=True, explicitly_stated=False, stt_confidence=0.5)
        assert c.score == pytest.approx(0.6 * 0.5)

    def test_confirmed_threshold_met(self):
        c = compute_confidence(validated=True, explicitly_stated=True)
        assert c.score >= 0.8

    def test_confirmed_threshold_not_met_when_inferred(self):
        c = compute_confidence(validated=True, explicitly_stated=False)
        assert c.score < 0.8

    def test_stt_can_push_inferred_below_threshold(self):
        c = compute_confidence(validated=True, explicitly_stated=False, stt_confidence=0.5)
        assert c.score < 0.8


# ────────────────────────────────────── assign_flag ───────────────────────────

class TestAssignFlag:
    def test_conflicting_values_wins(self):
        field = make_field("server", conflicting_values=["host"])
        assert assign_flag(field, reprompt_capped=True) == FieldFlag.CONFLICTING

    def test_reprompt_capped_gives_needs_review(self):
        field = make_field("Maria")
        assert assign_flag(field, reprompt_capped=True) == FieldFlag.NEEDS_REVIEW

    def test_high_score_confirmed(self):
        field = make_field("Maria", validated=True, explicitly_stated=True)
        assert field.confidence.score >= 0.8
        assert assign_flag(field, reprompt_capped=False) == FieldFlag.CONFIRMED

    def test_low_score_needs_review(self):
        field = make_field("Maria", validated=True, explicitly_stated=False)  # score=0.6
        assert assign_flag(field, reprompt_capped=False) == FieldFlag.NEEDS_REVIEW

    def test_invalid_always_needs_review(self):
        field = make_field("Maria", validated=False, explicitly_stated=True)  # score=0.3
        assert assign_flag(field, reprompt_capped=False) == FieldFlag.NEEDS_REVIEW

    def test_stt_reduction_below_threshold(self):
        # validated+explicit but stt=0.5 → 0.45 < 0.8 → needs_review
        field = make_field("Maria", validated=True, explicitly_stated=True, stt_confidence=0.5)
        assert assign_flag(field, reprompt_capped=False) == FieldFlag.NEEDS_REVIEW

    def test_conflicting_beats_reprompt_cap(self):
        field = make_field("host", conflicting_values=["server"])
        result = assign_flag(field, reprompt_capped=True)
        assert result == FieldFlag.CONFLICTING


# ─────────────────────────────────── render_reviewer_table ────────────────────

class TestRenderReviewerTable:
    def test_contains_confirmed_icon(self):
        record = make_record(candidate_name=make_field("Maria"))
        table = render_reviewer_table(record)
        assert "✓" in table

    def test_contains_missing_icon_for_none_fields(self):
        record = ScreeningRecord()
        table = render_reviewer_table(record)
        assert "✗" in table

    def test_contains_needs_review_icon(self):
        field = make_field("Maria", validated=True, explicitly_stated=False, flag=FieldFlag.NEEDS_REVIEW)
        record = make_record(candidate_name=field)
        table = render_reviewer_table(record)
        assert "⚠" in table

    def test_contains_conflicting_icon(self):
        field = make_field("host", conflicting_values=["server"], flag=FieldFlag.CONFLICTING)
        record = make_record(position_applied_for=field)
        table = render_reviewer_table(record)
        assert "!" in table

    def test_contains_field_value(self):
        record = make_record(candidate_name=make_field("Maria Gonzalez"))
        table = render_reviewer_table(record)
        assert "Maria Gonzalez" in table

    def test_returns_nonempty_string(self):
        record = ScreeningRecord()
        table = render_reviewer_table(record)
        assert len(table) > 0

    def test_all_fields_present_in_table(self):
        record = ScreeningRecord()
        table = render_reviewer_table(record)
        # Human-readable labels should appear
        for label in ["Name", "Position", "Experience", "Skills", "Availability", "Start Date", "Work Auth"]:
            assert label in table


# ──────────────────────────────────────── build_summary ───────────────────────

class TestBuildSummary:
    def test_returns_nonempty_string(self):
        record = ScreeningRecord()
        summary = build_summary(record)
        assert len(summary) > 0

    def test_contains_candidate_name(self):
        record = make_record(candidate_name=make_field("Sofia Ramirez"))
        summary = build_summary(record)
        assert "Sofia Ramirez" in summary

    def test_missing_summary_section_present(self):
        record = ScreeningRecord()
        summary = build_summary(record)
        assert "Missing" in summary or "missing" in summary

    def test_confirmed_count_in_summary(self):
        record = make_record(
            candidate_name=make_field("Maria"),
            position_applied_for=make_field("server"),
        )
        summary = build_summary(record)
        assert "Confirmed" in summary

    def test_full_record_no_missing(self):
        record = make_record(
            candidate_name=make_field("Maria"),
            position_applied_for=make_field("server"),
            years_experience=make_field(4),
            relevant_skills=make_field(["grill"]),
            availability=make_field(["weekday_evening"]),
            earliest_start_date=make_field("2026-06-23"),
            work_authorization=make_field(True),
        )
        summary = build_summary(record)
        # With all fields confirmed, "Missing:" should not appear (or list should be empty)
        # The summary line format is "Missing: <list>", only present when fields are missing
        assert "7/7" in summary or "Missing: " not in summary


# ────────────────────────── render_candidate_confirmation (language) ──────────


class TestRenderCandidateConfirmation:
    def _full_record(self):
        return make_record(
            candidate_name=make_field("Jane Doe"),
            position_applied_for=make_field("server"),
            years_experience=make_field(2),
            relevant_skills=make_field(["service"]),
            availability=make_field(["weekday_day"]),
            earliest_start_date=make_field("immediate"),
            work_authorization=make_field(True),
        )

    def test_en_default_contains_english(self):
        text = render_candidate_confirmation(self._full_record())
        assert "Jane Doe" in text
        assert "server" in text

    def test_en_explicit_contains_english(self):
        text = render_candidate_confirmation(self._full_record(), "en")
        assert "right away" in text  # EN frame for "immediate"

    def test_es_contains_spanish(self):
        text = render_candidate_confirmation(self._full_record(), "es")
        assert "Jane Doe" in text
        assert "mesero" in text.lower()
        assert "inmediato" in text.lower()

    def test_es_years_in_spanish(self):
        text = render_candidate_confirmation(self._full_record(), "es")
        assert "años" in text
