"""Unit tests for validator helpers and per-field normalization in extraction.py.

Tests are pure-function: no LLM, no filesystem, no network.
"""

import pytest

from agent.extraction import _process_field
from agent.schemas import (
    validate_availability,
    validate_position,
    validate_start_date,
    validate_years_experience,
)


# ─────────────────────────────────────── validate_years_experience ────────────

class TestValidateYearsExperience:
    def test_zero(self):
        assert validate_years_experience(0) is True

    def test_valid_int(self):
        assert validate_years_experience(4) is True

    def test_upper_bound(self):
        assert validate_years_experience(60) is True

    def test_above_upper_bound(self):
        assert validate_years_experience(61) is False

    def test_negative(self):
        assert validate_years_experience(-1) is False

    def test_string_int(self):
        # Strings that parse as ints should work
        assert validate_years_experience("5") is True

    def test_float_string(self):
        assert validate_years_experience("3.5") is False

    def test_none(self):
        assert validate_years_experience(None) is False

    def test_non_numeric_string(self):
        assert validate_years_experience("lots") is False


# ─────────────────────────────────────────── validate_position ────────────────

class TestValidatePosition:
    @pytest.mark.parametrize("pos", ["server", "line_cook", "host", "shift_manager", "other"])
    def test_valid_positions(self, pos):
        assert validate_position(pos) is True

    def test_invalid_string(self):
        assert validate_position("bartender") is False

    def test_non_string(self):
        assert validate_position(42) is False

    def test_none(self):
        assert validate_position(None) is False

    def test_empty_string(self):
        assert validate_position("") is False


# ─────────────────────────────────────────── validate_availability ────────────

class TestValidateAvailability:
    def test_single_valid_shift(self):
        assert validate_availability(["weekday_day"]) is True

    def test_multiple_valid_shifts(self):
        assert validate_availability(["weekday_evening", "weekend_day", "weekend_evening"]) is True

    def test_all_shifts(self):
        shifts = ["weekday_day", "weekday_evening", "weekend_day", "weekend_evening"]
        assert validate_availability(shifts) is True

    def test_invalid_shift(self):
        assert validate_availability(["night_shift"]) is False

    def test_mixed_valid_invalid(self):
        assert validate_availability(["weekday_day", "night_shift"]) is False

    def test_empty_list(self):
        assert validate_availability([]) is False

    def test_non_list(self):
        assert validate_availability("weekday_day") is False

    def test_none(self):
        assert validate_availability(None) is False


# ─────────────────────────────────────────── validate_start_date ─────────────

class TestValidateStartDate:
    def test_iso_date(self):
        assert validate_start_date("2026-06-23") == "2026-06-23"

    def test_invalid_date(self):
        assert validate_start_date("2026-13-40") is None

    def test_non_string(self):
        assert validate_start_date(20260623) is None

    def test_none(self):
        assert validate_start_date(None) is None

    @pytest.mark.parametrize("sentinel", [
        "immediate", "asap", "now", "right away", "immediately", "today",
        "ASAP", "Now", "Immediately", "RIGHT AWAY",  # case-insensitive
        "  asap  ",  # whitespace
    ])
    def test_immediate_sentinels(self, sentinel):
        assert validate_start_date(sentinel) == "immediate"

    def test_junk_string(self):
        assert validate_start_date("some day soon") is None


# ───────────────────────────────────── _process_field per-field ───────────────

class TestProcessField:
    def test_candidate_name_valid(self):
        val, ok = _process_field("candidate_name", "Maria Gonzalez")
        assert val == "Maria Gonzalez"
        assert ok is True

    def test_candidate_name_empty(self):
        val, ok = _process_field("candidate_name", "")
        assert val is None
        assert ok is False

    def test_candidate_name_whitespace_only(self):
        val, ok = _process_field("candidate_name", "   ")
        assert val is None
        assert ok is False

    def test_position_valid(self):
        val, ok = _process_field("position_applied_for", "server")
        assert val == "server"
        assert ok is True

    def test_position_invalid(self):
        val, ok = _process_field("position_applied_for", "dishwasher")
        # value may be returned (for display) but validated=False
        assert ok is False

    def test_years_experience_valid(self):
        val, ok = _process_field("years_experience", 4)
        assert val == 4
        assert ok is True

    def test_years_experience_out_of_range(self):
        val, ok = _process_field("years_experience", 99)
        assert val is not None  # int parsed
        assert ok is False

    def test_years_experience_non_numeric(self):
        val, ok = _process_field("years_experience", "lots")
        assert val is None
        assert ok is False

    def test_relevant_skills_valid(self):
        val, ok = _process_field("relevant_skills", ["grill station", "prep"])
        assert val == ["grill station", "prep"]
        assert ok is True

    def test_relevant_skills_empty_list(self):
        val, ok = _process_field("relevant_skills", [])
        assert val is None
        assert ok is False

    def test_relevant_skills_non_list(self):
        val, ok = _process_field("relevant_skills", "grill")
        assert val is None
        assert ok is False

    def test_relevant_skills_strips_blanks(self):
        val, ok = _process_field("relevant_skills", ["grill", "", "  prep  "])
        assert ok is True
        assert "grill" in val
        assert "prep" in val

    def test_availability_valid(self):
        val, ok = _process_field("availability", ["weekday_evening", "weekend_day"])
        assert ok is True

    def test_availability_invalid(self):
        val, ok = _process_field("availability", ["night_shift"])
        assert ok is False

    def test_start_date_immediate(self):
        val, ok = _process_field("earliest_start_date", "asap")
        assert val == "immediate"
        assert ok is True

    def test_start_date_iso(self):
        val, ok = _process_field("earliest_start_date", "2026-07-01")
        assert val == "2026-07-01"
        assert ok is True

    def test_start_date_invalid(self):
        val, ok = _process_field("earliest_start_date", "whenever")
        assert val is None
        assert ok is False

    def test_work_authorization_true(self):
        val, ok = _process_field("work_authorization", True)
        assert val is True
        assert ok is True

    def test_work_authorization_false(self):
        val, ok = _process_field("work_authorization", False)
        assert val is False
        assert ok is True

    def test_work_authorization_non_bool(self):
        val, ok = _process_field("work_authorization", "yes")
        assert val is None
        assert ok is False

    def test_location_preference_valid(self):
        val, ok = _process_field("location_preference", "downtown")
        assert val == "downtown"
        assert ok is True

    def test_location_preference_empty(self):
        val, ok = _process_field("location_preference", "")
        assert val is None
        assert ok is False
