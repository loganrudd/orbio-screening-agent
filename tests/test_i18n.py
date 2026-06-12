"""Tests for agent/i18n.py: language detection, string tables, confirmation prose, tts_voice."""

from __future__ import annotations

import pytest

from agent.i18n import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    build_confirmation,
    detect_language,
    fallback,
    get_strings,
    greeting,
    tts_voice,
)

from tests.helpers import make_field, make_record


# --------------------------------------------------------------------------- detection


class TestDetectLanguage:
    def test_detects_english(self):
        assert detect_language("Hello my name is John and I want to apply for a job") == "en"

    def test_detects_spanish(self):
        assert detect_language("Me llamo Sofia y quiero trabajar como mesera en el restaurante") == "es"

    def test_empty_string_falls_back(self):
        assert detect_language("") == DEFAULT_LANGUAGE

    def test_whitespace_only_falls_back(self):
        assert detect_language("   ") == DEFAULT_LANGUAGE

    def test_returns_supported_code(self):
        result = detect_language("any input")
        assert result in SUPPORTED_LANGUAGES

    def test_spanish_restaurant_turn(self):
        # Realistic short turn from the ES seed
        assert detect_language("Tengo cinco años de experiencia en restaurantes.") == "es"

    def test_english_restaurant_turn(self):
        assert detect_language("I have five years of experience in restaurants.") == "en"


# --------------------------------------------------------------------------- string tables


class TestGetStrings:
    def test_en_table_present(self):
        s = get_strings("en")
        assert "name" in s.greeting.lower() or "restaurant" in s.greeting.lower()

    def test_es_table_present(self):
        s = get_strings("es")
        assert "nombre" in s.greeting.lower() or "restaurante" in s.greeting.lower()

    def test_unknown_language_falls_back_to_en(self):
        s = get_strings("fr")
        s_en = get_strings("en")
        assert s.greeting == s_en.greeting

    def test_table_completeness(self):
        """Every language table must have the same set of fields as the EN baseline."""
        en = get_strings("en")
        en_fields = {f.name for f in en.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        for lang in SUPPORTED_LANGUAGES:
            s = get_strings(lang)
            s_fields = {f.name for f in s.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            assert s_fields == en_fields, f"Language '{lang}' missing fields: {en_fields - s_fields}"

    def test_avail_display_has_all_shifts(self):
        shifts = {"weekday_day", "weekday_evening", "weekend_day", "weekend_evening"}
        for lang in SUPPORTED_LANGUAGES:
            s = get_strings(lang)
            assert shifts <= set(s.avail_display.keys()), f"avail_display incomplete for '{lang}'"

    def test_position_display_has_all_positions(self):
        positions = {"server", "line_cook", "host", "shift_manager", "other"}
        for lang in SUPPORTED_LANGUAGES:
            s = get_strings(lang)
            assert positions <= set(s.position_display.keys()), f"position_display incomplete for '{lang}'"


class TestGreetingClosing:
    def test_en_greeting_is_english(self):
        g = greeting("en")
        assert "name" in g.lower() or "restaurant" in g.lower()

    def test_es_greeting_is_spanish(self):
        g = greeting("es")
        assert "nombre" in g.lower() or "restaurante" in g.lower()

    def test_en_closing_is_english(self):
        from agent.i18n import closing
        c = closing("en")
        assert "thank" in c.lower()

    def test_es_closing_is_spanish(self):
        from agent.i18n import closing
        c = closing("es")
        assert "gracias" in c.lower()

    def test_fallback_returns_string(self):
        for lang in SUPPORTED_LANGUAGES:
            assert isinstance(fallback(lang), str)
            assert fallback(lang)


# --------------------------------------------------------------------------- tts_voice


class TestTtsVoice:
    def test_en_returns_en_voice(self):
        v = tts_voice("en")
        assert v.endswith("-en") or "en" in v

    def test_es_returns_es_voice(self):
        v = tts_voice("es")
        assert v.endswith("-es") or "es" in v

    def test_unknown_language_returns_en_voice(self):
        v = tts_voice("fr")
        assert "en" in v

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_TTS_MODEL", "aura-custom-model")
        v = tts_voice("es")
        assert v == "aura-custom-model"

    def test_env_override_cleared(self, monkeypatch):
        monkeypatch.delenv("DEEPGRAM_TTS_MODEL", raising=False)
        # Should return language-specific voice when override absent
        v = tts_voice("es")
        assert v != "aura-asteria-en"


# --------------------------------------------------------------------------- confirmation prose (EN)


class TestBuildConfirmationEN:
    def _full_record(self):
        return make_record(
            candidate_name=make_field("Jane Doe"),
            position_applied_for=make_field("server"),
            years_experience=make_field(3),
            relevant_skills=make_field(["customer service", "POS systems"]),
            availability=make_field(["weekday_evening", "weekend_day"]),
            earliest_start_date=make_field("2025-07-01"),
            work_authorization=make_field(True),
            location_preference=make_field("Downtown"),
        )

    def test_name_present(self):
        rec = self._full_record()
        text = build_confirmation(rec, "en")
        assert "Jane Doe" in text

    def test_position_present(self):
        rec = self._full_record()
        text = build_confirmation(rec, "en")
        assert "server" in text

    def test_experience_present(self):
        rec = self._full_record()
        text = build_confirmation(rec, "en")
        assert "3 years" in text

    def test_skills_present(self):
        rec = self._full_record()
        text = build_confirmation(rec, "en")
        assert "customer service" in text

    def test_availability_localized(self):
        rec = self._full_record()
        text = build_confirmation(rec, "en")
        assert "weekday evening" in text
        assert "weekend day" in text

    def test_start_date_present(self):
        rec = self._full_record()
        text = build_confirmation(rec, "en")
        assert "2025-07-01" in text

    def test_immediate_start(self):
        rec = make_record(
            candidate_name=make_field("Joe"),
            position_applied_for=make_field("host"),
            years_experience=make_field(1),
            relevant_skills=make_field(["communication"]),
            availability=make_field(["weekday_day"]),
            earliest_start_date=make_field("immediate"),
            work_authorization=make_field(True),
        )
        text = build_confirmation(rec, "en")
        assert "right away" in text

    def test_work_auth_yes(self):
        rec = self._full_record()
        text = build_confirmation(rec, "en")
        assert "authorized" in text.lower()

    def test_work_auth_no(self):
        rec = make_record(
            candidate_name=make_field("Bob"),
            position_applied_for=make_field("line_cook"),
            years_experience=make_field(2),
            relevant_skills=make_field(["grilling"]),
            availability=make_field(["weekday_day"]),
            earliest_start_date=make_field("immediate"),
            work_authorization=make_field(False),
        )
        text = build_confirmation(rec, "en")
        assert "not currently authorized" in text.lower()

    def test_missing_fields_mentioned(self):
        rec = make_record(
            candidate_name=make_field("Alice"),
            position_applied_for=make_field("host"),
        )
        text = build_confirmation(rec, "en")
        assert "capture" in text.lower() or "able" in text.lower()

    def test_single_year_singular(self):
        rec = make_record(
            candidate_name=make_field("Pat"),
            position_applied_for=make_field("server"),
            years_experience=make_field(1),
            relevant_skills=make_field(["service"]),
            availability=make_field(["weekday_day"]),
            earliest_start_date=make_field("immediate"),
            work_authorization=make_field(True),
        )
        text = build_confirmation(rec, "en")
        assert "1 year" in text
        assert "1 years" not in text

    def test_location_present(self):
        rec = self._full_record()
        text = build_confirmation(rec, "en")
        assert "Downtown" in text


# --------------------------------------------------------------------------- confirmation prose (ES)


class TestBuildConfirmationES:
    def _full_record(self):
        return make_record(
            candidate_name=make_field("Sofia Ramirez"),
            position_applied_for=make_field("server"),
            years_experience=make_field(5),
            relevant_skills=make_field(["servicio al cliente", "caja registradora"]),
            availability=make_field(["weekday_evening", "weekend_day"]),
            earliest_start_date=make_field("immediate"),
            work_authorization=make_field(True),
        )

    def test_name_present(self):
        text = build_confirmation(self._full_record(), "es")
        assert "Sofia Ramirez" in text

    def test_position_display_localized(self):
        text = build_confirmation(self._full_record(), "es")
        assert "mesero" in text.lower()

    def test_experience_spanish_unit(self):
        text = build_confirmation(self._full_record(), "es")
        assert "años" in text

    def test_immediate_start_spanish(self):
        text = build_confirmation(self._full_record(), "es")
        assert "inmediato" in text

    def test_availability_spanish(self):
        text = build_confirmation(self._full_record(), "es")
        assert "semana" in text.lower()

    def test_workauth_spanish(self):
        text = build_confirmation(self._full_record(), "es")
        assert "autorizado" in text.lower()

    def test_list_join_uses_y(self):
        rec = make_record(
            candidate_name=make_field("Ana"),
            position_applied_for=make_field("server"),
            years_experience=make_field(2),
            relevant_skills=make_field(["a", "b"]),
            availability=make_field(["weekday_day", "weekday_evening"]),
            earliest_start_date=make_field("immediate"),
            work_authorization=make_field(True),
        )
        text = build_confirmation(rec, "es")
        assert " y " in text
