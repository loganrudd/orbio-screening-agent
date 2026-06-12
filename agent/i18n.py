"""Internationalization helpers for the screening agent.

All candidate-facing strings live here. Adding a new language means adding one new
entry to _STRINGS and _TTS_VOICES — no engine code changes required.

Detection uses py3langid (offline, deterministic, no seed needed). Language is always
stored as a canonical code ("en", "es"); anything unrecognized falls back to "en".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import ScreeningRecord

SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "es")
DEFAULT_LANGUAGE = "en"


# --------------------------------------------------------------------------- detection


def detect_language(text: str) -> str:
    """Return a supported language code detected from *text*, or DEFAULT_LANGUAGE.

    Uses py3langid for offline, deterministic classification. Falls back to
    DEFAULT_LANGUAGE if the library is unavailable or the detected code is unsupported.
    """
    if not text or not text.strip():
        return DEFAULT_LANGUAGE
    try:
        from py3langid import classify  # type: ignore[import-untyped]

        code, _score = classify(text)
        if code in SUPPORTED_LANGUAGES:
            return code
    except Exception:
        pass
    return DEFAULT_LANGUAGE


# --------------------------------------------------------------------------- string tables


@dataclass(frozen=True)
class _Strings:
    greeting: str
    confirming_intro: str
    confirming_outro: str
    closing: str
    fallback: str
    # Availability display map: canonical key → display phrase
    avail_display: dict[str, str]
    # Position display map: canonical key → display phrase
    position_display: dict[str, str]
    # Pluralisation: (singular, plural) for "year(s) of experience"
    year_singular: str
    year_plural: str
    year_phrase: str          # "{n} {unit} of experience"
    # List join word (e.g. "and" / "y")
    list_join: str
    # Sentence frames for confirmation (see build_confirmation)
    frame_name_pos: str       # "{name}, applying for {position}"
    frame_name: str
    frame_pos: str
    frame_exp_skills: str     # "{yrs} of experience, skills in {skills}"
    frame_exp: str
    frame_skills: str
    frame_avail_one: str
    frame_avail_two: str
    frame_avail_many: str
    frame_start_immediate: str
    frame_start_date: str
    frame_workauth_yes: str
    frame_workauth_no: str
    frame_location: str
    frame_missing: str


_STRINGS: dict[str, _Strings] = {
    "en": _Strings(
        greeting=(
            "Hi! I'm here to help with your application to our restaurant team. "
            "This will just take a few minutes — I'll ask you a few quick questions about your background. "
            "Let's start: what's your name?"
        ),
        confirming_intro="Here's what I have for your application:\n\n",
        confirming_outro=(
            "\n\nDoes everything look correct? "
            "Feel free to clarify or correct anything."
        ),
        closing=(
            "Thank you for your time! "
            "We'll review your application and be in touch soon."
        ),
        fallback="Thank you! Your screening is complete.",
        avail_display={
            "weekday_day": "weekday day",
            "weekday_evening": "weekday evening",
            "weekend_day": "weekend day",
            "weekend_evening": "weekend evening",
        },
        position_display={
            "server": "server",
            "line_cook": "line cook",
            "host": "host",
            "shift_manager": "shift manager",
            "other": "other",
        },
        year_singular="year",
        year_plural="years",
        year_phrase="{n} {unit} of experience",
        list_join="and",
        frame_name_pos="I have you down as {name}, applying for the {position} position.",
        frame_name="I have you down as {name}.",
        frame_pos="You're applying for the {position} position.",
        frame_exp_skills="You have {exp} of experience, with skills in {skills}.",
        frame_exp="You have {exp} of experience.",
        frame_skills="Your skills include {skills}.",
        frame_avail_one="You're available {avail}.",
        frame_avail_two="You're available {a1} and {a2}.",
        frame_avail_many="You're available {avail}.",
        frame_start_immediate="You can start right away.",
        frame_start_date="Your earliest start date is {date}.",
        frame_workauth_yes="You're authorized to work.",
        frame_workauth_no="You mentioned you're not currently authorized to work.",
        frame_location="Your preferred location is {location}.",
        frame_missing="I wasn't able to capture: {fields}.",
    ),
    "es": _Strings(
        greeting=(
            "¡Hola! Estoy aquí para ayudarte con tu solicitud para nuestro equipo. "
            "Solo tomaré unos minutos — te haré algunas preguntas sobre tu experiencia. "
            "Empecemos: ¿cuál es tu nombre?"
        ),
        confirming_intro="Esto es lo que tengo para tu solicitud:\n\n",
        confirming_outro=(
            "\n\n¿Todo está correcto? "
            "Puedes aclarar o corregir cualquier cosa."
        ),
        closing=(
            "¡Gracias por tu tiempo! "
            "Revisaremos tu solicitud y nos pondremos en contacto pronto."
        ),
        fallback="¡Gracias! Tu evaluación ha finalizado.",
        avail_display={
            "weekday_day": "días de semana por la mañana",
            "weekday_evening": "días de semana por la tarde",
            "weekend_day": "fines de semana por la mañana",
            "weekend_evening": "fines de semana por la tarde",
        },
        position_display={
            "server": "mesero/a",
            "line_cook": "cocinero/a de línea",
            "host": "anfitrión/a",
            "shift_manager": "gerente de turno",
            "other": "otro",
        },
        year_singular="año",
        year_plural="años",
        year_phrase="{n} {unit} de experiencia",
        list_join="y",
        frame_name_pos="Te tengo registrado/a como {name}, solicitando el puesto de {position}.",
        frame_name="Te tengo registrado/a como {name}.",
        frame_pos="Estás solicitando el puesto de {position}.",
        frame_exp_skills="Tienes {exp}, con habilidades en {skills}.",
        frame_exp="Tienes {exp}.",
        frame_skills="Tus habilidades incluyen {skills}.",
        frame_avail_one="Estás disponible {avail}.",
        frame_avail_two="Estás disponible {a1} y {a2}.",
        frame_avail_many="Estás disponible {avail}.",
        frame_start_immediate="Puedes empezar de inmediato.",
        frame_start_date="Tu fecha de inicio más temprana es {date}.",
        frame_workauth_yes="Estás autorizado/a para trabajar.",
        frame_workauth_no="Mencionaste que actualmente no estás autorizado/a para trabajar.",
        frame_location="Tu ubicación preferida es {location}.",
        frame_missing="No pude capturar: {fields}.",
    ),
}


def get_strings(language: str) -> _Strings:
    """Return the string table for *language*, falling back to EN."""
    return _STRINGS.get(language, _STRINGS[DEFAULT_LANGUAGE])


def greeting(language: str) -> str:
    return get_strings(language).greeting


def closing(language: str) -> str:
    return get_strings(language).closing


def fallback(language: str) -> str:
    return get_strings(language).fallback


# --------------------------------------------------------------------------- TTS voice map

# Default voices per language. Env override (DEEPGRAM_TTS_MODEL) always wins.
# aura-2-carina-es supports bilingual EN+ES switching.
_TTS_VOICES: dict[str, str] = {
    "en": "aura-asteria-en",
    "es": "aura-2-carina-es",
}


def tts_voice(language: str) -> str:
    """Return the Deepgram TTS model ID for *language*.

    The DEEPGRAM_TTS_MODEL env var overrides all language-specific defaults.
    """
    env_override = os.getenv("DEEPGRAM_TTS_MODEL")
    if env_override:
        return env_override
    return _TTS_VOICES.get(language, _TTS_VOICES[DEFAULT_LANGUAGE])


# --------------------------------------------------------------------------- confirmation prose


def build_confirmation(record: "ScreeningRecord", language: str) -> str:
    """Build deterministic, TTS-tuned candidate-facing confirmation prose.

    Each sentence ends with a period so TTS pauses naturally between fields.
    Sentence frames are taken from the language string table; values are
    formatted using language-appropriate display labels.
    """
    s = get_strings(language)

    def _val(fname: str):
        ef = getattr(record, fname)
        return ef.value if ef else None

    sentences: list[str] = []

    name = _val("candidate_name")
    position_raw = _val("position_applied_for")
    position = s.position_display.get(str(position_raw), str(position_raw).replace("_", " ")) if position_raw is not None else None
    experience = _val("years_experience")
    skills = _val("relevant_skills")
    availability = _val("availability")
    start_date = _val("earliest_start_date")
    work_auth = _val("work_authorization")
    location = _val("location_preference")

    # Name + position
    if name and position:
        sentences.append(s.frame_name_pos.format(name=name, position=position))
    elif name:
        sentences.append(s.frame_name.format(name=name))
    elif position:
        sentences.append(s.frame_pos.format(position=position))

    # Experience + skills
    if experience is not None:
        unit = s.year_singular if experience == 1 else s.year_plural
        exp_str = s.year_phrase.format(n=experience, unit=unit)
    else:
        exp_str = None

    if exp_str and skills:
        skill_str = _join_list(
            [str(sk).replace("_", " ") for sk in skills] if isinstance(skills, list) else [str(skills).replace("_", " ")],
            s.list_join,
        )
        sentences.append(s.frame_exp_skills.format(exp=exp_str, skills=skill_str))
    elif exp_str:
        sentences.append(s.frame_exp.format(exp=exp_str))
    elif skills:
        skill_str = _join_list(
            [str(sk).replace("_", " ") for sk in skills] if isinstance(skills, list) else [str(skills).replace("_", " ")],
            s.list_join,
        )
        sentences.append(s.frame_skills.format(skills=skill_str))

    # Availability
    if availability:
        avail_list = (
            [s.avail_display.get(str(a), str(a).replace("_", " ")) for a in availability]
            if isinstance(availability, list)
            else [s.avail_display.get(str(availability), str(availability).replace("_", " "))]
        )
        if len(avail_list) == 1:
            sentences.append(s.frame_avail_one.format(avail=avail_list[0]))
        elif len(avail_list) == 2:
            sentences.append(s.frame_avail_two.format(a1=avail_list[0], a2=avail_list[1]))
        else:
            joined = _join_list(avail_list, s.list_join)
            sentences.append(s.frame_avail_many.format(avail=joined))

    # Start date
    if start_date == "immediate":
        sentences.append(s.frame_start_immediate)
    elif start_date:
        sentences.append(s.frame_start_date.format(date=start_date))

    # Work authorization
    if work_auth is True:
        sentences.append(s.frame_workauth_yes)
    elif work_auth is False:
        sentences.append(s.frame_workauth_no)

    # Location (optional)
    if location:
        sentences.append(s.frame_location.format(location=location))

    # Missing required fields
    from .schemas import ScreeningRecord as _SR  # local import avoids circular
    missing_labels = [
        fname for fname in _SR.required_fields()
        if getattr(record, fname) is None
    ]
    if missing_labels:
        sentences.append(s.frame_missing.format(fields=", ".join(missing_labels)))

    return " ".join(sentences)


# --------------------------------------------------------------------------- internal helpers


def _join_list(items: list[str], join_word: str) -> str:
    """Join a list of strings with a language-appropriate conjunction."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {join_word} {items[1]}"
    return ", ".join(items[:-1]) + f", {join_word} {items[-1]}"
