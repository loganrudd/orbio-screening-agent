# Plan 004: Phase 4 — Multilingual EN + ES

**Status:** Draft
**Phase:** 4 of 6
**Created:** 2026-06-12
**Files touched:** `agent/i18n.py` (new), `agent/output.py`, `agent/conversation.py`,
`agent/storage.py`, `agent/voice.py`, `cli.py`, `requirements.txt`,
`tests/test_i18n.py` (new), `tests/test_output.py`, `tests/test_conversation.py`,
`tests/test_storage.py`, `README.md`, `docs/architecture/decisions.md`,
`docs/plans/004-phase-4-multilingual-en-es.md` (new — this plan, saved on execution)

> NOTE: This plan currently lives in the harness plan file (plan mode can only write
> here). **Execution step 0** is to save this content to
> `docs/plans/004-phase-4-multilingual-en-es.md` so it sits alongside 001/003.

## Goal

A candidate can be screened end-to-end in Spanish as cleanly as in English: the agent
auto-detects ES from the first candidate turn (or honors an explicit `--lang es`),
speaks all candidate-facing text in ES, extracts canonical English-keyed fields, and —
in voice mode — uses a Spanish Aura-2 TTS voice. Adding a third language is a config
edit, not a code change.

## Context

Phases 1–3 already laid most of the multilingual rails, so Phase 4 is narrower than
the build-plan headline suggests:

**Already works** — `--lang` flag → `engine.start()` → persisted `snapshot.language`
→ extraction + conversation system prompt; localized greeting constants; **canonical
normalization of ES input is already implemented and already scored** by the eval
harness ([data/seed_transcripts/seed_es_normalization.json](data/seed_transcripts/seed_es_normalization.json):
`mesera`→server, `cinco años`→5). Voice **STT** is already language-aware
([agent/voice.py:286](agent/voice.py#L286)). The Phase 4 gate ("ES extracts canonical
fields") is effectively already passing — this de-risks the phase.

**The real gaps:**
1. **No auto-detection** — language only comes from `--lang` (build-plan step 16 wants
   detection on the first substantive turn).
2. **Candidate-facing output is hardcoded English** — `render_candidate_confirmation`
   ([agent/output.py:63-142](agent/output.py#L63-L142)), the CONFIRMING wrapper
   ([agent/conversation.py:159-162](agent/conversation.py#L159-L162)), and the SUMMARY
   closing + fallback text ([agent/conversation.py:172](agent/conversation.py#L172),
   [:175](agent/conversation.py#L175)). These are spoken to the candidate and must
   localize. (The **reviewer** table/summary stay English by design — the reviewer
   reads English.)
3. **No central strings module** — language strings are scattered.
4. **Voice TTS is English-only** ([agent/voice.py:48](agent/voice.py#L48)
   `aura-asteria-en`; Aura v1 is EN-only — ES needs an Aura-2 Spanish voice).

**Confirmed decisions (this session):** offline detection library; deterministic
per-language confirmation templates; full ES voice (Aura-2 Spanish TTS + ES demo).

## Approach

**Step 0 — Save this plan** to `docs/plans/004-phase-4-multilingual-en-es.md`.

**Step 1 — `agent/i18n.py` (new, foundational, pure, no engine coupling).**
Centralize everything language-specific so a new language = one new table entry.
- `SUPPORTED_LANGUAGES = ("en", "es")`, `DEFAULT_LANGUAGE = "en"`.
- `detect_language(text: str) -> str` wrapping **`py3langid`** (deterministic, offline,
  no seed needed); map result to a supported language, fall back to `DEFAULT_LANGUAGE`
  on uncertainty or unsupported output. Pure function → trivially unit-testable.
- A per-language string table (dataclass or dict) holding: `greeting`,
  `confirming_intro`/`confirming_outro`, `closing` (SUMMARY state reply),
  `fallback`, plus the **display lexicons** the confirmation prose needs (position-enum
  → display, availability-shift → display, "year(s)"/"año(s)", list-join word
  "and"/"y", and the sentence frames).
- `build_confirmation(record, language) -> str` — per-language sentence-frame builders
  (grammar differs: gender, "años", "y"), keeping the deterministic, period-separated
  TTS-tuned style of the current EN prose.
- `tts_voice(language) -> str` — language→Aura model map (EN `aura-asteria-en`; ES an
  Aura-2 Spanish voice — **verify exact id against Deepgram docs at execution; do not
  guess**). Env override still wins.
- `requirements.txt`: add `py3langid`.
- **Test:** `tests/test_i18n.py` — detect EN/ES + fallback; both confirmation renders
  contain expected localized substrings; every string key exists for every supported
  language (table-completeness guard); `tts_voice` per language.

**Step 2 — `agent/output.py`.** Change `render_candidate_confirmation(record)` →
`render_candidate_confirmation(record, language)`, delegating to
`i18n.build_confirmation`. Leave `render_reviewer_table` / `build_summary` English.
Update `tests/test_output.py` (signature + an ES assertion).

**Step 3 — `agent/storage.py`.** Add persisted `auto_detect: bool = False` to
`ConversationSnapshot` (+ `_snapshot_to_dict`/`_from_dict`, `new_conversation`), so the
"should I detect on the first turn" decision survives reload (statelessness). Update
`tests/test_storage.py` round-trip.

**Step 4 — `agent/conversation.py`.** Source greeting/CONFIRMING/SUMMARY/fallback text
from `i18n` by `snapshot.language`; remove `_GREETING_EN/ES` constants. `start()` gains
`auto_detect: bool`. In `handle_turn`, at the GREETING→COLLECTING transition (runs
exactly once, on the first candidate turn):
```
if snapshot.state == GREETING:
    if snapshot.auto_detect:
        snapshot.language = detect_language(turn.content) or snapshot.language
    snapshot.state = COLLECTING
```
All downstream (extraction language, reply language, confirmation) reads
`snapshot.language`. Pass `snapshot.language` to `render_candidate_confirmation`.
Add `tests/test_conversation.py` cases: auto_detect overrides on first ES turn;
explicit language (`auto_detect=False`) is preserved.

**Step 5 — `cli.py`.** Default `--lang auto`; map `auto`→`(initial="en",
auto_detect=True)`, explicit `en|es`→`(that, auto_detect=False)`. Pass `auto_detect`
into `engine.start`. (Voice ES demo uses explicit `--lang es` so its greeting is ES
from turn 1.)

**Step 6 — `agent/voice.py`.** Replace the hardcoded `aura-asteria-en` with
`i18n.tts_voice(self._language)` (env override preserved). STT already language-aware.

**Step 7 — Demo + docs.** Record one EN and one ES conversation (text), plus one ES
**voice** conversation. README: new "Multilingual" section — detection (offline lib +
`--lang auto` sentinel), canonical normalization (already proven by the ES seed),
extensibility (add a language = add an i18n table entry + voice id), and the documented
limitation (in `auto` mode the *greeting* is EN until the first turn is seen).
`docs/architecture/decisions.md`: #11 (offline detection lib + `auto` sentinel vs.
LLM/flag-only) and #12 (deterministic per-language confirmation templates vs.
LLM-generated readback).

## Tradeoffs Considered

- **Detection: offline lib (chosen) vs. piggyback-on-extraction vs. flag-only.**
  Offline `py3langid` keeps detection a pure, deterministic, network-free function —
  matching the project's testability/determinism values — at the cost of one small
  dependency. Piggybacking on the first LLM extraction call avoids a dependency but
  couples detection into the extraction schema and isn't deterministic in tests.
- **Confirmation: deterministic templates (chosen) vs. LLM-generated readback.**
  Templates stay deterministic, unit-testable, and TTS-tuned (the period-separated
  prose), and are "localized templates, extensible via config" — on-brand. Cost: more
  code per language. LLM readback scales to any language with little code but is
  non-deterministic and loses TTS tuning.
- **`auto` greeting language.** In `auto` mode the greeting precedes any candidate
  text, so it's EN until detection runs. Accepted and documented; an explicit
  `--lang es` (used for the ES voice demo) avoids it. A bilingual greeting was
  rejected as awkward for TTS.

## Validation

- After each step, run the targeted suite (`pytest tests/test_i18n.py`,
  `test_output.py`, `test_conversation.py`, `test_storage.py`).
- Full suite stays green (~209 tests, 1 skipped) and requires no network/creds.
- Eval harness still scores the ES seed correctly (extraction unchanged):
  `python -m eval.harness` (or its existing entry) — ES seed precision/FP unchanged.
- Manual end-to-end:
  - `python cli.py` then type Spanish → agent switches to ES for the rest.
  - `python cli.py --lang es` → ES from the greeting.
  - `python cli.py --lang es --voice` → ES STT + ES Aura-2 TTS (records the ES voice demo).
- Gate: an ES conversation extracts canonical-form fields correctly (already covered by
  the seed; manual run confirms the localized candidate-facing surface).

## Open Questions

- **Exact Aura-2 Spanish voice id** — confirm against current Deepgram docs at
  execution before wiring (style rule: never guess).
- **`py3langid` vs `langdetect`** — defaulting to `py3langid` (deterministic without a
  seed); will switch to `langdetect` (with `DetectorFactory.seed=0`) only if py3langid
  proves unavailable/unsuitable on the pinned Python.
