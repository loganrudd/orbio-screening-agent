# Plan 003: Phase 3 — Voice Adapter (discrete STT → engine → TTS)

**Status:** Draft
**Phase:** 3 of 6
**Created:** 2026-06-12
**Canonical location on approval:** copy this file to `docs/plans/003-voice-adapter.md`
(plan mode only permits editing the scratch plan file; the durable copy is created at execute time).

## Goal

One clean voice conversation runs through the **unchanged** conversation engine:
candidate audio → Deepgram STT → engine (text) → Deepgram Aura TTS → audio out, with
each extracted field attributed to the **exact timestamped sub-span** of the
candidate's speech that produced it, and STT word-confidence feeding rule-derived
confidence. Voice degrades to text cleanly. Engine and all three first-class concepts
work identically with voice removed.

## Context

Phase 1 + 2 are complete. The codebase is already ~90% staged for voice — voice is the
highest-priority bonus because it is the most on-brand for the reviewer's product:

- `CandidateInput` ([agent/voice.py:18](agent/voice.py#L18)) already carries
  `stt_confidence`, `audio_start_s/end_s`.
- `Turn` ([agent/storage.py:24](agent/storage.py#L24)) carries `audio_start_s/end_s`.
- `Provenance` ([agent/schemas.py:54](agent/schemas.py#L54)) has audio-span fields.
- `compute_confidence` ([agent/output.py:35](agent/output.py#L35)) accepts `stt_confidence`.
- `extraction.py` ([agent/extraction.py:114](agent/extraction.py#L114)) has the literal
  `# Phase 3: pass STT word confidence here` hook.
- `VoiceAdapter` ([agent/voice.py:49](agent/voice.py#L49)) currently delegates to
  `TextAdapter` (graceful-degradation default).

**Gaps to close:** (1) `Turn` has no `stt_confidence`/word data and
[cli.py:46](cli.py#L46) drops `candidate_input.stt_confidence`; (2) `VoiceAdapter` is a
stub; (3) per-field attribution requires aligning each field's `source_text` to STT
word timings.

**Decisions locked with the user (2026-06-12):**
- **Per-field word-aligned attribution** (not utterance-level): each field gets the exact
  sub-span + min word-confidence of the words that produced it.
- **Live mic push-to-talk** for the demo surface (discrete record, not streaming).

**Gotchas found in Explore:**
- `deepgram-sdk` **7.3.1** is installed; build-plan/`requirements.txt` reference v3.x.
  The v4+ SDK was rewritten — **execute MUST verify the prerecorded-STT and Aura-TTS API
  against the installed 7.3.1 surface** (WebFetch current Deepgram Python docs + a tiny
  throwaway probe) before writing adapter code. Do not code STT/TTS from memory.
- `sounddevice` is **not installed**. Phase 3 adds an audio-I/O dependency.
- All Deepgram calls must pass through `ConcurrencyLimiter`
  ([agent/concurrency.py](agent/concurrency.py)) per `style.md`.

## Approach

### Step 0 — Verify SDK + deps (no code yet)
- WebFetch Deepgram Python SDK v7 docs; confirm: (a) async prerecorded transcription
  returning per-word `{word, start, end, confidence}`, (b) Aura TTS returning audio
  bytes. Probe with a ~10-line throwaway script (gated on `DEEPGRAM_API_KEY`).
- Add to `requirements.txt`: `sounddevice`, `numpy`; pin `deepgram-sdk>=7.3,<8`.
- **Validate:** probe prints a transcript with word timings + plays/produces TTS bytes.

### Step 1 — Engine-side generic timing + per-field alignment (no network)
Keep the engine modality-agnostic: it consumes a **provider-neutral** `WordTiming`, never
a Deepgram type. Text mode passes nothing → behaves exactly as today.

- `agent/storage.py`: add `@dataclass WordTiming(word: str, start_s: float, end_s: float,
  confidence: float)`. Extend `Turn` with `stt_confidence: Optional[float] = None` and
  `words: Optional[list[WordTiming]] = None`. Serialize `stt_confidence` in
  `_snapshot_to_dict`/`_from_dict` (utterance-level audit value); keep `words` **transient**
  (in-memory only, defaults `None` on load) to avoid transcript bloat — per-field spans are
  already persisted inside `Provenance`.
- `agent/extraction.py`: add a **pure** helper
  `_align_span(source_text, words) -> tuple[start_s, end_s, min_conf] | None` — normalize
  (lowercase, strip punctuation), tokenize the quote, find the best contiguous word window,
  return `(first.start, last.end, min(confidence over matched words))`. In `extract_turn`,
  when `latest_turn.words` is present, derive per-field `(start, end, stt_conf)` via
  alignment; **fall back** to `latest_turn.audio_start_s/end_s` + `latest_turn.stt_confidence`
  when alignment fails or words absent. Feed the span into `Provenance` and `stt_conf` into
  `compute_confidence` (replacing the `stt_confidence=None` placeholder).
- **Validate:** new unit tests — alignment (exact / punctuation+case / multi-word /
  no-match fallback / min-confidence pick) and an extraction test (MockLLM proposal with a
  known `source_text` + a synthetic `words` list) asserting the per-field span + reduced
  `confidence.score`. Existing suite stays green; text mode unchanged.

### Step 2 — VoiceAdapter STT path
- `agent/voice.py`: add `words: Optional[list[WordTiming]]` to `CandidateInput`. In
  `VoiceAdapter.__init__`, read `DEEPGRAM_API_KEY`; if absent/import fails → `degraded=True`
  and delegate to `TextAdapter` (already the stub behavior). Take an optional
  `ConcurrencyLimiter`.
- `read_candidate`: push-to-talk — Enter starts mic capture (`sounddevice`, wrapped in
  `asyncio.to_thread`), Enter stops; buffer → Deepgram **prerecorded** STT (async, via
  limiter) → parse transcript + word list; utterance span = first→last word time,
  utterance `stt_confidence` = min word confidence. Return
  `CandidateInput(text, audio_start_s, audio_end_s, stt_confidence, words)`. Any
  failure → log (structlog) + degrade to text for that turn.
- **Validate:** `tests/test_voice.py` with Deepgram client + `sounddevice.rec`
  monkeypatched (no hardware/network) → asserts `CandidateInput` fields map from a canned
  STT response; no-API-key path degrades to text (monkeypatched `input`).

### Step 3 — VoiceAdapter TTS path
- `emit_agent`: Deepgram Aura TTS (async, via limiter) → audio bytes → play via
  `sounddevice` (`asyncio.to_thread`). Also print the agent text (demo shows transcript).
  Failure → text fallback, no crash.
- **Validate:** test with TTS + playback monkeypatched → no crash; failure path degrades.

### Step 4 — CLI wiring
- [cli.py:46](cli.py#L46): pass `stt_confidence=candidate_input.stt_confidence` and
  `words=candidate_input.words` into `Turn`.
- **Validate:** `python cli.py` (text) smoke unchanged; `python cli.py --voice` dry-run
  reaches the mic prompt and degrades cleanly if no key.

### Step 5 — Demo + docs (manual)
- Record ONE voice conversation (audio in → transcript → reviewer panel → audio out).
- README "Voice adapter" section (decoupled adapter, per-field STT attribution, discrete
  pipeline, Gemini-Live as the production path — **not built**); `docs/architecture/`
  writeup for the per-field-alignment + generic-WordTiming decisions.
- Update `MEMORY.md` / phase-status memory: Phase 3 complete.

## Tradeoffs Considered

- **Per-field word-aligned vs utterance-level attribution** → chose per-field (user-approved).
  Utterance-level is already mostly wired and trivial, but gives every field in a turn the
  same span; per-field is the literal first-class-concept #1 and the strongest talking point.
  Cost: a pure alignment helper + a graceful fallback (already specified).
- **Generic `WordTiming` on `Turn` vs Deepgram types in the engine** → chose generic. Keeps
  the engine modality-agnostic (Deepgram→WordTiming mapping stays in `voice.py`); text mode
  passes `None` and is unaffected. Preserves the load-bearing modality-decoupling decision.
- **Persist raw word list vs transient** → transient. Per-field spans already live in
  `Provenance`; persisting full word arrays bloats transcripts for no reviewer gain. Persist
  only the utterance-level `stt_confidence` for audit.
- **`sounddevice` vs pyaudio/afplay** → `sounddevice` (+`numpy`): cross-platform, clean
  numpy buffers, one dependency covering both capture and playback; afplay is macOS-only.
- **Live mic vs file input** → live mic (user-approved) for the demo; unit tests mock
  Deepgram + the mic regardless, so CI stays hardware/network-free either way.

## Validation

- `pytest` (default suite) green with **no network, credentials, or audio hardware** —
  Deepgram, mic, and playback all mocked.
- Alignment helper unit-tested as a pure function.
- Extraction test proves per-field span + STT-reduced confidence thread through with MockLLM.
- `python cli.py` text mode unchanged; `python cli.py --voice` runs the discrete pipeline
  (or degrades cleanly without a key).
- Manual gate: one recorded voice conversation completes through the unchanged engine.

## Open Questions

- Exact Deepgram 7.3.1 method names/options for prerecorded STT word timings and Aura TTS
  byte output — resolved by the Step 0 probe before any adapter code is written.
- STT confidence aggregation: defaulting to **min** word-confidence (conservative,
  anti-false-positive aligned); revisit if it flags too aggressively in the live demo.
