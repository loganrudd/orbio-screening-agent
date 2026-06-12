# Architecture & Design Decisions

Public-facing rationale (feeds the README). Each load-bearing decision gets the
tradeoff treatment: options, tradeoffs, choice, why.

## 1. Modality-decoupled conversation engine
**Decision:** the conversation engine consumes/emits text and is modality- and
language-agnostic; voice (STT/TTS) and language live at the edges.
**Why:** keeps the state machine, extraction, attribution, and eval testable and
independent of audio plumbing; voice becomes an adapter, not a rewrite. Same engine
serves text or voice. _TODO: expand with the options considered (monolithic voice loop
vs. decoupled adapter) and tradeoffs._

## 2. False-positive definition (eval)
**Decision:** a false positive = a field recorded that the candidate never stated
(hallucinated/over-inferred). Precision = fraction of claimed-collected fields that are
correct AND genuinely stated.
**Why:** for an HR screening agent, inventing fields is the dangerous failure mode;
the metric must penalize exactly that. _TODO: expand._

## 3. Rule-derived confidence
**Decision:** confidence = f(validation status, stated-vs-inferred, STT word
confidence) — NOT the model's self-report.

**Formula (implemented in `Confidence.score` as a `@computed_field`):**
- 0.9 — validated AND explicitly stated by the candidate
- 0.6 — validated AND inferred (not directly stated)
- 0.3 — failed validation
- × stt_confidence when the input came via voice (0..1)

**CONFIRMED threshold: score ≥ 0.8** — only validated + explicitly-stated fields
reach CONFIRMED. Inferred or invalid fields are always NEEDS_REVIEW or lower.

**Options considered:**
1. Hard binary gate (stated vs. not-stated): simpler but loses granularity between
   "inferred but plausible" and "failed validation."
2. Model self-reported confidence: rejected — LLM confidence is poorly calibrated and
   not auditable; an HR agent must justify every field value.
3. Graded 3-level formula (chosen): produces a sortable score for the reviewer, a
   single tunable threshold, and maps cleanly to the output-contract flags. The
   constants (0.9/0.6/0.3) are documented and can be calibrated in Phase 2 against
   the seed transcripts.

**Why this matters for an HR agent:** A reviewer can see the score and understand
exactly why a field is CONFIRMED vs. NEEDS_REVIEW. The score is computed from
deterministic inputs — not a black box.

## 3a. `earliest_start_date` sentinel value
**Decision:** relative/now answers ("ASAP", "immediately", "right away") are stored
as the literal string `"immediate"` rather than being normalized to the run-date.

**Options considered:**
1. Normalize to today's date: introduces false precision (we don't know when "now"
   is from the candidate's perspective, and a "today" from last week is wrong).
2. Mark as NEEDS_REVIEW / missing: wastes a clear, honest answer.
3. Sentinel string `"immediate"` (chosen): honest — it records what the candidate
   actually said without inventing a date they didn't state. Can never be a false
   positive. The eval harness matches on the exact sentinel string.

**Implementation:** `validate_start_date()` in `schemas.py` returns `"immediate"` for
any of: immediate / asap / now / right away / immediately / today.

## 4. Discrete STT→LLM→TTS pipeline (not a real-time native-audio API)
**Decision:** discrete voice pipeline for the build; a real-time native-audio API (e.g. Gemini Live) named as the
production path.
**Why:** real-time streaming would bypass the text engine and the first-class concepts,
and burn the time budget on audio plumbing. _TODO: expand._

## 5. Statelessness via storage interface
**Decision:** all conversation state flows through `ConversationStore`; no in-process
state.
**Why:** horizontal scaling becomes a deployment concern, not a rewrite. _TODO: expand._

## 6. Scaling & bottlenecks (README writeup + diagram)
- Real bottleneck: LLM-provider I/O, rate limits, per-conversation token cost,
  provider failover — not compute.
- Mitigations built: async + concurrency limiter, stateless engine, Dockerfile.
- Production path: stateless replicas behind a load balancer; durable execution
  (e.g. Temporal) for long/resumable interviews; a real-time native-audio API for real-time voice;
  cost-aware multi-model routing (cheap model for extraction, expensive only when
  needed).
- _TODO: insert architecture diagram (ASCII or image)._

## 7. Eval harness determinism strategy (Phase 2)

**Decision:** the CI eval harness replays **pre-recorded per-turn `TurnExtraction`
proposals** through the real `Extractor` pipeline (merge/validate/conflict/flag logic),
rather than scoring pre-baked complete records or hitting the live API every run.

**Options considered:**
1. Pre-extracted full records: simplest, but bypasses the pipeline — only tests the
   scoring math, not the extraction/merge/conflict code.
2. Recorded per-turn proposals replayed through real `Extractor` (chosen): deterministic
   and credential-free in CI, yet exercises the actual merge/conflict/provenance/flag
   logic. Fixtures captured once via `python -m eval.record` (flagged).
3. Live API every run: non-deterministic, requires credentials in CI, fails offline.

**Why this matters:** The harness proves the pipeline, not just the scoring formula.
A change to `_merge_field` or the anti-over-inference guard will be caught by a fixture
regression, not just a unit test.

**Fixture capture results (as of 2026-06-12):**
- Precision: 0.929 | FP rate: 0.000 | Recall: 1.000
- 1 mis-extraction: `earliest_start_date` year ambiguity ("June 23rd" → 2025 vs. expected 2026).
  This is a documented harness limitation for relative date expressions without an explicit year.
  Future mitigation: seed transcripts should include 4-digit years for unambiguous dates.

## 8. `conflicting` field scoring in the eval harness (Phase 2)

**Decision:** conflicting fields (those where the candidate gave contradictory values) are
scored as a separate sub-metric (`conflict_correct` / `conflict_missed`), **excluded from
the precision denominator**, and never counted as false positives.

**Options considered:**
1. Count as mis-extraction: penalizes correct conflict-handling — wrong incentive.
2. Count as false positive: wrong — the field WAS stated, just contradictorily.
3. Separate sub-metric, excluded from precision denominator (chosen): rewards surfacing
   contradictions honestly. A `CONFLICTING` flag with both values set is correct
   handling; a single value with no flag is `conflict_missed`.

**Why:** An agent that surfaces a candidate's self-contradiction is doing its job. The
reviewer-facing output (first-class concept #2) explicitly requires `CONFLICTING` flagging
for detected contradictions. The harness must reward, not penalize, this behavior.

## 9. Per-field word-aligned STT attribution (Phase 3)

**Decision:** each extracted field's `Provenance` carries the exact timestamped
sub-span of the candidate's speech that produced it — not the whole utterance span.
STT word-level confidence feeds the `stt_confidence` multiplier in the confidence
formula.

**Options considered:**
1. Utterance-level span only: almost free (already wired). Every field extracted
   from a turn shares the same `(audio_start_s, audio_end_s)`. Simpler, but coarse —
   a "name + position + experience" turn would point all three fields at a 10-second
   span rather than the specific words.
2. Per-field word-aligned span (chosen): `_align_span(source_text, words)` in
   `extraction.py` normalizes both the field's `source_text` quote and the STT word
   list (lowercase, punctuation stripped), finds the best-matching contiguous word
   window (≥50% token match), and returns `(first.start, last.end, min_confidence)`.
   Falls back to utterance-level when words are absent or alignment fails. Requires a
   pure alignment helper + one extra field on `Turn`.

**Why per-field:** first-class concept #1 (source attribution) says "the exact
timestamped span of the candidate's actual speech." Utterance-level violates the
spirit for multi-field turns. Per-field attribution is the literal differentiator and
the clearest talking point for an HR agent reviewer who needs to audit a value.

**Why min word-confidence:** conservative by design (anti-false-positive aligned). A
field is only as trustworthy as its weakest word. Reviewers see a lower confidence
score on low-quality STT segments — which is the correct behavior for an HR trust
instrument.

**WordTiming is provider-neutral:** the engine stores `WordTiming(word, start_s,
end_s, confidence)` — never a Deepgram type. The mapping from `deepgram.word.start`
→ `WordTiming.start_s` lives entirely in `voice.py`. Swapping STT providers requires
touching only the adapter.

**`words` is transient (not persisted):** per-field spans are already written into
`Provenance` on every `ExtractedField`. Persisting the full word list would bloat
transcripts with redundant data. Only `stt_confidence` (utterance-level, for audit)
is written to the JSON transcript.

## 10. Discrete push-to-talk mic input (Phase 3)

**Decision:** push-to-talk (Enter-to-start, Enter-to-stop) using `sounddevice` for
mic capture, then a single Deepgram prerecorded-transcription API call. Async,
behind the `ConcurrencyLimiter`.

**Options considered:**
1. File-based input (pre-recorded WAVs): fully deterministic, great for CI and scripted
   demos, but not live.
2. Live mic push-to-talk (chosen): most compelling demo surface for a voice-interview
   product. `sounddevice` captures to a NumPy buffer; the buffer is sent as raw PCM
   to Deepgram in one batch call. CI tests mock `sounddevice.rec` — no hardware needed.
3. VAD-based auto-segmentation: more natural but adds complexity and a VAD dependency.
   Out-of-scope at this phase; the discrete pipeline is the right thing to ship first.

**TTS model:** now language-keyed via `i18n.tts_voice(language)` — EN uses
`aura-asteria-en`, ES uses `aura-2-carina-es` (bilingual EN+ES capable). Overridable
via `DEEPGRAM_TTS_MODEL` env var. ElevenLabs is a one-line swap behind `_speak()`.

**Graceful degradation:** `VoiceAdapter` degrades to `TextAdapter` if
`DEEPGRAM_API_KEY` is absent, or if `deepgram` or `sounddevice` failed to import.
Any per-turn STT or TTS error falls back for that turn only, without crashing.

## 11. Language detection: offline library vs. LLM vs. flag-only

**Decision:** `py3langid` (offline library) for automatic language detection on the
first candidate turn, with `--lang <code>` as an explicit override.

**Options considered:**
1. **Flag-only (`--lang en|es`):** zero extra dependency, fully deterministic, but
   requires the user to know the language in advance. Skips build-plan step 16.
2. **Piggyback on the extraction LLM call:** zero extra API calls; reuses the
   structured-output path. Couples detection into the extraction schema; non-deterministic
   in tests (mock must be language-aware); doesn't run until after the first LLM call.
3. **Offline library — `py3langid` (chosen):** pure, deterministic, network-free
   function. Runs synchronously before extraction. Trivially unit-testable without
   mocking. One small dependency. EN/ES discrimination is reliable even on short text
   (tested on realistic restaurant candidate turns).

**Why this matters:** language detection is a boundary concern, not an extraction
concern. Keeping it a pure function preserves the determinism guarantees that make
the eval harness trustworthy.

**Limitation:** in `auto` mode the greeting is always EN (no candidate text exists
yet). Documented; use `--lang es` for the ES voice demo.

## 12. Candidate-facing confirmation: deterministic templates vs. LLM-generated readback

**Decision:** per-language, deterministic sentence-frame templates in `agent/i18n.py`.

**Options considered:**
1. **LLM-generated readback:** zero templates; scales to any language with little code.
   Non-deterministic, needs mocking in tests, and the TTS-tuned period-separated prose
   would be unpredictable. Any hallucination in the readback is a false positive
   presented directly to the candidate.
2. **Deterministic templates (chosen):** fixed sentence frames per language in a table.
   Deterministic, fully unit-testable, TTS-tuned (each sentence ends with `.` so TTS
   pauses naturally), and on-brand with "localized templates, extensible via config."
   Adding a new language = adding one new `_Strings` table entry.
   Cost: more code per language (acceptable for EN+ES scope).

**Why this matters:** the confirmation is spoken back to the candidate and forms part
of the CONFIRMING state. Determinism here means the reviewer-candidate trust loop is
predictable and auditable.

## Scope: what was intentionally left out, and why
- Kubernetes manifests (Dockerfile is proportionate); web UI (CLI demonstrates the
  contract); multi-agent framework (state machine is clearer); database (JSON via the
  store interface is enough at this scope). _TODO: expand._
