# Plan 001: Phase 1 — Core Engine (text-only, end-to-end)

**Status:** Draft
**Phase:** 1 of 6 (Core Engine)
**Created:** 2026-06-12
**Files touched:**
- `agent/schemas.py` (modify)
- `agent/storage.py` (modify)
- `agent/concurrency.py` (modify — minor)
- `agent/llm.py` (modify)
- `agent/extraction.py` (modify)
- `agent/conversation.py` (modify)
- `agent/output.py` (modify)
- `cli.py` (modify)
- `docs/architecture/decisions.md` (modify — record confidence + immediate-date decisions)

## Goal

A full **text** screening interview runs end-to-end via `python cli.py`: the agent greets,
collects the fixed field set with incremental attributed extraction, re-prompts invalid/missing
fields (cap 2), confirms, renders a reviewer table with per-field confidence + flags, prints a
summary, and persists a JSON record to `data/conversations/{id}.json`. No voice, no bonuses.

## Context

The repo is typed stubs with `TODO(execute)` markers — the interfaces (`LLMClient`,
`ModalityAdapter`, `ConversationStore`) are intentional contracts to fill, not replace. Explore
established:

- **Model id:** stub default `claude-sonnet-4-5` is stale. Use **`claude-opus-4-8`** (decided:
  correctness-first for the HR/trust framing; per-role cost-routing is a documented "with more
  time" lever, not built).
- **Missing piece:** the LLM must not populate `confidence`/`flag`/`provenance` (those are
  rule-derived downstream). We need a thin **`TurnExtraction`** proposal model — raw per-field
  value + `explicitly_stated` + `source_text` — distinct from `ScreeningRecord`. This is the hinge
  of attribution (concept #1) and anti-over-inference.
- **Structured output:** prefer `client.messages.parse(output_format=TurnExtraction)` →
  `.parsed_output` over strict tool use; `AsyncAnthropic`, key from `ANTHROPIC_API_KEY`.
- **Decisions (user):** `earliest_start_date` uses sentinel string `"immediate"` for relative/now
  answers, ISO otherwise. Confidence = validated&stated→0.9 / validated&inferred→0.6 /
  unvalidated→0.3, ×STT-confidence for voice, **CONFIRMED at ≥0.8**.
- **Fix in passing:** `work_authorization` → canonical `bool` (seeds use `true`); `cli.py`
  docstring says "Google STT/TTS" but the project chose **Deepgram** — correct it.

## Approach

Built in dependency order (each step compiles and is independently checkable before the next).
Steps 3–4 are validated with an **inline fake `LLMClient`** so no network/creds are needed in
Phase 1; the real Anthropic call is exercised only in the final end-to-end run (step 7).

**1. `agent/schemas.py`** — implement `Confidence.score` semantics, the four `validate_*`
helpers, and add the `TurnExtraction` proposal model.
- `validate_years_experience` (0–60), `validate_position` (→`Position`), `validate_availability`
  (non-empty `Shift` list), `validate_start_date` (ISO `date` | literal `"immediate"` | `None`).
- `TurnExtraction`: optional raw fields mirroring `ScreeningRecord`'s value types, each paired with
  an `explicitly_stated: bool` and `source_text: str`. This is the only schema the LLM fills.
- Canonicalize `work_authorization` to `bool`.
- _Validate:_ `python -c "import agent.schemas"` clean; quick asserts on each validator
  (e.g. `validate_start_date("immediate") == "immediate"`, rejects `2026-13-40`).

**2. `agent/storage.py`** — `JsonFileStore.{new_conversation,load,save}`.
- `new_conversation`: uuid + ISO timestamp, `GREETING`, persist, return snapshot.
- `save`: serialize snapshot (incl. nested `ExtractedField`/`Provenance`/`Confidence`) via Pydantic;
  **atomic write** (temp file + `os.replace`) so a crash never corrupts a transcript.
- `load`: deserialize; return `None` if absent.
- _Validate:_ round-trip script — `new_conversation` → mutate → `save` → `load` → assert equal;
  confirm a `.json` lands in `data/conversations/`.

**3. `agent/llm.py` + `agent/concurrency.py`** — `ClaudeClient` on `AsyncAnthropic`,
model `claude-opus-4-8`; accept an injected `ConcurrencyLimiter`.
- `extract_structured`: `await client.messages.parse(model, system, messages,
  output_format=schema)` → `.parsed_output`, wrapped in `limiter.run(...)`; let the SDK handle
  429/5xx retries via `max_retries`. `respond`: plain `messages.create`, return text.
- Fix the stale model-id default.
- _Validate:_ `python -c "import agent.llm"` clean; defer live call to step 7. Define the inline
  `FakeLLM(LLMClient)` here (or in a scratch file) for steps 4–6.

**4. `agent/extraction.py`** — incremental attributed extraction.
- Build the extraction prompt from current known state + latest turn (fill gaps/corrections only).
- Call `extract_structured(..., schema=TurnExtraction)`; **drop any value not grounded** in
  `source_text` (anti-over-inference → no false positives); attach `Provenance`
  (`turn_index` + STT span when present).
- `_merge_field`: keep existing vs incoming; on contradiction, **append to `conflicting_values`**
  rather than overwrite.
- _Validate:_ feed the three seed turns through `extract_turn` with `FakeLLM` returning canned
  `TurnExtraction`s; assert clean seed yields the 7 fields with provenance, traps seed leaves
  `years_experience`/`work_authorization` absent and records `position` conflict.

**5. `agent/conversation.py`** — state machine + turn loop.
- `start` → `new_conversation` + greeting. `handle_turn`: load → append → `extract_turn` → merge →
  recompute outstanding → `_next_state` → re-prompt only invalid/low-conf/missing, **cap 2** then
  accept-with-flag → save → reply.
- Completion = every required field is either collected or re-prompt-capped (MISSING is allowed).
  COLLECTING → CONFIRMING (read back) → SUMMARY.
- _Validate:_ scripted multi-turn run with `FakeLLM`: assert state progression, that a never-stated
  required field ends MISSING (not invented), and re-prompt cap halts the loop.

**6. `agent/output.py`** — confidence, flags, rendering.
- `compute_confidence`: the decided formula (0.9/0.6/0.3, ×STT for voice).
- `assign_flag`: CONFIRMED iff validated & explicitly_stated & score ≥ 0.8; CONFLICTING when
  `conflicting_values` non-empty; else NEEDS_REVIEW; record-level MISSING.
- `render_reviewer_table` (✓/⚠/✗/!) and `build_summary`.
- Record the confidence formula + immediate-date decision in `docs/architecture/decisions.md`.
- _Validate:_ unit-style asserts on flag boundaries (stated+valid→confirmed, inferred→needs_review,
  conflict→conflicting); eyeball the rendered table.

**7. `cli.py`** — wire `TextAdapter` → engine loop → table + summary; fix the "Google" docstring.
- _Validate (Phase 1 gate):_ `ANTHROPIC_API_KEY=… python cli.py` — complete a real text interview,
  see the reviewer table + summary, confirm `data/conversations/{id}.json` written with provenance
  on every collected field.

## Tradeoffs Considered

- **Separate `TurnExtraction` proposal model vs. LLM filling `ScreeningRecord`.** Chosen: separate.
  Letting the model emit `confidence`/`flag` would violate the rule-derived-confidence concept and
  invite self-reported confidence; a thin proposal model keeps attribution and anti-over-inference
  enforceable in our code. Cost: one extra schema + a mapping step.
- **`messages.parse()` vs strict tool use vs `output_config.format`.** Chosen: `parse()`. Native
  Pydantic validation, least plumbing, and validation failure maps cleanly to "treat field as
  not-collected." All three sit behind `LLMClient`, so this is reversible.
- **Graded confidence (0.9/0.6/0.3) vs a hard stated-vs-inferred gate.** Chosen: graded. Produces a
  sortable score for the reviewer and a single tunable threshold, while still preventing
  inferred/invalid fields from reaching CONFIRMED. Slightly more arbitrary constants — documented.
- **Sync `ConversationStore` under an async engine.** Chosen: keep sync. Local JSON I/O is fast and
  the interface is what proves statelessness; async file I/O is premature abstraction here. Noted as
  a brief event-loop block to revisit if storage moves to a network backend.
- **`"immediate"` sentinel vs normalize-to-run-date vs needs_review.** Chosen: sentinel. Honest —
  it never encodes a date the candidate didn't state, so it can't become a false positive.
- **Opus 4.8 everywhere vs split conversation/extraction models.** Chosen: Opus everywhere.
  Correctness-first for an HR decision surface; the cheaper-extraction-model split is described in
  the README scaling writeup as a future lever, not built.

## Validation

- Per-step checks above (imports clean; pure-function asserts; `FakeLLM`-driven scripted runs — all
  offline, no creds).
- **Phase 1 gate:** one real `python cli.py` text interview runs to completion, renders the reviewer
  table + summary, and persists a JSON record with provenance on every collected field. No
  hallucinated fields on a transcript where the candidate omits one.
- Full deterministic unit tests + the precision/FP eval harness are **Phase 2** (build-plan steps
  8–11); Phase 1 only needs the offline scripted checks plus the live end-to-end run.

## Open Questions

- Exact minimum `anthropic` version that supports `messages.parse` — verify on PyPI before bumping
  `requirements.txt` (currently `>=0.40`, too old).
- Extraction prompt wording will be tuned against the seed transcripts in Phase 2; Phase 1 just needs
  a correct, attributed first cut.
- How the eval harness scores the traps seed's `conflicting` key (separate sub-metric vs folded into
  precision/FP) — a Phase 2 decision, surfaced now so the `conflicting_values` plumbing lands in
  Phase 1.
- Deepgram STT/Aura SDK syntax is unverified and out of scope until Phase 3 (will WebFetch the v3.x
  docs before writing `voice.py`).
