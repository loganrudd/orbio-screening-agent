# Plan 002: Phase 2 — Tests + Eval Harness

**Status:** Draft
**Phase:** 2 of 6 (Tests + Eval Harness)
**Created:** 2026-06-12
**Files touched:**
- `pytest.ini` (create — markers + asyncio mode)
- `tests/conftest.py` (create — `MockLLM`, builders, fixtures)
- `tests/test_validation.py` (create)
- `tests/test_extraction.py` (create)
- `tests/test_conversation.py` (create)
- `tests/test_output.py` (create)
- `tests/test_storage.py` (create)
- `tests/test_eval.py` (create)
- `tests/test_cli_smoke.py` (create)
- `tests/test_integration_live.py` (create — flagged, real Claude)
- `eval/replay.py` (create — `ReplayLLM` + offline `extract_fn` factory)
- `eval/record.py` (create — flagged fixture-capture script)
- `eval/harness.py` (modify — implement `score_transcript`, `run`, `ScoreReport.render`, `__main__`)
- `data/seed_transcripts/recorded/*.json` (generate via `eval/record.py`)
- `requirements.txt` (modify only if a dev dep is missing)
- `docs/architecture/decisions.md` (modify — record the eval determinism + conflict-scoring decisions)

## Goal

`pytest` passes the full unit suite **with no network or credentials** (the one live
test auto-skips), and `python -m eval.harness` prints a deterministic precision +
false-positive-rate report over the three seed transcripts — making the three
first-class concepts (attribution, reviewer output, measured extraction) provable, not
asserted.

## Context

Phase 1 shipped a complete, gated text engine. **No tests exist** (`tests/` holds only
`__init__.py`); `eval/harness.py` is a stub (`load_seed_transcripts` works;
`score_transcript`/`run` raise `NotImplementedError`). Explore established the seams and
resolved the open questions left by Plan 001:

- **Mock seam:** [`agent/llm.py`](../../agent/llm.py) — `LLMClient` is an ABC
  (`extract_structured`, `respond`). Subclass it; never patch `ClaudeClient`. (Note: the
  shipped client uses **tool-use**, not `messages.parse` as Plan 001 speculated —
  irrelevant to tests since everything mocks the ABC.)
- **Eval seam:** [`eval/harness.py`](../../eval/harness.py) `run(extract_fn)` already
  injects the extractor; `extract_fn(transcript) -> extracted_record_dict` is the
  determinism boundary.
- **Pure-logic targets** are isolated and need no LLM: `Extractor._merge_field`,
  `_process_field`, `_values_differ`, `extract_turn`'s empty-`source_text` drop guard
  ([`extraction.py:103`](../../agent/extraction.py)); `compute_confidence`,
  `assign_flag`; the four `validate_*`; `_next_state`, `_outstanding_fields`,
  `MAX_REPROMPTS_PER_FIELD = 2`.
- **Seeds** ([`data/seed_transcripts/`](../../data/seed_transcripts/)): `seed_en_clean`
  (7 clean fields), `seed_en_traps` (never-stated `years_experience`/`work_authorization`
  + `conflicting` position), `seed_es_normalization` (ES → canonical, `de inmediato` →
  `"immediate"`). Ground-truth schema: `should_contain` (dict), `should_not_contain`
  (list), optional `conflicting` (dict field→[values]), `notes`.

**Decisions confirmed with the user (this session):**
1. **Eval = recorded per-turn proposals replayed through the real `Extractor`** (not
   pre-baked records) — deterministic CI that still exercises merge/validate/conflict/flag.
2. **Fixtures captured once from real Claude** via a flagged script that also prints the
   live precision/FP for the README.
3. **One flagged live integration test**, excluded from default CI.

## Approach

Built bottom-up: scaffolding → pure-function tests → engine tests → eval harness →
fixture capture → smoke/live. Each step is green before the next.

**1. `pytest.ini` + `tests/conftest.py`** — scaffolding the rest depends on.
- `pytest.ini`: `asyncio_mode = auto` (avoid per-test decorators), markers `live` and
  `integration`, `testpaths = tests`.
- `conftest.py`:
  - `MockLLM(LLMClient)` — `extract_structured` returns queued `TurnExtraction`s in call
    order (or a single canned one); `respond` returns a canned/scripted string. Optional
    "raise" mode to test the extractor's exception path.
  - Builders: `make_confidence`, `make_field`, `make_provenance`, `make_record` to
    assemble `ExtractedField`/`ScreeningRecord` states without the LLM.
- _Validate:_ `pytest --collect-only` succeeds; a trivial async test runs under
  `asyncio_mode=auto`.

**2. `tests/test_validation.py`** — pure validators + per-field normalization.
- `validate_years_experience` (0–60 bounds), `validate_position` (→`Position`),
  `validate_availability` (non-empty `Shift` list, rejects junk),
  `validate_start_date` (valid ISO passes; `2026-13-40` rejected; every sentinel form —
  immediate/asap/now/right away/today — → `"immediate"`).
- `_process_field` per field: valid→`(value, True)`, junk→`(None, False)` skip signal.
- _Validate:_ `pytest tests/test_validation.py` green.

**3. `tests/test_extraction.py`** — the attribution + anti-over-inference heart.
- `extract_turn` merges a `MockLLM` proposal into the record with correct provenance
  (`turn_index` + `source_text`).
- **Empty/blank `source_text` → field dropped** (anti-over-inference; the headline FP
  defense).
- Normalization failure (e.g. bad position) → field skipped, stays `None`.
- LLM raises → record returned unchanged.
- `_merge_field`: same value keeps higher-confidence + accumulates provenance; differing
  value → `CONFLICTING` with both in `conflicting_values`.
- `_values_differ`: list comparison order-agnostic.
- _Validate:_ `pytest tests/test_extraction.py` green.

**4. `tests/test_conversation.py`** — state machine + re-prompt cap.
- `_next_state` covers GREETING→COLLECTING→CONFIRMING→SUMMARY.
- `_outstanding_fields` excludes collected and reprompt-capped fields.
- Re-prompt cap: after 2 re-prompts a still-missing required field drops from outstanding
  (accept-with-flag, no infinite loop).
- `handle_turn` end-to-end with `MockLLM` + `JsonFileStore(tmp_path)` reaches
  `done=True`; conflicting value across turns surfaces as `CONFLICTING`.
- _Validate:_ `pytest tests/test_conversation.py` green.

**5. `tests/test_output.py` + `tests/test_storage.py`** — flags/confidence + persistence.
- `compute_confidence`: 0.9 (validated+stated), 0.6 (validated+inferred), 0.3
  (unvalidated), × `stt_confidence` when voice.
- `assign_flag` priority: conflicting > reprompt-capped(needs_review) > score≥0.8
  confirmed > needs_review.
- `render_reviewer_table` contains ✓/⚠/✗/! ; `build_summary` non-empty.
- Storage round-trip: `JsonFileStore` save→load preserves state incl. the
  `Confidence.score` computed-field (`extra='ignore'`) and `reprompt_counts`.
- _Validate:_ both files green.

**6. `eval/replay.py` + `eval/harness.py`** — the deterministic harness.
- `eval/replay.py`: `ReplayLLM(LLMClient)` pops recorded `TurnExtraction`s in order
  (raises if exhausted); `replay_extract_fn(recorded_dir)` returns
  `extract_fn(transcript)` that drives `Extractor.extract_turn` over the transcript's
  **candidate** turns with a `ReplayLLM` seeded from `recorded/{seed_id}.json`, returning
  `ScreeningRecord.model_dump(mode="json")`.
- `eval/harness.py`:
  - `score_transcript(transcript, extracted)` →
    - **should_contain**: recorded + normalized match → `correct_stated`; recorded but
      wrong → `mis_extraction`; absent → `false_negative`.
    - **should_not_contain**: recorded → `false_positive`.
    - **conflicting** (optional): field flag must be `CONFLICTING` and
      `{value} ∪ conflicting_values` set-equals expected — scored as correct-handling,
      **excluded from the precision denominator and never an FP** (resolves Plan 001's
      open question).
    - `claimed_collected` = recorded, non-`None`, non-conflicting fields.
  - Normalized compare: scalars exact (int/str/bool/date-string); lists order-agnostic +
    case-insensitive set equality.
  - `run(extract_fn)` aggregates → `ScoreReport(precision, false_positive_rate, recall,
    mis_extraction_rate, per_field)`; `render()` prints an aggregate + per-field table.
  - `__main__`: offline by default (`replay_extract_fn`); `--live` swaps real
    `ClaudeClient`.
- _Validate:_ `tests/test_eval.py` — `score_transcript` on hand-built extracted dicts:
  precision=1.0 clean; FP detected when a `should_not_contain` field is recorded;
  conflicting field scored as correct-handling not FP. (Runs before fixtures exist using
  synthetic dicts.)

**7. `eval/record.py` → capture fixtures → live numbers.** (Requires `ANTHROPIC_API_KEY`.)
- `RecordingLLM(LLMClient)` wraps real `ClaudeClient`, saving each validated proposal;
  run all seeds, write `data/seed_transcripts/recorded/{seed_id}.json` (ordered list of
  `TurnExtraction` dumps), print the live `ScoreReport`.
- Commit the fixtures. Extend `tests/test_eval.py` with a case asserting
  `run(replay_extract_fn(...))` over the committed fixtures yields the expected aggregate.
- _Validate:_ `python -m eval.record` prints sane numbers; `python -m eval.harness`
  (offline) reproduces them deterministically with no creds.

**8. `tests/test_cli_smoke.py` + `tests/test_integration_live.py`.**
- Smoke: scripted conversation through the engine/CLI with `MockLLM` to SUMMARY; assert
  `done` and a persisted record under `tmp_path`.
- Live: `@pytest.mark.live`, skipped unless `RUN_LIVE=1`; one real-Claude extraction.
- Record both eval-determinism and conflict-scoring decisions in
  `docs/architecture/decisions.md`.
- _Validate (Phase 2 gate):_ `pytest -m "not live"` green with no env keys;
  `python -m eval.harness` prints precision + FP; `RUN_LIVE=1 pytest -m live` passes.

## Tradeoffs Considered

- **Recorded per-turn proposals (chosen) vs pre-extracted full records vs live-every-run.**
  Pre-extracted records are simplest but bypass the merge/conflict/validation/flag code —
  they'd test the scoring, not the pipeline. Live-every-run violates the
  no-network-in-CI rule. Recorded proposals replayed through the real `Extractor` are
  deterministic, credential-free, *and* exercise the real logic; cost is a one-time
  capture step and a `recorded/` fixture dir to maintain.
- **Capture fixtures from real Claude (chosen) vs hand-author proposals.** Hand-authored
  proposals are fully offline but synthetic — they'd prove the scoring math, not the
  model. Capturing once yields an honest README precision/FP number and realistic
  proposals; cost is one keyed run (never in CI).
- **`conflicting` as a separate sub-metric (chosen) vs folded into precision/FP.** Folding
  it in would punish correct conflict-handling as either a miss or an FP — the opposite of
  the intent. A separate sub-metric (excluded from the precision denominator, never an FP)
  rewards the agent for surfacing contradictions honestly. Resolves Plan 001's open question.
- **`asyncio_mode=auto` (chosen) vs per-test `@pytest.mark.asyncio`.** Auto removes
  decorator noise across many async engine/extractor tests; cost is a slightly less
  explicit config, documented in `pytest.ini`.
- **`relevant_skills` exact normalized-set match (chosen) vs fuzzy/semantic match.** Exact
  set (lowercased) keeps the core metric deterministic per `eval.md` (no LLM judge).
  Semantic/subset matching is noted as a documented "with more time" item — its
  brittleness is a known limitation, not a silent one.
- **`JsonFileStore(tmp_path)` in tests (chosen) vs a separate in-memory mock store.** Using
  the real store under `tmp_path` also smoke-tests serialization (incl. the computed-field
  quirk) for free; cost is touching the filesystem, which `tmp_path` isolates.

## Validation

- Per-step checks above (collect-only, per-file `pytest`, all offline).
- **Phase 2 gate:**
  1. `pytest -m "not live"` — full unit + eval suite green with **no network/creds**;
     live test auto-skipped.
  2. `python -m eval.harness` — deterministic precision + FP + per-field table, no creds.
  3. `ANTHROPIC_API_KEY=… python -m eval.record` — captures fixtures, prints the live
     README number (run once; commit fixtures).
  4. `RUN_LIVE=1 pytest -m live` — the single live integration test passes.

## Open Questions

- Whether `relevant_skills` exact-set matching is too strict against the real captured
  proposals (e.g. "grill station" vs "grill") — decide after step 7's capture; loosen to
  normalized-subset only if the live run shows false misses, and document it.
- Final `recall` / `mis_extraction_rate` reporting depth — include as optional columns;
  trim from `render()` if the table gets noisy.
- Minimum viable seed count if time runs short — `eval.md` says the harness shrinks
  (drop a seed) before it's cut; `seed_en_traps` is the one that must never be dropped (it
  carries the FP + conflict cases).
