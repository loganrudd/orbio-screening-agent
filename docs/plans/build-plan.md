# Build Plan — Per-Phase Task Graph

> Nothing transfers from any prior repo — this is a from-scratch build. The Explore
> phase is therefore short: confirm the Claude and Deepgram SDK syntax/auth against
> their current docs, then proceed. No reuse inventory needed.

## Workflow: Explore → Plan → Execute
- **Explore (brief):** confirm current Anthropic (messages + tool-use/structured
  output) and Deepgram (STT + Aura TTS) SDK usage and auth against their docs; confirm
  pinned versions in `requirements.txt`. Output: any syntax deltas to fold into the
  stubs.
- **Plan:** lock the task list below.
- **Execute:** build phase by phase, gated.

## Phase 1 — Core Engine (no bonuses)
1. `schemas.py` — field models, provenance type, flag enum, confidence type, validators.
2. `storage.py` — `ConversationStore` interface + JSON-file implementation.
3. `llm.py` — Anthropic Claude client (structured-output/tool-use, retries, timeouts).
4. `extraction.py` — incremental structured extraction + attribution + anti-over-inference.
5. `conversation.py` — state machine, turn loop, re-prompt cap, conflict detection.
6. `output.py` — confidence computation, flag assignment, reviewer table + summary.
7. `cli.py` — terminal loop wiring it together.
**Gate:** a full text screening runs end-to-end in the CLI and persists a record.

## Phase 2 — Tests + Eval Harness
8. Unit tests (extraction, validation, conversation, output) — LLM mocked.
9. `data/seed_transcripts/` — labeled ground-truth transcripts (seed set provided).
10. `eval/harness.py` — precision + false-positive scoring; deterministic.
11. CLI smoke test.
**Gate:** tests pass without network/creds; harness prints precision + FP numbers.

## Phase 3 — Voice Adapter (highest-priority bonus)
12. `ModalityAdapter` interface; refactor `cli.py` to go through it (text impl first).
13. `voice.py` — Deepgram STT in (word timestamps + confidence) / Deepgram Aura TTS out.
14. Wire STT confidence + timestamps into confidence + provenance.
15. Record one voice conversation for the demo.
**Gate:** one clean voice conversation runs through the unchanged engine.

## Phase 4 — Multilingual EN + ES
16. Language detection on first substantive turn; carry `language` in state.
17. Localized prompt templates; extraction normalizes values to canonical form
    regardless of input language.
18. Demo one EN and one ES conversation. Document extensibility to more languages.
**Gate:** ES conversation extracts canonical-form fields correctly.

## Phase 5 — MLflow Tracing
19. `observability.py` — wrap each turn (extraction, retrieval, LLM call) in traces;
    capture per-turn latency + token cost; no-op if MLflow unconfigured.
**Gate:** a traced run produces an inspectable artifact.

## Phase 6 — Stretch (only if time remains)
20. `agent/rag.py` — vector retrieval over the restaurant FAQ.
21. `agent/sentiment.py` — per-turn frustration flag.
22. `api.py` — FastAPI/SSE web UI.

## Cross-cutting (do alongside, not as a phase)
- `Dockerfile` once Phase 1 is runnable.
- README sections accrue as phases complete (don't leave to the end).
- `docs/architecture/` decision writeups for the load-bearing decisions.
