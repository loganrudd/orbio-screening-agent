# Restaurant Screening Agent

A conversational AI agent that conducts a short pre-screening interview for hourly
restaurant roles (server, line cook, host, shift manager), extracts structured
candidate data, and produces a **reviewer-facing** summary a hiring manager can act on.

> Built as a take-home. Three things are treated as first-class, beyond what a basic
> brief would require, because they are what make an HR screening agent trustworthy:
> **(1) every extracted field is attributable to its source, (2) the output is built
> for a human reviewer with per-field confidence and flagging, and (3) a real eval
> harness measures extraction precision and false-positive rate.**

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Auth via API keys (no cloud-project setup):
# export ANTHROPIC_API_KEY=...      # LLM
# export DEEPGRAM_API_KEY=...       # STT + TTS (voice mode only)
```

Run:
```bash
python cli.py                  # text mode, auto-detect language from first turn
python cli.py --voice          # voice mode (Deepgram STT in / Aura TTS out)
python cli.py --lang es        # force Spanish (greeting in ES, no auto-detect)
python cli.py --lang es --voice  # ES voice mode (Aura-2 Spanish TTS)
```

Tests + eval:
```bash
pytest                   # unit suite (no network/credentials required)
python -m eval.harness   # extraction precision + false-positive report
```

## System Architecture

<!-- TODO(execute): insert architecture diagram (ASCII or image). -->

The conversation engine is **modality- and language-agnostic** — it consumes and emits
text. Voice (STT/TTS) and language live at the edges, so the engine, extraction,
attribution, and eval stay clean and testable.

```
audio? --STT--> [ ConversationEngine (text) ] --TTS--> audio?
                       |  state machine
                       |  incremental extraction (+ attribution)
                       |  reviewer-facing output (confidence + flags)
                 ConversationStore (stateless: any replica serves any turn)
```

## Key Design Decisions

See `docs/architecture/decisions.md` for the full tradeoff writeups. Summary:

- **Modality-decoupled engine** — voice is an adapter, not a rewrite; same engine
  serves text or voice.
- **State-machine control flow** over free-form chat or an agent framework — legible
  and testable; control flow driven by structured state, not transcript re-parsing.
- **API-based model (Anthropic Claude)** over self-hosting — right call for this use
  case; strong structured-output/tool-use for extraction.
- **False positive = a field the candidate never stated.** Precision = correct-and-stated
  over claimed-collected. The agent prefers leaving a field missing over guessing it.
- **Rule-derived confidence** (validation × stated-vs-inferred × STT confidence), not
  model self-report — defensible and auditable.
- **Discrete STT→LLM→TTS** voice pipeline; a real-time native-audio API (e.g. Gemini Live) is the production path
  for real-time turn-taking, deliberately scoped out here.
- **Multilingual (EN + ES)** via a central `i18n` module — detection, candidate-facing
  strings, and TTS voice selection are all language-keyed. Adding a third language is
  one new entry in the string table.

## Multilingual Support

**Languages:** English (default) and Spanish. Adding more is a config edit.

**How detection works:**

```bash
python cli.py           # --lang auto (default): greets in EN, detects language
                        # from the first candidate turn, switches for the rest
python cli.py --lang es  # force ES from the greeting (use for voice demos)
```

Detection uses `py3langid` — an offline, deterministic library. No API call, no
seed needed, network-free. Runs exactly once, on the first candidate turn, then
the detected code is persisted in the conversation snapshot (stateless — any
replica that picks up the conversation later reads the correct language from
storage). An explicit `--lang` flag bypasses detection entirely.

**Canonical normalization:** Spanish input always extracts to English-keyed canonical
values (e.g. "mesera" → `server`, "cinco años" → `5`, "de inmediato" → `"immediate"`).
This is verified by the eval harness against `seed_es_normalization.json`.

**Candidate-facing strings** (greeting, confirmation readback, closing) come from
per-language templates in `agent/i18n.py`. The **reviewer table and summary stay
English** — the reviewer reads English regardless of the candidate's language.

**TTS voice:** EN uses `aura-asteria-en`; ES uses `aura-2-carina-es` (bilingual
EN+ES capable). A new language needs one new entry in `_TTS_VOICES` in `i18n.py`.

**Known limitation:** In auto-detect mode the greeting is EN (it runs before any
candidate text exists). Use `--lang es` for an ES voice demo so the greeting starts
in Spanish.

## Scaling & Bottlenecks

The real bottleneck at this product's scale is **LLM-provider I/O** — rate limits and
per-conversation token cost — not compute. The design reflects that:

- **Stateless** engine via the `ConversationStore` interface — horizontal scaling is a
  deployment concern (N replicas behind a load balancer), not a rewrite.
- **Async + a concurrency limiter** on all provider calls to respect rate limits.
- **Dockerfile** for deployment (the proportionate infra artifact — no Kubernetes
  theater for a service this size).

Production path: stateless replicas behind a load balancer; durable execution
(e.g. Temporal) for long or resumable interviews; a **real-time native-audio API** (e.g. Gemini Live) for real-time
voice; cost-aware multi-model routing (cheap model for extraction, expensive only when
needed).

## Observability (MLflow Tracing)

Tracing is **opt-in** and a complete no-op by default — CI and standard runs never
touch `mlflow.db` or require a server.

**Enable:**
```bash
MLFLOW_TRACING=1 python cli.py        # traces land in ./mlflow.db (SQLite, gitignored)
# or point at a remote server:
MLFLOW_TRACKING_URI=http://localhost:5000 python cli.py
```

**What each trace contains:**

Each `handle_turn()` call produces **one MLflow trace** with:
- Root span `handle_turn` — decision-path attributes: `conversation_id`, `turn_index`,
  `state_before`, `language`
- Child span `extraction` — `turn_index`, `conv`
- Child span `respond` (when in COLLECTING state) — `state`, `conv`
- Nested Anthropic SDK auto-spans (via `mlflow.anthropic.autolog`) carrying `input_tokens`,
  `output_tokens`, and per-call latency — no changes to `llm.py` required

**Inspect:**
```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow ui   # opens http://127.0.0.1:5000
```

Experiment name: `orbio-screening`. One run per conversation; one trace per turn.

## Potential Improvements

- Web UI (FastAPI/SSE) for the reviewer panel.
- LLM-as-a-judge scoring for summary quality (kept out of the core precision/FP metric
  on purpose).
- Additional languages (architecture supports it via config; only EN/ES are exercised
  here).
- Persistent store (Redis/Postgres) behind the existing `ConversationStore` interface.

## Demo

<!-- TODO(execute): link the demo video showing one voice conversation + the reviewer
     output, plus one EN and one ES text conversation. -->
```
