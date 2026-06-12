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
python cli.py            # text mode
python cli.py --voice    # voice mode (Deepgram STT in / Aura TTS out)
python cli.py --lang es  # start in Spanish
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
