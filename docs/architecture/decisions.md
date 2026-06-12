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
**Why:** LLM self-reported confidence is poorly calibrated; rule-grounded confidence is
defensible and auditable. _TODO: expand._

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

## Scope: what was intentionally left out, and why
- Kubernetes manifests (Dockerfile is proportionate); web UI (CLI demonstrates the
  contract); multi-agent framework (state machine is clearer); database (JSON via the
  store interface is enough at this scope). _TODO: expand._
