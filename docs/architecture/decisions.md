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

## Scope: what was intentionally left out, and why
- Kubernetes manifests (Dockerfile is proportionate); web UI (CLI demonstrates the
  contract); multi-agent framework (state machine is clearer); database (JSON via the
  store interface is enough at this scope). _TODO: expand._
