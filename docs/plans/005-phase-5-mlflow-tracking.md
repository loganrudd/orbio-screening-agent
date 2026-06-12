# Plan 005: Phase 5 — MLflow Tracing

**Status:** Draft
**Phase:** 5 of 6
**Created:** 2026-06-12
**Files touched:** `agent/observability.py`, `agent/conversation.py` (one wrap), `cli.py` (init call), `tests/test_observability.py` (new), `README.md`, `docs/architecture/decisions.md`, `docs/plans/005-phase-5-mlflow-tracing.md` (this plan, persisted)

## Goal

A traced screening run produces an inspectable MLflow artifact — one trace per turn, with per-turn latency, token usage, and the decision path — while tracing stays a complete no-op by default (CI/tests emit nothing, no creds/server required).

## Context

CLAUDE.md scopes Phase 5 as "MLflow tracing on the turn loop (per-turn latency, token cost, decision path)" and the build-plan gate is "a traced run produces an inspectable artifact." Most of the scaffolding already exists:

- `agent/observability.py` is a stub with the right no-op shape (`try: import mlflow`) and a `trace_turn(name, **attrs)` context manager that currently only `yield`s. Its `TODO(execute)` already names the task: "use mlflow.start_span / genai tracing; record latency + token cost."
- `agent/conversation.py` is **already instrumented**: `trace_turn("extraction", …)` wraps extraction at line 98 and `trace_turn("respond", …)` wraps the reply LLM call at line 144. We fill in the implementation behind these, not add call sites.
- `agent/llm.py`'s `ClaudeClient` wraps `anthropic.AsyncAnthropic` and discards `response.usage` today — token capture is the gap.
- `mlflow>=3.0` is already in `requirements.txt`.

**Confirmed design decisions (this session):**
1. **Hybrid instrumentation** — `mlflow.anthropic.autolog()` auto-captures token usage + latency on every Claude call; flesh out `trace_turn()` to create the turn/decision-path spans those auto-spans nest under. Minimal change to `llm.py`.
2. **One trace per turn** — `handle_turn()` is the root span; extraction / respond / nested LLM calls are children. Matches the stateless "each turn = one request" model.
3. **Env-var gate only** — tracing activates only when an env var is set; off by default everywhere including CI.

## Approach

### Step 1 — Persist this plan
Write this plan to `docs/plans/005-phase-5-mlflow-tracing.md` (the in-repo durable copy; first execution action).

### Step 2 — Flesh out `agent/observability.py`
Add an explicit enablement gate and real span creation. Keep all of it import-safe and no-op when disabled.

- **Enablement gate** — a module-level `_tracing_enabled()` that returns True only when both: `mlflow` imported successfully AND an env var is set. Use `MLFLOW_TRACING` (truthy) as the primary switch; also treat a set `MLFLOW_TRACKING_URI` as enabling. Off → every function below is a pure no-op.
- **`init_tracing()`** — idempotent setup, called once at process start (from `cli.py`). When enabled: set the tracking URI (default to a local `./mlruns` file store if unset), set/confirm an experiment name (e.g. `orbio-screening`), and call `mlflow.anthropic.autolog()` so all `AsyncAnthropic` calls are auto-traced with token usage + latency. No-op (and never raises) when disabled or mlflow missing.
- **`trace_turn(name, **attrs)`** — when enabled, `mlflow.start_span(name=name)` and set `attrs` as span attributes; otherwise `yield` unchanged. Same signature the existing call sites already use, so `extraction` / `respond` spans light up for free. Wrap span creation in try/except so a tracing failure can never break a turn (observational only).
- **New `trace_root(name, **attrs)`** (or reuse `trace_turn`) — a root-span context manager used to wrap the whole turn, so extraction + respond + their nested autolog LLM spans group into **one trace per turn**.

Keep `from __future__ import annotations`, structured-logging-friendly, single responsibility. No pricing math required for the gate — "token cost" is satisfied by autolog's token counts; optionally set a derived USD attribute on the root span from the known Opus rates ($5/$25 per MTok) as a documented nicety, not core.

### Step 3 — Wrap the turn in `agent/conversation.py`
In `handle_turn()`, wrap the body in the root-span context manager so the existing `extraction` and `respond` child spans (and their autolog LLM spans) nest under one per-turn trace. Set decision-path attributes on the root span: `conversation_id`, `turn_index`, `state` (before/after), `language`, `outstanding_fields`, and `done`. This is the only conversation.py change — no logic moves.

### Step 4 — Initialize tracing in `cli.py`
Call `observability.init_tracing()` once at the top of `run()` (before the engine starts). No-op unless the env var is set, so the default CLI experience is unchanged.

### Step 5 — Tests: `tests/test_observability.py` (new)
Deterministic, no network, no live mlflow server required:
- **Disabled by default**: with env unset, `trace_turn(...)` / `trace_root(...)` are no-ops (context manager enters/exits cleanly, yields None) and `init_tracing()` does nothing / never raises — even when `mlflow` is importable.
- **mlflow-absent safety**: simulate import failure (monkeypatch `_MLFLOW_AVAILABLE`/`mlflow` to None) and assert the same no-op behavior — tracing is never required.
- **Enabled path (no real server)**: monkeypatch the env var on and stub/spy `mlflow.start_span` + `mlflow.anthropic.autolog` to assert spans are created with the expected `name` and attributes, and that the turn still completes. Assert on span **structure/attributes**, never on latency values (determinism rule).
- Optionally one engine-level test driving a mocked-LLM turn with tracing enabled, asserting a root span named for the turn wraps `extraction` + `respond` children.

Confirm the full suite still runs with `pytest` and that no `mlruns/` directory is created during a default (disabled) test run.

### Step 6 — Docs
- `docs/architecture/decisions.md` — add decision #13: hybrid autolog + manual spans, one-trace-per-turn, env-gated no-op default; note token cost = autolog token counts, and that the production evolution (real-time native-audio API) and richer per-route model cost tracking are future levers.
- `README.md` — short "Observability (MLflow tracing)" section: how to enable (`MLFLOW_TRACING=1`), what a trace contains (per-turn latency, token usage, decision path), and how to inspect it (`mlflow ui` against the local `./mlruns` store). One screenshot/线 optional for the demo.

## Tradeoffs Considered

- **Hybrid (chosen)** vs **manual-only** vs **autolog-only.** Manual-only gives total control but changes `LLMClient` return types and adds the most code for no real gain — autolog already captures usage/latency cleanly. Autolog-only is simplest but throws away the explicit decision-path structure the existing `trace_turn` call sites express (state, outstanding fields). Hybrid keeps `llm.py` essentially untouched, captures token cost for free, and preserves the structured turn/decision-path view — best fit for the "decision path" requirement in CLAUDE.md.
- **One trace per turn (chosen)** vs **per conversation.** Per-turn matches the stateless reload-per-turn design (each `handle_turn` = one request) and is the natural MLflow unit; per-conversation would need cross-turn span threading that fights the statelessness the architecture deliberately enforces.
- **Env-var gate (chosen)** vs import-presence gate. Because `mlflow>=3.0` is always importable in this repo, gating on import alone would emit traces (and create `mlruns/`) during every CI run — violating the "tracing never required / CI clean" rule. An explicit env var keeps the default a true no-op.

## Validation

- `pytest` — full suite green; new `test_observability.py` covers disabled-default, mlflow-absent, and enabled-with-stub paths. No `mlruns/` created on a default run.
- **Gate (manual/demo):** `MLFLOW_TRACING=1 python cli.py` for a short text screening, then `mlflow ui` → confirm one trace per turn, each with child `extraction` + `respond` spans, nested LLM spans carrying token usage + latency, and decision-path attributes on the root span. This is the inspectable artifact the phase gate requires.
- Re-run a default `python cli.py` (no env var) and confirm behavior/output is byte-for-byte unchanged and nothing is emitted.

## Open Questions

- Exact MLflow 3.x symbol names (`mlflow.start_span`, `mlflow.anthropic.autolog`) should be confirmed against the installed version during execution (the skill confirms `response.usage` shape; the MLflow API is verified at Step 2). If `mlflow.anthropic.autolog()` doesn't cover the async client in the pinned version, fall back to setting token attributes manually from `response.usage` inside `trace_turn` — keep that as the documented contingency, not the default path.
