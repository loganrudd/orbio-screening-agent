"""Optional MLflow tracing for the conversation turn loop.

Wraps each turn (extraction, retrieval, LLM call) to capture per-turn latency, token
cost, and the decision path. No-op if MLflow is not configured — tracing must never be
required for the agent to run. See CLAUDE.md (MLflow scope) and eval.md.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

try:
    import mlflow  # type: ignore

    _MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    _MLFLOW_AVAILABLE = False


@contextlib.contextmanager
def trace_turn(name: str, **attrs: object) -> Iterator[None]:
    """Trace a turn/span if MLflow is available; otherwise a no-op.

    Usage:
        with trace_turn("extraction", turn_index=3):
            ...
    """
    if not _MLFLOW_AVAILABLE:
        yield
        return
    # TODO(execute): use mlflow.start_span / genai tracing; record latency + token cost
    #   + the attrs as span attributes.
    yield
