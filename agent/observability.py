"""Optional MLflow tracing for the conversation turn loop.

Wraps each turn to capture per-turn latency, token usage (via autolog), and the
decision path. Tracing activates when init_tracing() is called (cli.py does this
unconditionally). Tests never call init_tracing() so they are unaffected.

Architecture:
- init_tracing()   — call once at process start (cli.py). Always-on unless
                     MLFLOW_TRACING=0 opts out or mlflow is not importable.
- trace_root()     — root-span context manager; wraps the whole handle_turn() call.
- trace_turn()     — child-span context manager; wraps extraction / respond sub-steps.

All functions are safe to call when tracing is disabled: they are pure no-ops and
never raise. A span creation failure also never propagates — tracing is observational.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Generator

import structlog

log = structlog.get_logger(__name__)

try:
    import mlflow  # type: ignore

    _MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    _MLFLOW_AVAILABLE = False

_EXPERIMENT_NAME = "orbio-screening"
_enabled: bool = False  # set to True by init_tracing(); never mutated elsewhere


def _tracing_enabled() -> bool:
    """True only after init_tracing() has successfully completed."""
    return _enabled


def init_tracing() -> None:
    """Enable tracing for this process.

    Called once at CLI startup. Always-on unless:
    - mlflow is not importable, or
    - MLFLOW_TRACING=0 is set (explicit opt-out).

    Defaults to a local SQLite store (mlflow.db) — the MLflow 3.x recommended
    local backend. Override with MLFLOW_TRACKING_URI for a remote server.
    """
    global _enabled
    if not _MLFLOW_AVAILABLE:
        return
    if os.environ.get("MLFLOW_TRACING") == "0":
        return  # explicit opt-out
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(_EXPERIMENT_NAME)
        mlflow.anthropic.autolog(log_traces=True)
        _enabled = True
        log.info("mlflow_tracing_enabled", tracking_uri=tracking_uri, experiment=_EXPERIMENT_NAME)
    except Exception as exc:  # pragma: no cover
        log.warning("mlflow_init_failed", error=str(exc))


@contextlib.contextmanager
def trace_root(name: str, **attrs: Any) -> Generator[None, None, None]:
    """Root span for a single handle_turn() call.

    All child trace_turn() spans and autolog LLM spans nest under this one,
    producing one trace per turn in the MLflow UI. No-op when tracing is disabled.
    """
    if not _tracing_enabled():
        yield
        return
    _yielded = False
    try:
        with mlflow.start_span(name=name, attributes=attrs or None):
            _yielded = True
            yield
    except Exception as exc:
        if _yielded:
            raise  # body exception — propagate, don't yield a second time
        log.warning("mlflow_root_span_failed", name=name, error=str(exc))
        yield


@contextlib.contextmanager
def trace_turn(name: str, **attrs: Any) -> Generator[None, None, None]:
    """Child span for an extraction or respond sub-step within a turn.

    Usage (already wired in conversation.py):
        with trace_turn("extraction", turn_index=3, conv=conv_id):
            ...

    No-op when tracing is disabled; never raises.
    """
    if not _tracing_enabled():
        yield
        return
    _yielded = False
    try:
        with mlflow.start_span(name=name, attributes=attrs or None):
            _yielded = True
            yield
    except Exception as exc:
        if _yielded:
            raise  # body exception — propagate, don't yield a second time
        log.warning("mlflow_child_span_failed", name=name, error=str(exc))
        yield
