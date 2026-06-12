"""Optional MLflow tracing for the conversation turn loop.

Wraps each turn to capture per-turn latency, token usage (via autolog), and the
decision path. Tracing is a complete no-op unless the env var MLFLOW_TRACING=1 (or
MLFLOW_TRACKING_URI) is set — CI and default runs never touch mlruns/.

Architecture:
- init_tracing()   — call once at process start (cli.py). Sets experiment, calls autolog.
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


def _tracing_enabled() -> bool:
    """True only when mlflow is importable AND an explicit env var opts in."""
    if not _MLFLOW_AVAILABLE:
        return False
    return bool(os.environ.get("MLFLOW_TRACING") or os.environ.get("MLFLOW_TRACKING_URI"))


def init_tracing() -> None:
    """Idempotent setup: set experiment and enable Anthropic autolog.

    Call once at process start (cli.py). No-op and never raises when disabled.
    """
    if not _tracing_enabled():
        return
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(_EXPERIMENT_NAME)
        mlflow.anthropic.autolog(log_traces=True)
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
