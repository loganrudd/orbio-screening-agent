"""Unit tests for agent/observability.py.

All tests run without a live MLflow server and without creating mlflow.db.
Three paths are covered:
  1. Disabled before init_tracing() — pure no-op, nothing emitted.
  2. mlflow import absent — same no-op behavior even if init_tracing() is called.
  3. Enabled after init_tracing() — spans are created with correct names/attributes.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import pytest

import agent.observability as obs
from agent.observability import init_tracing, trace_root, trace_turn


# ─────────────────────────────────────── helpers ──────────────────────────────


def _reset(monkeypatch) -> None:
    """Reset module-level _enabled flag between tests."""
    monkeypatch.setattr(obs, "_enabled", False)


def _make_mock_span() -> MagicMock:
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    return span


# ─────────────────── disabled before init_tracing() is called ─────────────────


class TestDisabledByDefault:
    """Before init_tracing() is called tracing must be a pure no-op."""

    def test_tracing_enabled_returns_false(self, monkeypatch):
        _reset(monkeypatch)
        assert obs._tracing_enabled() is False

    def test_init_tracing_does_nothing_when_mlflow_unavailable(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", False)
        init_tracing()
        assert obs._tracing_enabled() is False

    def test_init_tracing_respects_explicit_opt_out(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setenv("MLFLOW_TRACING", "0")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        mock_mlflow = MagicMock()
        monkeypatch.setattr(obs, "mlflow", mock_mlflow)
        init_tracing()
        assert obs._tracing_enabled() is False
        mock_mlflow.set_tracking_uri.assert_not_called()

    def test_trace_turn_noop_before_init(self, monkeypatch):
        _reset(monkeypatch)
        ran = []
        with trace_turn("extraction", turn_index=0):
            ran.append(True)
        assert ran == [True]

    def test_trace_root_noop_before_init(self, monkeypatch):
        _reset(monkeypatch)
        ran = []
        with trace_root("handle_turn", conversation_id="abc", turn_index=0):
            ran.append(True)
        assert ran == [True]

    def test_trace_turn_noop_does_not_suppress_exceptions(self, monkeypatch):
        _reset(monkeypatch)
        with pytest.raises(ValueError, match="boom"):
            with trace_turn("extraction"):
                raise ValueError("boom")

    def test_trace_root_noop_does_not_suppress_exceptions(self, monkeypatch):
        _reset(monkeypatch)
        with pytest.raises(RuntimeError, match="crash"):
            with trace_root("handle_turn"):
                raise RuntimeError("crash")


# ──────────────────────────────── mlflow absent ───────────────────────────────


class TestMlflowAbsent:
    """Simulate mlflow not being importable — init_tracing() stays a no-op."""

    def test_tracing_stays_disabled_when_mlflow_unavailable(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", False)
        init_tracing()
        assert obs._tracing_enabled() is False

    def test_trace_turn_noop_when_unavailable(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", False)
        ran = []
        with trace_turn("extraction"):
            ran.append(True)
        assert ran == [True]

    def test_trace_root_noop_when_unavailable(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", False)
        ran = []
        with trace_root("handle_turn"):
            ran.append(True)
        assert ran == [True]


# ───────────────────────────── enabled after init ─────────────────────────────


class TestEnabledPath:
    """After init_tracing() succeeds, spans are created with expected structure."""

    def _init(self, monkeypatch) -> MagicMock:
        """Call init_tracing() with a stubbed mlflow; return the mock."""
        _reset(monkeypatch)
        monkeypatch.delenv("MLFLOW_TRACING", raising=False)
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        mock_mlflow = MagicMock()
        monkeypatch.setattr(obs, "mlflow", mock_mlflow)
        init_tracing()
        assert obs._tracing_enabled() is True
        return mock_mlflow

    def test_init_tracing_calls_setup(self, monkeypatch):
        mock_mlflow = self._init(monkeypatch)
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once_with(obs._EXPERIMENT_NAME)
        mock_mlflow.anthropic.autolog.assert_called_once_with(log_traces=True)

    def test_init_tracing_uses_custom_tracking_uri(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        mock_mlflow = MagicMock()
        monkeypatch.setattr(obs, "mlflow", mock_mlflow)
        init_tracing()
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_trace_turn_creates_span_with_name_and_attrs(self, monkeypatch):
        mock_mlflow = self._init(monkeypatch)
        mock_span = _make_mock_span()
        mock_mlflow.start_span.return_value = mock_span

        ran = []
        with trace_turn("extraction", turn_index=2, conv="test-conv"):
            ran.append(True)

        assert ran == [True]
        mock_mlflow.start_span.assert_called_once_with(
            name="extraction",
            attributes={"turn_index": 2, "conv": "test-conv"},
        )

    def test_trace_root_creates_span_with_decision_path_attrs(self, monkeypatch):
        mock_mlflow = self._init(monkeypatch)
        mock_span = _make_mock_span()
        mock_mlflow.start_span.return_value = mock_span

        ran = []
        with trace_root(
            "handle_turn",
            conversation_id="abc-123",
            turn_index=1,
            state_before="collecting",
            language="en",
        ):
            ran.append(True)

        assert ran == [True]
        mock_mlflow.start_span.assert_called_once_with(
            name="handle_turn",
            attributes={
                "conversation_id": "abc-123",
                "turn_index": 1,
                "state_before": "collecting",
                "language": "en",
            },
        )

    def test_trace_turn_span_failure_degrades_gracefully(self, monkeypatch):
        """If mlflow.start_span raises, the body still runs."""
        mock_mlflow = self._init(monkeypatch)
        mock_mlflow.start_span.side_effect = RuntimeError("mlflow unavailable")

        ran = []
        with trace_turn("extraction"):
            ran.append(True)
        assert ran == [True]

    def test_trace_root_span_failure_degrades_gracefully(self, monkeypatch):
        mock_mlflow = self._init(monkeypatch)
        mock_mlflow.start_span.side_effect = RuntimeError("mlflow unavailable")

        ran = []
        with trace_root("handle_turn"):
            ran.append(True)
        assert ran == [True]

    def test_trace_turn_propagates_body_exception_when_span_succeeds(self, monkeypatch):
        mock_mlflow = self._init(monkeypatch)
        mock_mlflow.start_span.return_value = _make_mock_span()

        with pytest.raises(ValueError, match="body error"):
            with trace_turn("extraction"):
                raise ValueError("body error")

    def test_trace_turn_no_attrs_passes_none(self, monkeypatch):
        mock_mlflow = self._init(monkeypatch)
        mock_mlflow.start_span.return_value = _make_mock_span()

        with trace_turn("extraction"):
            pass

        mock_mlflow.start_span.assert_called_once_with(name="extraction", attributes=None)


# ──────────────────────────── integration with engine ─────────────────────────


class TestObservabilityWithEngine:
    """Verify handle_turn completes and trace_root is called after init_tracing."""

    async def test_handle_turn_completes_with_tracing_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(obs, "_enabled", False)
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        monkeypatch.delenv("MLFLOW_TRACING", raising=False)

        root_span_calls: list[tuple] = []

        import contextlib as _cl

        @_cl.contextmanager
        def _fake_start_span(name, attributes=None):
            root_span_calls.append((name, attributes))
            yield MagicMock()

        mock_mlflow = MagicMock()
        mock_mlflow.start_span.side_effect = _fake_start_span
        monkeypatch.setattr(obs, "mlflow", mock_mlflow)
        init_tracing()

        from agent.conversation import ConversationEngine
        from agent.extraction import Extractor
        from agent.schemas import TurnExtraction
        from agent.storage import JsonFileStore, Turn
        from tests.helpers import MockLLM

        llm = MockLLM(extraction=TurnExtraction(), reply="Got it.")
        store = JsonFileStore(str(tmp_path))
        engine = ConversationEngine(store=store, llm=llm, extractor=Extractor(llm=llm))

        conv_id, _ = await engine.start("en")
        turn = Turn(
            role="candidate",
            content="Hi, I'm Alex",
            ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        reply = await engine.handle_turn(conv_id, "Hi, I'm Alex", turn)

        assert reply.done is False
        assert len(root_span_calls) >= 2
        assert "handle_turn" in [c[0] for c in root_span_calls]
