"""Unit tests for agent/observability.py.

All tests run without a live MLflow server and without creating mlruns/.
Three paths are covered:
  1. Disabled by default (no env vars set) — pure no-op, nothing emitted.
  2. mlflow import absent — same no-op behavior regardless of env vars.
  3. Enabled via env var — spans are created with correct names/attributes.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, call, patch

import pytest

import agent.observability as obs
from agent.observability import init_tracing, trace_root, trace_turn


# ─────────────────────────────────────── helpers ──────────────────────────────


def _make_mock_span() -> MagicMock:
    """Return a mock that acts like an mlflow LiveSpan context manager."""
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    return span


# ─────────────────────────────── disabled by default ──────────────────────────


class TestDisabledByDefault:
    """When no env var is set tracing must be a pure no-op."""

    def test_tracing_enabled_returns_false(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACING", raising=False)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert obs._tracing_enabled() is False

    def test_init_tracing_does_nothing(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACING", raising=False)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        # Must not raise and must not create any mlflow state
        with patch.object(obs, "mlflow", create=True) as mock_mlflow:
            init_tracing()
            mock_mlflow.set_tracking_uri.assert_not_called()
            mock_mlflow.set_experiment.assert_not_called()

    def test_trace_turn_noop_yields(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACING", raising=False)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        ran = []
        with trace_turn("extraction", turn_index=0):
            ran.append(True)
        assert ran == [True]

    def test_trace_root_noop_yields(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACING", raising=False)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        ran = []
        with trace_root("handle_turn", conversation_id="abc", turn_index=0):
            ran.append(True)
        assert ran == [True]

    def test_trace_turn_noop_does_not_suppress_exceptions(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACING", raising=False)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        with pytest.raises(ValueError, match="boom"):
            with trace_turn("extraction"):
                raise ValueError("boom")

    def test_trace_root_noop_does_not_suppress_exceptions(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACING", raising=False)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        with pytest.raises(RuntimeError, match="crash"):
            with trace_root("handle_turn"):
                raise RuntimeError("crash")


# ──────────────────────────────── mlflow absent ───────────────────────────────


class TestMlflowAbsent:
    """Simulate mlflow not being importable — must behave identically to disabled."""

    def test_tracing_disabled_when_mlflow_unavailable(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", False)
        assert obs._tracing_enabled() is False

    def test_trace_turn_noop_when_unavailable(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", False)
        ran = []
        with trace_turn("extraction"):
            ran.append(True)
        assert ran == [True]

    def test_trace_root_noop_when_unavailable(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", False)
        ran = []
        with trace_root("handle_turn"):
            ran.append(True)
        assert ran == [True]

    def test_init_tracing_noop_when_unavailable(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", False)
        # Should complete without error even with env var set
        init_tracing()


# ───────────────────────────────── enabled path ───────────────────────────────


class TestEnabledPath:
    """When MLFLOW_TRACING=1, spans are created with the expected structure."""

    def test_tracing_enabled_with_mlflow_tracing_var(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        assert obs._tracing_enabled() is True

    def test_tracing_enabled_with_tracking_uri_var(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        monkeypatch.delenv("MLFLOW_TRACING", raising=False)
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        assert obs._tracing_enabled() is True

    def test_init_tracing_calls_setup(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        mock_mlflow = MagicMock()
        monkeypatch.setattr(obs, "mlflow", mock_mlflow)
        init_tracing()
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once_with(obs._EXPERIMENT_NAME)
        mock_mlflow.anthropic.autolog.assert_called_once_with(log_traces=True)

    def test_trace_turn_creates_span_with_name_and_attrs(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        mock_span = _make_mock_span()
        mock_mlflow = MagicMock()
        mock_mlflow.start_span.return_value = mock_span
        monkeypatch.setattr(obs, "mlflow", mock_mlflow)

        ran = []
        with trace_turn("extraction", turn_index=2, conv="test-conv"):
            ran.append(True)

        assert ran == [True]
        mock_mlflow.start_span.assert_called_once_with(
            name="extraction",
            attributes={"turn_index": 2, "conv": "test-conv"},
        )

    def test_trace_root_creates_span_with_decision_path_attrs(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        mock_span = _make_mock_span()
        mock_mlflow = MagicMock()
        mock_mlflow.start_span.return_value = mock_span
        monkeypatch.setattr(obs, "mlflow", mock_mlflow)

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
        """If mlflow.start_span raises, the body still runs (observational-only rule)."""
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        mock_mlflow = MagicMock()
        mock_mlflow.start_span.side_effect = RuntimeError("mlflow unavailable")
        monkeypatch.setattr(obs, "mlflow", mock_mlflow)

        ran = []
        with trace_turn("extraction"):
            ran.append(True)
        assert ran == [True]

    def test_trace_root_span_failure_degrades_gracefully(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        mock_mlflow = MagicMock()
        mock_mlflow.start_span.side_effect = RuntimeError("mlflow unavailable")
        monkeypatch.setattr(obs, "mlflow", mock_mlflow)

        ran = []
        with trace_root("handle_turn"):
            ran.append(True)
        assert ran == [True]

    def test_trace_turn_propagates_body_exception_when_span_succeeds(self, monkeypatch):
        """Span creation works but body raises — exception must propagate."""
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        mock_span = _make_mock_span()
        mock_mlflow = MagicMock()
        mock_mlflow.start_span.return_value = mock_span
        monkeypatch.setattr(obs, "mlflow", mock_mlflow)

        with pytest.raises(ValueError, match="body error"):
            with trace_turn("extraction"):
                raise ValueError("body error")

    def test_trace_turn_no_attrs_passes_none(self, monkeypatch):
        """Empty attrs dict should pass attributes=None to avoid polluting span."""
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)
        mock_span = _make_mock_span()
        mock_mlflow = MagicMock()
        mock_mlflow.start_span.return_value = mock_span
        monkeypatch.setattr(obs, "mlflow", mock_mlflow)

        with trace_turn("extraction"):
            pass

        mock_mlflow.start_span.assert_called_once_with(name="extraction", attributes=None)


# ──────────────────────────── integration with engine ─────────────────────────


class TestObservabilityWithEngine:
    """Verify handle_turn completes and trace_root is called when enabled."""

    async def test_handle_turn_completes_with_tracing_enabled(self, tmp_path, monkeypatch):
        """A full mocked turn runs to completion with MLFLOW_TRACING=1."""
        monkeypatch.setenv("MLFLOW_TRACING", "1")
        monkeypatch.setattr(obs, "_MLFLOW_AVAILABLE", True)

        root_span_calls: list[tuple] = []

        # Patch start_span so calls are recorded but context manager works
        real_contextmanager = __import__("contextlib").contextmanager

        @real_contextmanager
        def _fake_start_span(name, attributes=None):
            root_span_calls.append((name, attributes))
            span = MagicMock()
            span.set_attribute = MagicMock()
            yield span

        mock_mlflow = MagicMock()
        mock_mlflow.start_span.side_effect = _fake_start_span
        monkeypatch.setattr(obs, "mlflow", mock_mlflow)

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
        # handle_turn root span + at least one child span (extraction/respond)
        assert len(root_span_calls) >= 2
        root_names = [c[0] for c in root_span_calls]
        assert "handle_turn" in root_names
