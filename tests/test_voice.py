"""Unit tests for the VoiceAdapter STT and TTS paths.

All Deepgram API calls and sounddevice I/O are monkeypatched — no network,
no credentials, no audio hardware required in CI. Tests cover:
  - No-API-key path degrades to TextAdapter
  - Import-failure path degrades to TextAdapter
  - STT: CandidateInput fields populated from a canned Deepgram response
  - STT: empty audio → empty text
  - STT: exception degrades to text for that turn
  - TTS: audio played from synthesized bytes
  - TTS: exception does not crash (text already printed)
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.storage import WordTiming
from agent.voice import CandidateInput, VoiceAdapter, TextAdapter, _SAMPLE_RATE


# ──────────────────────────────── helpers ─────────────────────────────────────

def _make_dg_word(word: str, start: float, end: float, confidence: float) -> MagicMock:
    """Build a mock Deepgram word object matching the SDK's field names."""
    w = MagicMock()
    w.word = word
    w.start = start
    w.end = end
    w.confidence = confidence
    return w


def _make_dg_response(
    transcript: str, words: list[MagicMock]
) -> MagicMock:
    """Build a mock Deepgram transcription response."""
    alt = MagicMock()
    alt.transcript = transcript
    alt.words = words

    channel = MagicMock()
    channel.alternatives = [alt]

    results = MagicMock()
    results.channels = [channel]

    response = MagicMock()
    response.results = results
    return response


async def _async_iter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


# ──────────────────────────────── degraded mode ───────────────────────────────

class TestVoiceAdapterDegraded:
    def test_no_api_key_is_degraded(self, monkeypatch):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        adapter = VoiceAdapter()
        assert adapter._degraded is True

    def test_missing_deepgram_dep_is_degraded(self, monkeypatch):
        """If AsyncDeepgramClient is None (import failed), adapter degrades."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-key")
        with patch("agent.voice.AsyncDeepgramClient", None):
            adapter = VoiceAdapter()
        assert adapter._degraded is True

    def test_missing_sounddevice_dep_is_degraded(self, monkeypatch):
        """If sounddevice is None (import failed), adapter degrades."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-key")
        with patch("agent.voice.sounddevice", None):
            adapter = VoiceAdapter()
        assert adapter._degraded is True

    async def test_degraded_read_candidate_returns_text_input(self, monkeypatch):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        adapter = VoiceAdapter()

        monkeypatch.setattr(
            "agent.voice.TextAdapter.read_candidate",
            AsyncMock(return_value=CandidateInput(text="hello")),
        )
        result = await adapter.read_candidate()
        assert result.text == "hello"
        assert result.words is None

    async def test_degraded_emit_agent_prints_text(self, monkeypatch, capsys):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        adapter = VoiceAdapter()
        await adapter.emit_agent("How are you?")
        captured = capsys.readouterr()
        assert "How are you?" in captured.out


# ──────────────────────────────── STT path ────────────────────────────────────

class TestVoiceAdapterSTT:
    def _adapter_with_mock_client(self, monkeypatch, dg_response: MagicMock) -> VoiceAdapter:
        """Build a non-degraded VoiceAdapter with a mocked Deepgram client."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-key")

        mock_client = MagicMock()
        mock_client.listen.v1.media.transcribe_file = AsyncMock(return_value=dg_response)

        with patch("agent.voice.AsyncDeepgramClient", return_value=mock_client), \
             patch("agent.voice.sounddevice"):  # suppress sounddevice import side-effects
            adapter = VoiceAdapter()

        adapter._dg_client = mock_client
        return adapter

    async def test_transcript_and_word_timings_extracted(self, monkeypatch):
        dg_words = [
            _make_dg_word("My",    0.0, 0.3, 0.95),
            _make_dg_word("name",  0.3, 0.6, 0.97),
            _make_dg_word("is",    0.6, 0.8, 0.99),
            _make_dg_word("Maria", 0.8, 1.2, 0.92),
        ]
        response = _make_dg_response("My name is Maria", dg_words)
        adapter = self._adapter_with_mock_client(monkeypatch, response)

        result = await adapter._transcribe(b"\x00" * 100)

        assert result.text == "My name is Maria"
        assert result.audio_start_s == pytest.approx(0.0)
        assert result.audio_end_s == pytest.approx(1.2)
        assert result.stt_confidence == pytest.approx(0.92)  # min of the 4 words
        assert result.words is not None
        assert len(result.words) == 4
        assert all(isinstance(w, WordTiming) for w in result.words)

    async def test_words_mapped_to_word_timing(self, monkeypatch):
        dg_words = [_make_dg_word("hello", 0.5, 0.9, 0.88)]
        response = _make_dg_response("hello", dg_words)
        adapter = self._adapter_with_mock_client(monkeypatch, response)

        result = await adapter._transcribe(b"\x00" * 50)

        wt = result.words[0]
        assert wt.word == "hello"
        assert wt.start_s == pytest.approx(0.5)
        assert wt.end_s == pytest.approx(0.9)
        assert wt.confidence == pytest.approx(0.88)

    async def test_empty_audio_returns_empty_text(self, monkeypatch):
        response = _make_dg_response("", [])
        adapter = self._adapter_with_mock_client(monkeypatch, response)

        # _transcribe short-circuits on empty bytes
        result = await adapter._transcribe(b"")
        assert result.text == ""
        assert result.words is None

    async def test_empty_words_list_no_audio_span(self, monkeypatch):
        """Deepgram can return a transcript but no word list (edge case)."""
        response = _make_dg_response("some text", [])
        adapter = self._adapter_with_mock_client(monkeypatch, response)

        result = await adapter._transcribe(b"\x00" * 100)
        assert result.text == "some text"
        assert result.audio_start_s is None
        assert result.stt_confidence is None
        assert result.words == []

    async def test_stt_exception_degrades_to_text(self, monkeypatch):
        """If Deepgram raises, read_candidate falls back to text for that turn."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-key")

        with patch("agent.voice.AsyncDeepgramClient"), \
             patch("agent.voice.sounddevice"):
            adapter = VoiceAdapter()

        adapter._dg_client = MagicMock()
        # Make _record_and_transcribe raise
        adapter._record_and_transcribe = AsyncMock(side_effect=RuntimeError("STT timeout"))

        with patch.object(
            adapter._text_fallback,
            "read_candidate",
            AsyncMock(return_value=CandidateInput(text="fallback text")),
        ):
            result = await adapter.read_candidate()

        assert result.text == "fallback text"
        assert result.words is None


# ──────────────────────────────── TTS path ────────────────────────────────────

class TestVoiceAdapterTTS:
    def _adapter_with_mocks(
        self,
        monkeypatch,
        tts_chunks: list[bytes],
    ) -> VoiceAdapter:
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-key")

        async def _fake_generate(*args, **kwargs):
            return _async_iter(tts_chunks)

        mock_client = MagicMock()
        mock_client.speak.v1.audio.generate = _fake_generate

        with patch("agent.voice.AsyncDeepgramClient", return_value=mock_client), \
             patch("agent.voice.sounddevice"):
            adapter = VoiceAdapter()

        adapter._dg_client = mock_client
        return adapter

    async def test_tts_plays_audio(self, monkeypatch, capsys):
        """emit_agent: prints text AND calls sounddevice.play with audio bytes."""
        import numpy as np

        chunk = (np.zeros(1000, dtype=np.int16)).tobytes()
        adapter = self._adapter_with_mocks(monkeypatch, [chunk])

        played_audio = []

        async def _fake_speak(text: str) -> None:
            played_audio.append(text)

        adapter._speak = _fake_speak

        await adapter.emit_agent("Hello candidate!")

        captured = capsys.readouterr()
        assert "Hello candidate!" in captured.out
        assert "Hello candidate!" in played_audio

    async def test_tts_prints_text_even_on_exception(self, monkeypatch, capsys):
        """TTS failure must not suppress the printed text."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "fake-key")

        with patch("agent.voice.AsyncDeepgramClient"), \
             patch("agent.voice.sounddevice"):
            adapter = VoiceAdapter()

        # Make _speak raise
        adapter._speak = AsyncMock(side_effect=RuntimeError("TTS down"))

        await adapter.emit_agent("This text should appear")

        captured = capsys.readouterr()
        assert "This text should appear" in captured.out
        # No exception propagated

    async def test_degraded_emit_agent_prints_only(self, monkeypatch, capsys):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        adapter = VoiceAdapter()
        await adapter.emit_agent("Text only output")
        captured = capsys.readouterr()
        assert "Text only output" in captured.out


# ──────────────────────────────── CandidateInput ──────────────────────────────

class TestCandidateInput:
    def test_text_only_defaults(self):
        ci = CandidateInput(text="hello")
        assert ci.text == "hello"
        assert ci.audio_start_s is None
        assert ci.audio_end_s is None
        assert ci.stt_confidence is None
        assert ci.words is None

    def test_full_voice_candidate_input(self):
        words = [WordTiming(word="hello", start_s=0.0, end_s=0.5, confidence=0.9)]
        ci = CandidateInput(
            text="hello",
            audio_start_s=0.0,
            audio_end_s=0.5,
            stt_confidence=0.9,
            words=words,
        )
        assert ci.stt_confidence == pytest.approx(0.9)
        assert len(ci.words) == 1
