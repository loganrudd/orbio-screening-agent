"""Modality adapter: text (baseline) and voice (discrete STT -> engine -> TTS).

Voice is a thin, decoupled adapter — NOT real-time streaming, NOT a native-audio API
(that is the production path, noted in the README only). The conversation engine never
knows which modality is in use. STT word timestamps feed provenance; STT word
confidence feeds rule-derived confidence. The voice provider is independent of the LLM
provider. See voice.md.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional


@dataclass
class CandidateInput:
    text: str
    audio_start_s: Optional[float] = None
    audio_end_s: Optional[float] = None
    stt_confidence: Optional[float] = None  # 0..1, voice only


class ModalityAdapter(abc.ABC):
    """Turns raw candidate input into text for the engine, and agent text into output."""

    @abc.abstractmethod
    async def read_candidate(self) -> CandidateInput:
        ...

    @abc.abstractmethod
    async def emit_agent(self, text: str) -> None:
        ...


class TextAdapter(ModalityAdapter):
    """Baseline: stdin/stdout. The engine's primary, always-available surface."""

    async def read_candidate(self) -> CandidateInput:
        # input() blocks the event loop; acceptable for single-user CLI.
        text = input("You: ").strip()
        return CandidateInput(text=text)

    async def emit_agent(self, text: str) -> None:
        print(f"\nAgent: {text}\n")


class VoiceAdapter(ModalityAdapter):
    """Discrete voice pipeline on Deepgram (single API key, no cloud-project setup).

      read_candidate:  capture/record audio -> Deepgram STT ->
                       text + word timestamps + confidence
      emit_agent:      Deepgram Aura TTS -> play audio

    Degrades to text mode cleanly if STT/TTS is unconfigured or fails. The TTS provider
    can be swapped (e.g. ElevenLabs) without touching the engine.
    """

    def __init__(self, *, language: str = "en") -> None:
        self._language = language
        self._text_fallback = TextAdapter()
        # Phase 3: init Deepgram client (key from env DEEPGRAM_API_KEY).
        # WebFetch deepgram-sdk v3.x docs before implementing to confirm exact API.

    async def read_candidate(self) -> CandidateInput:
        # Phase 3: record -> Deepgram STT -> CandidateInput with word timestamps + confidence.
        # Degrades to text fallback until Phase 3.
        return await self._text_fallback.read_candidate()

    async def emit_agent(self, text: str) -> None:
        # Phase 3: Deepgram Aura TTS -> playback.
        # Degrades to text fallback until Phase 3.
        await self._text_fallback.emit_agent(text)
