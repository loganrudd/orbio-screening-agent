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
        # TODO(execute): read a line from stdin.
        raise NotImplementedError

    async def emit_agent(self, text: str) -> None:
        # TODO(execute): print.
        raise NotImplementedError


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
        # TODO(execute): init Deepgram client (key from env DEEPGRAM_API_KEY).

    async def read_candidate(self) -> CandidateInput:
        # TODO(execute): record -> Deepgram STT -> CandidateInput with word timestamps
        #   + confidence.
        raise NotImplementedError

    async def emit_agent(self, text: str) -> None:
        # TODO(execute): Deepgram Aura synth -> playback.
        raise NotImplementedError
