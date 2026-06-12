"""Modality adapter: text (baseline) and voice (discrete STT -> engine -> TTS).

Voice is a thin, decoupled adapter — NOT real-time streaming, NOT a native-audio API
(that is the production path, noted in the README only). The conversation engine never
knows which modality is in use. STT word timestamps feed per-field provenance; STT
word confidence feeds rule-derived confidence. The voice provider is independent of
the LLM provider. See .claude/rules/voice.md.

Deepgram SDK 7.3.1 API used here:
  STT: AsyncDeepgramClient().listen.v1.media.transcribe_file(request=bytes, ...)
       Response: response.results.channels[0].alternatives[0].{transcript, words}
       Each word: .word, .start, .end, .confidence
  TTS: AsyncDeepgramClient().speak.v1.audio.generate(text=str, model=str, ...)
       Returns: AsyncIterator[bytes]
"""

from __future__ import annotations

import abc
import asyncio
import os
from dataclasses import dataclass
from typing import Optional

import structlog

from .concurrency import ConcurrencyLimiter
from .storage import WordTiming

# Lazy module-level imports — defined here so tests can monkeypatch them.
# Both are guarded: the voice adapter degrades to text if either is absent.
try:
    from deepgram import AsyncDeepgramClient
except ImportError:
    AsyncDeepgramClient = None  # type: ignore[assignment,misc]

try:
    import sounddevice
except ImportError:
    sounddevice = None  # type: ignore[assignment]

log = structlog.get_logger()

# Deepgram Aura TTS voice — clear EN female voice. Swap via env for demos.
_DEFAULT_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en")
_DEFAULT_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-2")

# Audio recording settings (push-to-talk mic capture)
_SAMPLE_RATE = 16_000   # Hz — Deepgram prefers 16 kHz
_CHANNELS = 1
_DTYPE = "int16"        # 16-bit PCM


@dataclass
class CandidateInput:
    text: str
    audio_start_s: Optional[float] = None
    audio_end_s: Optional[float] = None
    stt_confidence: Optional[float] = None  # utterance-level min word-confidence (voice only)
    words: Optional[list[WordTiming]] = None  # per-word timings for field-level alignment


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
        text = await asyncio.to_thread(input, "You: ")
        return CandidateInput(text=text.strip())

    async def emit_agent(self, text: str) -> None:
        print(f"\nAgent: {text}\n")


class VoiceAdapter(ModalityAdapter):
    """Discrete voice pipeline on Deepgram (single API key, no cloud-project setup).

      read_candidate:  push-to-talk mic capture → Deepgram STT →
                       text + per-word timings + utterance confidence
      emit_agent:      Deepgram Aura TTS → audio bytes → sounddevice playback
                       (also prints the agent text so the terminal stays readable)

    Degrades to TextAdapter cleanly if DEEPGRAM_API_KEY is absent, import fails,
    or any single call errors. The TTS provider can be swapped (e.g. ElevenLabs)
    without touching the engine.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        limiter: Optional[ConcurrencyLimiter] = None,
    ) -> None:
        self._language = language
        self._limiter = limiter or ConcurrencyLimiter()
        self._text_fallback = TextAdapter()
        self._degraded = False
        self._dg_client = None

        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            log.warning("voice.degraded", reason="DEEPGRAM_API_KEY not set — using text mode")
            self._degraded = True
            return

        if AsyncDeepgramClient is None or sounddevice is None:
            log.warning("voice.degraded", reason="deepgram or sounddevice not installed — using text mode")
            self._degraded = True
            return

        self._dg_client = AsyncDeepgramClient()

    # ------------------------------------------------------------------ STT path

    async def read_candidate(self) -> CandidateInput:
        if self._degraded:
            return await self._text_fallback.read_candidate()
        try:
            return await self._record_and_transcribe()
        except Exception as exc:
            log.warning("voice.stt_failed", error=str(exc), fallback="text mode for this turn")
            return await self._text_fallback.read_candidate()

    async def _record_and_transcribe(self) -> CandidateInput:
        """Capture mic audio (push-to-talk), transcribe via Deepgram, return CandidateInput."""
        audio_bytes = await asyncio.to_thread(self._record_audio)
        return await self._limiter.run(lambda: self._transcribe(audio_bytes))

    def _record_audio(self) -> bytes:
        """Blocking mic capture: press Enter to start, Enter to stop. Returns raw PCM bytes."""
        import numpy as np

        print("\n[Voice] Press Enter to start recording...", flush=True)
        input()
        print("[Voice] Recording — press Enter to stop.", flush=True)

        frames: list[np.ndarray] = []

        def _callback(indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
            frames.append(indata.copy())

        with sounddevice.InputStream(
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype=_DTYPE,
            callback=_callback,
        ):
            input()  # block until second Enter

        print("[Voice] Processing...", flush=True)
        if not frames:
            return b""
        audio = np.concatenate(frames, axis=0)
        return audio.tobytes()

    async def _transcribe(self, audio_bytes: bytes) -> CandidateInput:
        """Send PCM audio to Deepgram STT; parse transcript + per-word timings."""
        if not audio_bytes:
            return CandidateInput(text="")

        response = await self._dg_client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model=_DEFAULT_STT_MODEL,
            language=self._language,
            punctuate=True,
            encoding="linear16",
            sample_rate=_SAMPLE_RATE,
        )

        alt = response.results.channels[0].alternatives[0]
        transcript = alt.transcript or ""

        dg_words = alt.words or []
        word_timings = [
            WordTiming(
                word=w.word,
                start_s=w.start,
                end_s=w.end,
                confidence=w.confidence,
            )
            for w in dg_words
        ]

        if word_timings:
            audio_start = word_timings[0].start_s
            audio_end = word_timings[-1].end_s
            stt_confidence = min(w.confidence for w in word_timings)
        else:
            audio_start = audio_end = stt_confidence = None

        log.debug(
            "voice.stt_done",
            transcript=transcript,
            word_count=len(word_timings),
            stt_confidence=stt_confidence,
        )

        return CandidateInput(
            text=transcript,
            audio_start_s=audio_start,
            audio_end_s=audio_end,
            stt_confidence=stt_confidence,
            words=word_timings,
        )

    # ------------------------------------------------------------------ TTS path

    async def emit_agent(self, text: str) -> None:
        print(f"\nAgent: {text}\n", flush=True)

        if self._degraded:
            return

        try:
            await self._limiter.run(lambda: self._speak(text))
        except Exception as exc:
            log.warning("voice.tts_failed", error=str(exc))
            # Text already printed — no further fallback needed.

    async def _speak(self, text: str) -> None:
        """Synthesize text via Deepgram Aura TTS and play it through sounddevice."""
        import numpy as np

        audio_chunks: list[bytes] = []
        async for chunk in await self._dg_client.speak.v1.audio.generate(
            text=text,
            model=_DEFAULT_TTS_MODEL,
            encoding="linear16",
            sample_rate=_SAMPLE_RATE,
        ):
            audio_chunks.append(chunk)

        if not audio_chunks:
            return

        raw = b"".join(audio_chunks)
        audio = np.frombuffer(raw, dtype=np.int16)
        await asyncio.to_thread(sounddevice.play, audio, samplerate=_SAMPLE_RATE, blocking=True)
