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
import io
import os
import random
import wave
from dataclasses import dataclass
from typing import Optional

import structlog

from . import i18n
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

# httpx is a transitive dependency of the Deepgram SDK. Its timeout/transport
# exceptions leak through the SDK and — critically — have an EMPTY str(), so they
# can only be classified by TYPE, not by message substring. Guarded so the module
# still imports if httpx is somehow absent.
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

log = structlog.get_logger()

# STT model — language-independent
_DEFAULT_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-2")
# TTS model is language-keyed via i18n.tts_voice(); env override still wins.

# Audio recording settings
_SAMPLE_RATE = 16_000   # Hz — Deepgram prefers 16 kHz
_CHANNELS = 1
_DTYPE = "int16"        # 16-bit PCM

# Markers of TRANSIENT provider failures worth retrying. Deepgram's edge
# intermittently returns 401 INVALID_AUTH on a valid key (observed from distant
# regions, e.g. South America) — these are transient, not real credential
# errors, so we retry them here. The SDK's own retry layer deliberately skips
# 401. Network/transport and 5xx errors are also retried.
# Message-substring markers for transient failures that DO carry a message —
# chiefly Deepgram's JSON-body errors (intermittent edge 401s, 5xx). Network
# timeout/transport errors have an empty message and are matched by TYPE below.
_TRANSIENT_MARKERS = (
    "401", "INVALID_AUTH", "Invalid credentials",
    "500", "502", "503", "504", "INTERNAL", "Temporarily",
)
# Exception TYPES that are always transient regardless of message. httpx timeout
# and transport exceptions have an empty str(), so substring matching can never
# catch them — they must be classified by type. Builtin TimeoutError (== asyncio
# .TimeoutError) is what asyncio.wait_for raises on our per-call timeout below.
_TRANSIENT_EXC_TYPES: tuple[type[BaseException], ...] = (
    TimeoutError, ConnectionError,
) + ((httpx.TimeoutException, httpx.TransportError) if httpx is not None else ())

# Retry tuning. Edge blips can last a few seconds, so attempts must span a
# longer window than they cost — exponential backoff + jitter does that. All
# are env-tunable so a high-latency/distant network can dial them up without a
# code change.
_MAX_PROVIDER_ATTEMPTS = int(os.getenv("VOICE_MAX_RETRIES", "5"))
_RETRY_BASE_DELAY_S = float(os.getenv("VOICE_RETRY_BASE_DELAY_S", "0.5"))
_RETRY_MAX_DELAY_S = float(os.getenv("VOICE_RETRY_MAX_DELAY_S", "8.0"))
# Per-call timeout (synthesis/transcription only — NOT mic capture or playback).
# A hung connection trips this, raising TimeoutError, which is transient → retried.
_CALL_TIMEOUT_S = float(os.getenv("VOICE_CALL_TIMEOUT_S", "30.0"))


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int) -> bytes:
    """Wrap raw 16-bit PCM in a WAV container so Deepgram auto-detects the format."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)   # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


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

      read_candidate:  VAD mic capture (auto-start/stop on speech) → Deepgram STT →
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
            log.warning(
                "voice.stt_failed",
                error_type=type(exc).__name__,
                error=str(exc) or repr(exc),
                fallback="text mode for this turn",
            )
            return await self._text_fallback.read_candidate()

    async def _record_and_transcribe(self) -> CandidateInput:
        """Capture mic audio (push-to-talk), transcribe via Deepgram, return CandidateInput."""
        audio_bytes = await asyncio.to_thread(self._record_audio)
        return await self._with_retry(lambda: self._transcribe(audio_bytes), label="stt")

    async def _with_retry(self, factory, *, label: str):
        """Run a provider call through the limiter, retrying transient failures
        (network timeouts/transport errors and Deepgram's intermittent edge 401s)
        with exponential backoff. Each attempt is bounded by _CALL_TIMEOUT_S so a
        hung connection fails fast into a retry instead of blocking the turn."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_PROVIDER_ATTEMPTS):
            try:
                return await asyncio.wait_for(
                    self._limiter.run(factory), timeout=_CALL_TIMEOUT_S
                )
            except Exception as exc:
                last_exc = exc
                # Transient if the TYPE is a known timeout/transport error (these
                # have an empty message) OR the message matches a transient marker.
                is_transient = isinstance(exc, _TRANSIENT_EXC_TYPES) or any(
                    m in str(exc) for m in _TRANSIENT_MARKERS
                )
                is_last = attempt == _MAX_PROVIDER_ATTEMPTS - 1
                if not is_transient or is_last:
                    raise
                # Exponential backoff, capped, with jitter so retries don't all
                # land in lockstep inside the same edge blip.
                delay = min(_RETRY_BASE_DELAY_S * (2 ** attempt), _RETRY_MAX_DELAY_S)
                delay += random.uniform(0, _RETRY_BASE_DELAY_S)
                log.warning(
                    f"voice.{label}_retry",
                    attempt=attempt + 1,
                    of=_MAX_PROVIDER_ATTEMPTS,
                    delay_s=round(delay, 2),
                    error_type=type(exc).__name__,  # never blank, unlike str(exc)
                    error=str(exc)[:120] or repr(exc),
                )
                await asyncio.sleep(delay)
        assert last_exc is not None  # loop always sets it before raising
        raise last_exc

    def _record_audio(self) -> bytes:
        """Energy-based VAD mic capture. Returns raw PCM bytes.

        Opens the mic immediately. Calibrates background noise for ~400ms,
        then waits for speech onset (RMS > 3.5× noise floor). Records until
        ~1.5s of post-speech silence, then stops. No keypresses required.
        """
        import numpy as np

        _CHUNK_FRAMES = int(_SAMPLE_RATE * 0.05)   # 50ms per chunk
        _CALIB_CHUNKS = 8                           # ~400ms calibration window
        _PRE_BUFFER_N = 10                          # chunks prepended before onset (~0.5s)
        _SILENCE_STOP_N = 30                        # post-speech silence chunks to stop (~1.5s)
        _MIN_SPEECH_N = 6                           # minimum speech chunks to be valid (~0.3s)
        _MAX_CHUNKS = 600                           # hard stop at 30s

        print("\n[Listening...]", end="", flush=True)

        # --- Calibrate noise floor ---
        noise_rms: list[float] = []
        with sounddevice.InputStream(
            samplerate=_SAMPLE_RATE, channels=_CHANNELS, dtype=_DTYPE
        ) as stream:
            for _ in range(_CALIB_CHUNKS):
                data, _ = stream.read(_CHUNK_FRAMES)
                noise_rms.append(float(np.sqrt(np.mean(data.astype(np.float32) ** 2))))
        threshold = max(float(np.mean(noise_rms)) * 3.5, 150.0)

        # --- VAD recording loop ---
        frames: list[np.ndarray] = []
        pre_buffer: list[np.ndarray] = []
        speech_started = False
        silence_chunks = 0
        speech_chunks = 0

        with sounddevice.InputStream(
            samplerate=_SAMPLE_RATE, channels=_CHANNELS, dtype=_DTYPE
        ) as stream:
            for _ in range(_MAX_CHUNKS):
                data, _ = stream.read(_CHUNK_FRAMES)
                rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))

                if not speech_started:
                    pre_buffer.append(data.copy())
                    if len(pre_buffer) > _PRE_BUFFER_N:
                        pre_buffer.pop(0)
                    if rms > threshold:
                        speech_started = True
                        frames.extend(pre_buffer)
                        frames.append(data.copy())
                        speech_chunks = 1
                        silence_chunks = 0
                        print("\r[Recording ■]  ", end="", flush=True)
                else:
                    frames.append(data.copy())
                    if rms > threshold:
                        silence_chunks = 0
                        speech_chunks += 1
                    else:
                        silence_chunks += 1
                    if silence_chunks >= _SILENCE_STOP_N and speech_chunks >= _MIN_SPEECH_N:
                        break

        print("\r[Processing...]  ", flush=True)

        if not frames:
            return b""
        audio = np.concatenate(frames, axis=0)
        return audio.tobytes()

    async def _transcribe(self, audio_bytes: bytes) -> CandidateInput:
        """Send audio to Deepgram STT; parse transcript + per-word timings.

        Audio is wrapped in a WAV container before sending so Deepgram can
        auto-detect format — the transcribe_file API has no sample_rate param.
        """
        if not audio_bytes:
            return CandidateInput(text="")

        wav_bytes = _pcm_to_wav(audio_bytes, _SAMPLE_RATE, _CHANNELS)
        response = await self._dg_client.listen.v1.media.transcribe_file(
            request=wav_bytes,
            model=_DEFAULT_STT_MODEL,
            language=self._language,
            punctuate=True,
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

        # Synthesis is the network step — bounded by the per-call timeout and
        # retried on transient failures. Playback is local and happens once,
        # OUTSIDE retry, so a long reply's audio is never interrupted or replayed.
        try:
            audio = await self._with_retry(lambda: self._synthesize(text), label="tts")
        except Exception as exc:
            # Retries exhausted — text is already printed, so the conversation
            # continues uninterrupted in text for this turn.
            log.warning("voice.tts_failed", error_type=type(exc).__name__, error=str(exc) or repr(exc))
            return

        if audio is not None and audio.size:
            await asyncio.to_thread(sounddevice.play, audio, samplerate=_SAMPLE_RATE, blocking=True)

    async def _synthesize(self, text: str):
        """Synthesize text via Deepgram Aura TTS; return the decoded audio array.

        Network-only: returns the PCM samples for the caller to play. Playback is
        kept out of here so the per-call timeout/retry never spans audio playback.
        """
        import numpy as np

        # generate() is an async generator — do NOT await it, iterate directly.
        audio_chunks: list[bytes] = []
        async for chunk in self._dg_client.speak.v1.audio.generate(
            text=text,
            model=i18n.tts_voice(self._language),
            encoding="linear16",
            sample_rate=_SAMPLE_RATE,
        ):
            audio_chunks.append(chunk)

        if not audio_chunks:
            return None

        raw = b"".join(audio_chunks)
        return np.frombuffer(raw, dtype=np.int16)
