"""Async concurrency limiter for provider (LLM/STT/TTS) calls.

All provider I/O passes through here to respect rate limits — the real scaling
bottleneck is provider I/O, not compute (see CLAUDE.md scalability). A semaphore is
enough at this scope; a token-bucket can replace it without changing callers.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int = 8) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)

    async def run(self, coro_fn: Callable[[], Awaitable[T]]) -> T:
        async with self._sem:
            return await coro_fn()

    # TODO(execute): optional token-bucket for per-minute provider rate limits +
    #   retry/backoff integration.
