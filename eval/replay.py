"""Offline replay infrastructure for the eval harness.

ReplayLLM: pops pre-recorded TurnExtraction proposals in call order.
replay_extract_fn: builds an extract_fn(transcript) → record dict that drives the
real Extractor over a seed transcript's candidate turns using a ReplayLLM.

This is the CI-safe path: deterministic, no network, no credentials.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime
import json
from collections import deque
from pathlib import Path
from typing import Any, Callable, Type, TypeVar

from agent.extraction import Extractor
from agent.llm import LLMClient
from agent.schemas import ScreeningRecord, TurnExtraction
from agent.storage import Turn

T = TypeVar("T")


class ReplayLLM(LLMClient):
    """Pops pre-recorded TurnExtraction proposals in call order.

    Raises RuntimeError if more calls are made than recorded proposals.
    `respond` always returns a fixed canned string (not relevant for eval scoring).
    """

    def __init__(self, proposals: list[dict]) -> None:
        self._queue: deque[dict] = deque(proposals)

    async def extract_structured(
        self, *, system: str, messages: list[dict], schema: Type[T]
    ) -> T:
        if not self._queue:
            raise RuntimeError("ReplayLLM exhausted: more extraction calls than recorded proposals")
        raw = self._queue.popleft()
        return schema.model_validate(raw)  # type: ignore[return-value]

    async def respond(self, *, system: str, messages: list[dict]) -> str:
        return "Thank you, noted."


def replay_extract_fn(recorded_dir: str) -> Callable[[dict], dict]:
    """Return an extract_fn(transcript) for offline harness runs.

    For each seed transcript, loads the recorded proposals from
    `recorded_dir/{seed_id}.json`, drives Extractor.extract_turn over every
    candidate turn, and returns ScreeningRecord.model_dump(mode="json").

    The returned dict includes 'flag' on every field so the harness can score
    conflicting-field handling.
    """
    recorded_path = Path(recorded_dir)

    def extract_fn(transcript: dict) -> dict:
        seed_id = transcript["id"]
        proposals_file = recorded_path / f"{seed_id}.json"
        if not proposals_file.exists():
            raise FileNotFoundError(
                f"No recorded proposals for seed '{seed_id}' at {proposals_file}. "
                "Run `python -m eval.record` first."
            )
        proposals: list[dict] = json.loads(proposals_file.read_text())

        llm = ReplayLLM(proposals)
        extractor = Extractor(llm=llm)
        record = ScreeningRecord()
        language = transcript.get("language", "en")

        candidate_turns = [
            t for t in transcript.get("turns", []) if t["role"] == "candidate"
        ]

        async def _run() -> ScreeningRecord:
            nonlocal record
            for idx, raw_turn in enumerate(candidate_turns):
                turn = Turn(
                    role="candidate",
                    content=raw_turn["content"],
                    ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
                record = await extractor.extract_turn(
                    record=record,
                    latest_turn=turn,
                    turn_index=idx,
                    language=language,
                )
            return record

        # Run in a fresh thread with its own event loop so this works both in
        # sync contexts and when called from within an already-running loop
        # (e.g. eval/record.py's asyncio.run(main(...))).
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            final = pool.submit(asyncio.run, _run()).result()
        return final.model_dump(mode="json")

    return extract_fn
