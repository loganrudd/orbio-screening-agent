"""Fixture-capture script for the eval harness.

Runs all seed transcripts through a real ClaudeClient (once), saves the validated
TurnExtraction proposals as JSON fixtures, and prints the live ScoreReport.

The captured fixtures are committed to data/seed_transcripts/recorded/ and replayed
deterministically by eval/replay.py in CI (no credentials needed).

Usage:
    ANTHROPIC_API_KEY=... python -m eval.record [--output-dir data/seed_transcripts/recorded]

This script is NEVER called in CI. It is a one-time capture tool.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Type, TypeVar

from agent.extraction import Extractor
from agent.llm import ClaudeClient, LLMClient
from agent.schemas import ScreeningRecord, TurnExtraction
from agent.storage import Turn
from eval.harness import run, score_transcript
from eval.replay import replay_extract_fn

T = TypeVar("T")


class RecordingLLM(LLMClient):
    """Wraps a real LLMClient, saving each validated TurnExtraction proposal to a list."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.recorded_proposals: list[dict] = []

    async def extract_structured(
        self, *, system: str, messages: list[dict], schema: Type[T]
    ) -> T:
        result = await self._inner.extract_structured(system=system, messages=messages, schema=schema)
        if isinstance(result, TurnExtraction):
            self.recorded_proposals.append(result.model_dump(mode="json"))
        return result

    async def respond(self, *, system: str, messages: list[dict]) -> str:
        return await self._inner.respond(system=system, messages=messages)


async def record_seed(transcript: dict, output_dir: Path) -> dict:
    """Run one seed transcript through real Claude, save proposals, return extracted dict."""
    seed_id = transcript["id"]
    language = transcript.get("language", "en")
    candidate_turns = [t for t in transcript.get("turns", []) if t["role"] == "candidate"]

    real_llm = ClaudeClient()
    recording_llm = RecordingLLM(real_llm)
    extractor = Extractor(llm=recording_llm)
    record = ScreeningRecord()

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

    # Save recorded proposals
    output_dir.mkdir(parents=True, exist_ok=True)
    proposals_file = output_dir / f"{seed_id}.json"
    proposals_file.write_text(json.dumps(recording_llm.recorded_proposals, indent=2))
    print(f"  Saved {len(recording_llm.recorded_proposals)} proposals → {proposals_file}")

    return record.model_dump(mode="json")


async def main(output_dir: Path) -> None:
    from eval.harness import load_seed_transcripts

    transcripts = load_seed_transcripts()
    if not transcripts:
        print("No seed transcripts found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nRecording {len(transcripts)} seed transcript(s) through real Claude...\n")

    for transcript in transcripts:
        seed_id = transcript.get("id", "?")
        print(f"▶ {seed_id}")
        try:
            extracted = await record_seed(transcript, output_dir)
            scores = score_transcript(transcript, extracted)
            correct = scores["correct_stated"]
            claimed = scores["claimed_collected"]
            fp = scores["false_positive"]
            fn = scores["false_negative"]
            print(f"  correct={correct}, claimed={claimed}, FP={fp}, FN={fn}")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
        print()

    print("Running offline replay to verify fixtures...\n")
    extract_fn = replay_extract_fn(str(output_dir))
    try:
        report = run(extract_fn)
        print(report.render())
    except FileNotFoundError as e:
        print(f"Warning: could not run offline replay: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture TurnExtraction proposals from real Claude and save as fixtures."
    )
    parser.add_argument(
        "--output-dir",
        default="data/seed_transcripts/recorded",
        help="Directory to write recorded proposals (default: data/seed_transcripts/recorded)",
    )
    args = parser.parse_args()
    asyncio.run(main(Path(args.output_dir)))
