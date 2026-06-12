"""CLI entry point — primary dev and demo surface.

Wires the modality adapter (text by default, voice with --voice) to the conversation
engine. The engine is modality-agnostic; this file chooses the adapter.

Usage:
    python cli.py                 # text mode
    python cli.py --voice         # voice mode (Deepgram STT/TTS — Phase 3)
    python cli.py --lang es        # start in Spanish
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys

from agent.conversation import ConversationEngine
from agent.extraction import Extractor
from agent.llm import ClaudeClient
from agent.output import build_summary, render_reviewer_table
from agent.storage import JsonFileStore, Turn
from agent.voice import TextAdapter, VoiceAdapter


async def run(voice: bool, language: str) -> None:
    store = JsonFileStore()
    llm = ClaudeClient()
    extractor = Extractor(llm)
    engine = ConversationEngine(store=store, llm=llm, extractor=extractor)
    adapter = VoiceAdapter(language=language) if voice else TextAdapter()

    # Start conversation and emit greeting
    conversation_id, greeting = await engine.start(language)
    await adapter.emit_agent(greeting.text)

    # Turn loop
    while True:
        candidate_input = await adapter.read_candidate()
        text = candidate_input.text.strip()

        if not text:
            continue  # empty input — re-prompt without advancing state

        turn = Turn(
            role="candidate",
            content=text,
            ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            audio_start_s=candidate_input.audio_start_s,
            audio_end_s=candidate_input.audio_end_s,
            stt_confidence=candidate_input.stt_confidence,
            words=candidate_input.words,
        )

        reply = await engine.handle_turn(conversation_id, text, turn)
        await adapter.emit_agent(reply.text)

        # Reviewer panel is backend-only — printed directly, never spoken via the adapter.
        if reply.reviewer_output:
            print(reply.reviewer_output)

        if reply.done:
            break

    # Print conversation ID so the user can find the saved transcript
    print(f"\n[Transcript saved: data/conversations/{conversation_id}.json]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restaurant screening agent")
    parser.add_argument("--voice", action="store_true", help="use voice (Deepgram STT/TTS)")
    parser.add_argument("--lang", default="en", help="starting language (en/es)")
    args = parser.parse_args()

    try:
        asyncio.run(run(voice=args.voice, language=args.lang))
    except KeyboardInterrupt:
        print("\n[Screening interrupted]", file=sys.stderr)


if __name__ == "__main__":
    main()
