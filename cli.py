"""CLI entry point — primary dev and demo surface.

Wires the modality adapter (text by default, voice with --voice) to the conversation
engine. The engine is modality-agnostic; this file chooses the adapter.

Usage:
    python cli.py                 # text mode
    python cli.py --voice         # voice mode (Google STT/TTS)
    python cli.py --lang es        # start in Spanish
"""

from __future__ import annotations

import argparse
import asyncio

from agent.conversation import ConversationEngine
from agent.extraction import Extractor
from agent.llm import ClaudeClient
from agent.output import build_summary, render_reviewer_table
from agent.storage import JsonFileStore
from agent.voice import TextAdapter, VoiceAdapter


async def run(voice: bool, language: str) -> None:
    store = JsonFileStore()
    llm = ClaudeClient()
    engine = ConversationEngine(store=store, llm=llm, extractor=Extractor(llm))
    adapter = VoiceAdapter(language=language) if voice else TextAdapter()

    # TODO(execute):
    #   conversation_id, greeting = await engine.start(language)
    #   await adapter.emit_agent(greeting.text)
    #   loop: read candidate input -> engine.handle_turn -> emit reply until done
    #   then render_reviewer_table(...) + build_summary(...)
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Restaurant screening agent")
    parser.add_argument("--voice", action="store_true", help="use voice (STT/TTS)")
    parser.add_argument("--lang", default="en", help="starting language (en/es)")
    args = parser.parse_args()
    asyncio.run(run(voice=args.voice, language=args.lang))


if __name__ == "__main__":
    main()
