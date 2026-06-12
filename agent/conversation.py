"""Conversation state machine and turn loop.

Modality- and language-agnostic: consumes text, emits text. No STT/TTS here, no
hardcoded-language assumptions. Control flow is driven by structured state, not by
re-parsing the transcript. See conversation.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from .extraction import Extractor
from .llm import LLMClient
from .schemas import ConversationState, ScreeningRecord
from .storage import ConversationSnapshot, ConversationStore, Turn

MAX_REPROMPTS_PER_FIELD = 2


@dataclass
class AgentReply:
    text: str
    done: bool


class ConversationEngine:
    """Drives one screening conversation, persisting through the store each turn."""

    def __init__(
        self,
        *,
        store: ConversationStore,
        llm: LLMClient,
        extractor: Extractor,
    ) -> None:
        self._store = store
        self._llm = llm
        self._extractor = extractor
        self._reprompt_counts: dict[str, int] = {}

    async def start(self, language: str) -> tuple[str, AgentReply]:
        """Begin a conversation; returns (conversation_id, greeting reply)."""
        # TODO(execute): new_conversation, GREETING reply.
        raise NotImplementedError

    async def handle_turn(
        self, conversation_id: str, candidate_text: str, turn: Turn
    ) -> AgentReply:
        """Process one candidate turn and return the agent's next reply.

        Steps:
          1. load snapshot (statelessness — always from the store)
          2. append candidate turn to transcript
          3. extractor.extract_turn(...) -> merged record
          4. recompute outstanding fields; decide next state/prompt
          5. re-prompt only invalid/low-confidence/missing fields; cap then flag
          6. persist snapshot; return reply
        """
        # TODO(execute)
        raise NotImplementedError

    def _outstanding_fields(self, record: ScreeningRecord) -> list[str]:
        # TODO(execute): required minus confirmed.
        raise NotImplementedError

    def _next_state(
        self, snapshot: ConversationSnapshot, outstanding: list[str]
    ) -> ConversationState:
        # TODO(execute): COLLECTING -> CONFIRMING when complete -> SUMMARY.
        raise NotImplementedError
