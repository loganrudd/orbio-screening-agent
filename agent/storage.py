"""Conversation persistence behind an interface (statelessness — see style.md).

All conversation state flows through `ConversationStore`. The engine holds no
in-process conversation state, so any replica can serve any turn. The JSON-file
implementation is sufficient at this scope; swapping in Redis/Postgres later is an
implementation change only.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

from .schemas import ConversationState, ScreeningRecord


@dataclass
class Turn:
    role: str  # "agent" | "candidate"
    content: str
    ts: str  # ISO8601
    audio_start_s: Optional[float] = None
    audio_end_s: Optional[float] = None


@dataclass
class ConversationSnapshot:
    """The complete state of one conversation — the unit of persistence."""

    conversation_id: str
    created_at: str
    language: str
    state: ConversationState
    transcript: list[Turn] = field(default_factory=list)
    record: Optional[ScreeningRecord] = None
    sentiment_timeline: list[dict] = field(default_factory=list)
    summary: Optional[str] = None


class ConversationStore(abc.ABC):
    """Interface for loading/persisting conversation state. Keep the engine on this."""

    @abc.abstractmethod
    def load(self, conversation_id: str) -> Optional[ConversationSnapshot]:
        ...

    @abc.abstractmethod
    def save(self, snapshot: ConversationSnapshot) -> None:
        ...

    @abc.abstractmethod
    def new_conversation(self, language: str) -> ConversationSnapshot:
        ...


class JsonFileStore(ConversationStore):
    """One JSON file per conversation under `data/conversations/{id}.json`.

    All writes must be wrapped so a write error never loses a transcript (see
    CLAUDE.md error handling).
    """

    def __init__(self, root: str = "data/conversations") -> None:
        self._root = root
        # TODO(execute): ensure root exists.

    def load(self, conversation_id: str) -> Optional[ConversationSnapshot]:
        # TODO(execute): read + deserialize; return None if absent.
        raise NotImplementedError

    def save(self, snapshot: ConversationSnapshot) -> None:
        # TODO(execute): serialize + atomic write (temp file + rename).
        raise NotImplementedError

    def new_conversation(self, language: str) -> ConversationSnapshot:
        # TODO(execute): mint uuid + timestamp, GREETING state, persist, return.
        raise NotImplementedError
