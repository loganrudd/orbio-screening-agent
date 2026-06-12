"""Conversation persistence behind an interface (statelessness — see style.md).

All conversation state flows through `ConversationStore`. The engine holds no
in-process conversation state, so any replica can serve any turn. The JSON-file
implementation is sufficient at this scope; swapping in Redis/Postgres later is an
implementation change only.
"""

from __future__ import annotations

import abc
import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .schemas import ConversationState, ScreeningRecord


@dataclass
class WordTiming:
    """Per-word STT timing from speech recognition. Provider-neutral.

    Populated by the voice adapter (voice.py) and consumed by extraction.py
    for per-field source attribution. Kept transient — not persisted to disk
    because per-field spans are already stored inside Provenance.
    """

    word: str
    start_s: float
    end_s: float
    confidence: float  # 0..1, word-level STT confidence


@dataclass
class Turn:
    role: str  # "agent" | "candidate"
    content: str
    ts: str  # ISO8601
    audio_start_s: Optional[float] = None
    audio_end_s: Optional[float] = None
    stt_confidence: Optional[float] = None  # utterance-level min word-confidence (voice only)
    words: Optional[list["WordTiming"]] = None  # transient — not persisted; used for per-field alignment


@dataclass
class ConversationSnapshot:
    """The complete state of one conversation — the unit of persistence.

    reprompt_counts is persisted here (not in the engine) so that any replica
    picking up a conversation mid-flight can respect the cap correctly.
    """

    conversation_id: str
    created_at: str
    language: str
    state: ConversationState
    transcript: list[Turn] = field(default_factory=list)
    record: Optional[ScreeningRecord] = None
    sentiment_timeline: list[dict] = field(default_factory=list)
    summary: Optional[str] = None
    reprompt_counts: dict[str, int] = field(default_factory=dict)


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


# --------------------------------------------------------------------------- helpers


def _snapshot_to_dict(snapshot: ConversationSnapshot) -> dict:
    return {
        "conversation_id": snapshot.conversation_id,
        "created_at": snapshot.created_at,
        "language": snapshot.language,
        "state": snapshot.state.value,
        "transcript": [
            {
                "role": t.role,
                "content": t.content,
                "ts": t.ts,
                "audio_start_s": t.audio_start_s,
                "audio_end_s": t.audio_end_s,
                "stt_confidence": t.stt_confidence,
                # words is transient — not persisted; per-field spans live in Provenance
            }
            for t in snapshot.transcript
        ],
        # model_dump(mode="json") handles nested Pydantic models + computed fields
        "record": snapshot.record.model_dump(mode="json") if snapshot.record else None,
        "sentiment_timeline": snapshot.sentiment_timeline,
        "summary": snapshot.summary,
        "reprompt_counts": snapshot.reprompt_counts,
    }


def _snapshot_from_dict(d: dict) -> ConversationSnapshot:
    return ConversationSnapshot(
        conversation_id=d["conversation_id"],
        created_at=d["created_at"],
        language=d["language"],
        state=ConversationState(d["state"]),
        transcript=[
            Turn(
                role=t["role"],
                content=t["content"],
                ts=t["ts"],
                audio_start_s=t.get("audio_start_s"),
                audio_end_s=t.get("audio_end_s"),
                stt_confidence=t.get("stt_confidence"),
                # words is transient — always None on load
            )
            for t in d.get("transcript", [])
        ],
        # model_validate ignores the serialized computed_field 'score' (extra='ignore')
        record=ScreeningRecord.model_validate(d["record"]) if d.get("record") else None,
        sentiment_timeline=d.get("sentiment_timeline", []),
        summary=d.get("summary"),
        reprompt_counts=d.get("reprompt_counts", {}),
    )


# --------------------------------------------------------------- implementation


class JsonFileStore(ConversationStore):
    """One JSON file per conversation under `data/conversations/{id}.json`.

    All writes use an atomic temp-file + os.replace so a crash never corrupts a
    transcript mid-write.
    """

    def __init__(self, root: str = "data/conversations") -> None:
        self._root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, conversation_id: str) -> str:
        return os.path.join(self._root, f"{conversation_id}.json")

    def load(self, conversation_id: str) -> Optional[ConversationSnapshot]:
        try:
            with open(self._path(conversation_id)) as f:
                return _snapshot_from_dict(json.load(f))
        except FileNotFoundError:
            return None

    def save(self, snapshot: ConversationSnapshot) -> None:
        path = self._path(snapshot.conversation_id)
        data = json.dumps(_snapshot_to_dict(snapshot), indent=2)
        # Atomic write: write temp then rename — a crash cannot produce a partial file.
        with tempfile.NamedTemporaryFile(
            "w", dir=self._root, delete=False, suffix=".tmp"
        ) as f:
            f.write(data)
            tmp_path = f.name
        os.replace(tmp_path, path)

    def new_conversation(self, language: str) -> ConversationSnapshot:
        conversation_id = str(uuid.uuid4())
        snapshot = ConversationSnapshot(
            conversation_id=conversation_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            language=language,
            state=ConversationState.GREETING,
        )
        self.save(snapshot)
        return snapshot
