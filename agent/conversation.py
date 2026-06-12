"""Conversation state machine and turn loop.

Modality- and language-agnostic: consumes text, emits text. No STT/TTS here, no
hardcoded-language assumptions. Control flow is driven by structured state, not by
re-parsing the transcript. See conversation.md.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from . import i18n
from .extraction import Extractor
from .llm import LLMClient
from .observability import trace_root, trace_turn
from .output import build_summary, render_candidate_confirmation, render_reviewer_table
from .schemas import ConversationState, ScreeningRecord
from .storage import ConversationSnapshot, ConversationStore, Turn

MAX_REPROMPTS_PER_FIELD = 2


@dataclass
class AgentReply:
    text: str
    done: bool
    reviewer_output: str | None = None  # backend-only; not spoken to the candidate


class ConversationEngine:
    """Drives one screening conversation, persisting all state through the store."""

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
        # NOTE: no in-process conversation state — reprompt_counts live in snapshot.

    async def start(self, language: str, *, auto_detect: bool = False) -> tuple[str, AgentReply]:
        """Begin a conversation; returns (conversation_id, greeting reply)."""
        snapshot = self._store.new_conversation(language, auto_detect=auto_detect)
        greeting = i18n.greeting(language)

        agent_turn = Turn(
            role="agent",
            content=greeting,
            ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        snapshot.transcript.append(agent_turn)
        self._store.save(snapshot)

        return snapshot.conversation_id, AgentReply(text=greeting, done=False)

    async def handle_turn(
        self, conversation_id: str, candidate_text: str, turn: Turn
    ) -> AgentReply:
        """Process one candidate turn and return the agent's next reply.

        1. Load snapshot (statelessness — always from the store)
        2. Append candidate turn to transcript
        3. Run extraction for COLLECTING / CONFIRMING states
        4. Recompute outstanding fields and advance state
        5. Re-prompt cap: accept MISSING if field hit max prompts
        6. Generate reply; persist; return
        """
        snapshot = self._store.load(conversation_id)
        if snapshot is None:
            raise ValueError(f"Unknown conversation: {conversation_id!r}")

        candidate_turns_before = [t for t in snapshot.transcript if t.role == "candidate"]
        turn_index = len(candidate_turns_before)

        with trace_root(
            "handle_turn",
            conversation_id=conversation_id,
            turn_index=turn_index,
            state_before=snapshot.state.value,
            language=snapshot.language,
        ):
            # Append candidate turn
            snapshot.transcript.append(turn)

            # Initialize record on first real turn
            if snapshot.record is None:
                snapshot.record = ScreeningRecord()

            # Transition GREETING → COLLECTING on the first candidate response.
            # Language detection runs exactly once here (before any extraction).
            if snapshot.state == ConversationState.GREETING:
                if snapshot.auto_detect:
                    detected = i18n.detect_language(turn.content)
                    if detected:
                        snapshot.language = detected
                snapshot.state = ConversationState.COLLECTING

            # Run extraction while collecting or confirming (corrections)
            if snapshot.state in (ConversationState.COLLECTING, ConversationState.CONFIRMING):
                with trace_turn("extraction", turn_index=turn_index, conv=conversation_id):
                    snapshot.record = await self._extractor.extract_turn(
                        record=snapshot.record,
                        latest_turn=turn,
                        turn_index=turn_index,
                        language=snapshot.language,
                    )

            outstanding = self._outstanding_fields(snapshot)

            # Advance state
            if snapshot.state == ConversationState.COLLECTING and not outstanding:
                snapshot.state = ConversationState.CONFIRMING
            elif snapshot.state == ConversationState.CONFIRMING:
                snapshot.state = ConversationState.SUMMARY

            # Generate the reply based on the new state
            reply_text, done, reviewer_output = await self._generate_reply(snapshot, outstanding)

            # Append agent turn
            agent_turn = Turn(
                role="agent",
                content=reply_text,
                ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            snapshot.transcript.append(agent_turn)
            self._store.save(snapshot)

        return AgentReply(text=reply_text, done=done, reviewer_output=reviewer_output)

    async def _generate_reply(
        self, snapshot: ConversationSnapshot, outstanding: list[str]
    ) -> tuple[str, bool, str | None]:
        if snapshot.state == ConversationState.COLLECTING:
            # Increment reprompt count for the field we're about to ask about
            if outstanding:
                next_field = outstanding[0]
                snapshot.reprompt_counts[next_field] = (
                    snapshot.reprompt_counts.get(next_field, 0) + 1
                )

            messages = _build_messages(snapshot)
            # Seed with a minimal user message if transcript has only agent turns so far
            if not messages:
                messages = [{"role": "user", "content": "(start of conversation)"}]

            with trace_turn("respond", state="collecting", conv=snapshot.conversation_id):
                reply_text = await self._llm.respond(
                    system=_build_conversation_system(snapshot, outstanding),
                    messages=messages,
                )
            return reply_text, False, None

        if snapshot.state == ConversationState.CONFIRMING:
            # Candidate sees values only — no confidence scores or internal flags.
            s = i18n.get_strings(snapshot.language)
            field_list = render_candidate_confirmation(snapshot.record, snapshot.language)
            reply_text = f"{s.confirming_intro}{field_list}{s.confirming_outro}"
            return reply_text, False, None

        if snapshot.state == ConversationState.SUMMARY:
            table = render_reviewer_table(snapshot.record)
            summary = build_summary(snapshot.record)
            snapshot.summary = summary
            # reviewer_output is backend-only — printed to the terminal but not
            # spoken to the candidate (the adapter only receives reply_text).
            reviewer_output = f"{table}\n{summary}"
            reply_text = i18n.closing(snapshot.language)
            return reply_text, True, reviewer_output

        return i18n.fallback(snapshot.language), True, None

    def _outstanding_fields(self, snapshot: ConversationSnapshot) -> list[str]:
        """Required fields still None AND below the re-prompt cap."""
        required = ScreeningRecord.required_fields()
        outstanding = []
        for field_name in required:
            ef = getattr(snapshot.record, field_name) if snapshot.record else None
            if ef is None:
                count = snapshot.reprompt_counts.get(field_name, 0)
                if count <= MAX_REPROMPTS_PER_FIELD:
                    outstanding.append(field_name)
        return outstanding

    def _next_state(
        self, snapshot: ConversationSnapshot, outstanding: list[str]
    ) -> ConversationState:
        """Determine next conversation state (used by tests; handle_turn inlines logic)."""
        if snapshot.state == ConversationState.COLLECTING and not outstanding:
            return ConversationState.CONFIRMING
        if snapshot.state == ConversationState.CONFIRMING:
            return ConversationState.SUMMARY
        return snapshot.state


# --------------------------------------------------------------------------- helpers


def _build_messages(snapshot: ConversationSnapshot) -> list[dict]:
    """Build the messages array for the LLM, starting from the first candidate turn.

    Anthropic requires the first message to be 'user'. The agent greeting precedes
    any candidate message, so we skip turns before the first candidate turn.
    """
    messages: list[dict] = []
    for turn in snapshot.transcript:
        if turn.role == "candidate":
            messages.append({"role": "user", "content": turn.content})
        elif turn.role == "agent" and messages:
            # Only include agent turns after the first candidate turn (preserves alternation)
            messages.append({"role": "assistant", "content": turn.content})
    return messages


def _build_conversation_system(
    snapshot: ConversationSnapshot, outstanding: list[str]
) -> str:
    collected_parts: list[str] = []
    record = snapshot.record
    if record:
        for fname in ScreeningRecord.required_fields():
            ef = getattr(record, fname)
            if ef is not None:
                collected_parts.append(f"  - {fname}: {ef.value!r}")

    collected_str = "\n".join(collected_parts) if collected_parts else "  (none yet)"
    outstanding_str = (
        "\n".join(f"  - {f}" for f in outstanding) if outstanding else "  (all collected)"
    )

    return (
        "You are a friendly, efficient restaurant hiring agent conducting a brief phone screening.\n\n"
        "You are screening candidates for positions at a restaurant chain "
        "(servers, line cooks, hosts, shift managers).\n\n"
        f"Already collected:\n{collected_str}\n\n"
        f"Still needed:\n{outstanding_str}\n\n"
        "Instructions:\n"
        "- Ask concisely about ONE outstanding field at a time (1-2 sentences max)\n"
        "- Be warm but professional\n"
        "- For availability, mention the options: weekday day/evening, weekend day/evening\n"
        "- For start date, ask when they can start\n"
        "- Do NOT ask about location preference unless the candidate raises it\n"
        f"- Language: {snapshot.language}"
    )
