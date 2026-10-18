"""Cut a thread into short user/assistant conversations for training."""

from fym1.data.messenger import Message, Thread

Turn = dict[str, str]
Conversation = list[Turn]
GAP_MINUTES = 60  # a silence longer than this starts a new session


def _sessions(messages: list[Message], gap_ms: int) -> list[list[Message]]:
    """Split time-ordered messages wherever the silence is longer than `gap_ms`."""
    sessions: list[list[Message]] = []
    for message in messages:
        if sessions and message.ts_ms - sessions[-1][-1].ts_ms <= gap_ms:
            sessions[-1].append(message)
        else:
            sessions.append([message])
    return sessions


def _turns(session: list[Message], assistant: str) -> Conversation:
    """One turn per run of messages from the same side; `assistant` speaks as assistant."""
    turns: Conversation = []
    for message in session:
        role = "assistant" if message.sender == assistant else "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + message.text
        else:
            turns.append({"role": role, "content": message.text})
    return turns


def _trim(turns: Conversation) -> Conversation:
    """Drop a leading assistant turn and a trailing user turn."""
    start = 1 if turns and turns[0]["role"] == "assistant" else 0
    end = len(turns) - 1 if turns and turns[-1]["role"] == "user" else len(turns)
    return turns[start:end]


def _chunks(turns: Conversation, max_turns: int) -> list[Conversation]:
    pieces = (_trim(turns[i : i + max_turns]) for i in range(0, len(turns), max_turns))
    return [piece for piece in pieces if len(piece) >= 2]


def count_sessions(thread: Thread, gap_minutes: int = GAP_MINUTES) -> int:
    """How many time-gap sessions `to_conversations` cuts `thread` into, before turns and trimming."""
    return len(_sessions(thread.messages, gap_minutes * 60_000))


def to_conversations(
    thread: Thread, owner: str | None, gap_minutes: int = GAP_MINUTES, max_turns: int = 12
) -> list[Conversation]:
    """Conversations that start with a user turn, end with an assistant turn, have 2+ turns.

    A 1:1 thread yields each side once as the assistant. In a group thread the owner is
    the assistant and everyone else is merged into nameless user turns; with `owner=None`
    every participant takes that assistant seat in turn (exact duplicates are left to the caller).
    """
    is_group = len(thread.participants) > 2
    assistants = [owner] if is_group and owner is not None else thread.participants
    sessions = _sessions(thread.messages, gap_minutes * 60_000)
    return [
        conversation
        for assistant in assistants
        for session in sessions
        for conversation in _chunks(_turns(session, assistant), max_turns)
    ]
