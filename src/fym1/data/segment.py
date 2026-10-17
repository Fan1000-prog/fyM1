"""Cut a thread into short user/assistant conversations for training."""

from fym1.data.messenger import Message, Thread

Turn = dict[str, str]
Conversation = list[Turn]


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


def to_conversations(
    thread: Thread, owner: str, gap_minutes: int = 60, max_turns: int = 12
) -> list[Conversation]:
    """Conversations that start with a user turn, end with an assistant turn, have 2+ turns.

    A 1:1 thread yields each side once as the assistant. In a group thread the owner is
    the assistant and everyone else is merged into nameless user turns.
    """
    is_group = len(thread.participants) > 2
    assistants = [owner] if is_group else thread.participants
    sessions = _sessions(thread.messages, gap_minutes * 60_000)
    return [
        conversation
        for assistant in assistants
        for session in sessions
        for conversation in _chunks(_turns(session, assistant), max_turns)
    ]
