"""Gradio chat history -> the message list llama.cpp expects, kept inside the model's context window."""

from collections.abc import Callable

from generation import HISTORY_EXCHANGES, trim_history

# Rough cost of the chat template's role markers around each message, in tokens.
TOKENS_PER_MESSAGE_OVERHEAD = 8


class MessageTooLong(ValueError):
    """The system prompt plus the new message alone do not fit the token budget."""


def text_of(content) -> str:
    """Plain text of a message; Gradio 6 stores content as a list of typed parts."""
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content if part.get("type") == "text")


def build_messages(
    system_prompt: str,
    history: list[dict],
    message: str,
    count_tokens: Callable[[str], int],
    token_budget: int,
    max_exchanges: int = HISTORY_EXCHANGES,
) -> list[dict]:
    """System prompt, the last `max_exchanges` exchanges (as many as fit `token_budget`), then `message`.

    Oldest turns are dropped first, and the kept history starts on a user turn, as chat templates require.
    """

    def cost(text: str) -> int:
        return count_tokens(text) + TOKENS_PER_MESSAGE_OVERHEAD

    remaining = token_budget - cost(system_prompt) - cost(message)
    if remaining < 0:
        raise MessageTooLong(f"system prompt + message need {token_budget - remaining} tokens; budget is {token_budget}")
    turns = [{"role": t["role"], "content": text_of(t["content"])} for t in history]
    recent = trim_history(system_prompt, [*turns, {"role": "user", "content": message}], max_exchanges)[1:-1]
    kept = []
    for turn in reversed(recent):
        remaining -= cost(turn["content"])
        if remaining < 0:
            break
        kept.append(turn)
    kept.reverse()
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return [{"role": "system", "content": system_prompt}, *kept, {"role": "user", "content": message}]
