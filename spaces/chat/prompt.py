"""Gradio chat history -> the message list llama.cpp expects."""

# Keeps prompt + reply inside the context window on a 2-vCPU Space; older turns are dropped.
MAX_HISTORY_MESSAGES = 8


def text_of(content) -> str:
    """Plain text of a message; Gradio 6 stores content as a list of typed parts."""
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content if part.get("type") == "text")


def build_messages(system_prompt: str, history: list[dict], message: str) -> list[dict]:
    """System prompt, the recent history (starting on a user turn, as chat templates require), then `message`."""
    recent = history[-MAX_HISTORY_MESSAGES:]
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]
    return [
        {"role": "system", "content": system_prompt},
        *({"role": turn["role"], "content": text_of(turn["content"])} for turn in recent),
        {"role": "user", "content": message},
    ]
