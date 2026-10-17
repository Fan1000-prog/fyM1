"""Training text that matches what the model sees at inference.

Some chat templates (Gemma 4, Qwen 3 with thinking off) put an empty thinking stub after the assistant
header of the generation prompt but leave it out of earlier assistant turns. Training on the plain
render would teach a turn format the model never sees when it answers. `training_text` gives every
assistant turn the inference header. Templates are passed as `render(messages, add_generation_prompt)`.
"""

import re
from collections.abc import Callable

Render = Callable[..., str]

# One set of chat-template options for training, eval generation and the trim gates (03_trim uses the
# same value). Templates without a thinking switch ignore it.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
_Q, _A, _Q2 = "Q7xq", "A7xq", "Q8xq"


def _headers(render: Render) -> tuple[str, str, str]:
    """(user header, assistant header in history, assistant header of the generation prompt)."""
    user = {"role": "user", "content": _Q}
    base, gen = render([user]), render([user], add_generation_prompt=True)
    if not gen.startswith(base):
        raise ValueError("generation prompt does not extend the rendered conversation")
    history = render([user, {"role": "assistant", "content": _A}, {"role": "user", "content": _Q2}])
    end_of_turn = history[history.index(_Q2) + len(_Q2):]
    after_user = history.index(_Q) + len(_Q) + len(end_of_turn)
    assistant = history[after_user:history.index(_A)]
    between = history[history.index(_A) + len(_A):history.index(_Q2)]
    if not between.startswith(end_of_turn):
        raise ValueError("cannot find the end-of-turn marker")
    return between[len(end_of_turn):], assistant, gen[len(base):]


def turn_markers(render: Render) -> tuple[str, str]:
    """(instruction_part, response_part) for Unsloth `train_on_responses_only`."""
    user, _, response = _headers(render)
    return user, response


def training_text(render: Render, messages: list[dict]) -> str:
    """Rendered conversation where every assistant turn starts with the generation-prompt header."""
    _, assistant, response = _headers(render)
    text = render(messages)
    if response == assistant:
        return text
    if not response.startswith(assistant):
        raise ValueError(f"generation header {response!r} does not extend history header {assistant!r}")
    stub = response[len(assistant):]
    return re.sub(re.escape(assistant) + f"(?!{re.escape(stub)})", lambda _: response, text)
