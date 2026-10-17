from fym1.train.chat_format import training_text, turn_markers


def qwen_like(messages, add_generation_prompt=False):
    """Mimics Qwen3 with thinking off: assistant turns after the last user turn (and the generation prompt) get the stub."""
    out = ""
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    for i, m in enumerate(messages):
        stub = "<think></think>" if m["role"] == "assistant" and i > last_user else ""
        out += f"<{m['role']}>{stub}{m['content']}</s>"
    if add_generation_prompt:
        out += "<assistant><think></think>"
    return out


def plain(messages, add_generation_prompt=False):
    """Mimics Gemma 3: no thinking stub anywhere."""
    out = "<bos>" + "".join(f"<{m['role']}>{m['content']}</s>" for m in messages)
    return out + ("<assistant>" if add_generation_prompt else "")


CONV = [
    {"role": "user", "content": "U1"},
    {"role": "assistant", "content": "A1"},
    {"role": "user", "content": "U2"},
    {"role": "assistant", "content": "A2"},
]


def test_turn_markers_take_the_generation_prompt_as_response_marker():
    assert turn_markers(qwen_like) == ("<user>", "<assistant><think></think>")
    assert turn_markers(plain) == ("<user>", "<assistant>")


def test_training_text_gives_every_assistant_turn_the_inference_header():
    text = training_text(qwen_like, CONV)
    assert text == "<user>U1</s><assistant><think></think>A1</s><user>U2</s><assistant><think></think>A2</s>"


def test_training_text_leaves_templates_without_a_stub_unchanged():
    assert training_text(plain, CONV) == plain(CONV)
