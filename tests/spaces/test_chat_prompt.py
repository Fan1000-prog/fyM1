from prompt import MAX_HISTORY_MESSAGES, build_messages


def turn(role, text):
    # Gradio 6 hands history content over as a list of typed parts
    return {"role": role, "content": [{"type": "text", "text": text}]}


def test_system_prompt_history_then_new_message():
    history = [turn("user", "Sali ahn"), turn("assistant", "Salama!")]
    assert build_messages("sys", history, "De ahoana?") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Sali ahn"},
        {"role": "assistant", "content": "Salama!"},
        {"role": "user", "content": "De ahoana?"},
    ]


def test_plain_string_content_is_kept():
    history = [{"role": "user", "content": "Sali ahn"}, {"role": "assistant", "content": "Salama!"}]
    assert build_messages("sys", history, "x")[1:3] == [
        {"role": "user", "content": "Sali ahn"},
        {"role": "assistant", "content": "Salama!"},
    ]


def test_long_history_is_trimmed_and_starts_with_a_user_turn():
    history = [turn("user" if i % 2 == 0 else "assistant", str(i)) for i in range(2 * MAX_HISTORY_MESSAGES + 1)]
    kept = build_messages("sys", history, "new")[1:-1]
    assert len(kept) <= MAX_HISTORY_MESSAGES
    assert kept[0]["role"] == "user"
    assert kept[-1]["content"] == history[-1]["content"][0]["text"]
