import pytest
from prompt import TOKENS_PER_MESSAGE_OVERHEAD, MessageTooLong, build_messages


def turn(role, text):
    # Gradio 6 hands history content over as a list of typed parts
    return {"role": role, "content": [{"type": "text", "text": text}]}


def count_words(text):
    return len(text.split())


def build(history, message, budget=1000, system="sys"):
    return build_messages(system, history, message, count_words, budget)


def test_system_prompt_history_then_new_message():
    history = [turn("user", "Sali ahn"), turn("assistant", "Salama!")]
    assert build(history, "De ahoana?") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Sali ahn"},
        {"role": "assistant", "content": "Salama!"},
        {"role": "user", "content": "De ahoana?"},
    ]


def test_plain_string_content_is_kept():
    history = [{"role": "user", "content": "Sali ahn"}, {"role": "assistant", "content": "Salama!"}]
    assert build(history, "x")[1:3] == [
        {"role": "user", "content": "Sali ahn"},
        {"role": "assistant", "content": "Salama!"},
    ]


def test_oldest_turns_are_dropped_to_fit_the_budget_and_history_starts_on_a_user_turn():
    history = [turn("user" if i % 2 == 0 else "assistant", f"turn {i} " + "w " * 10) for i in range(10)]
    per_turn = 12 + TOKENS_PER_MESSAGE_OVERHEAD
    fixed = 2 * (1 + TOKENS_PER_MESSAGE_OVERHEAD)  # system "sys" + message "new"
    kept = build(history, "new", budget=fixed + 3 * per_turn)[1:-1]
    # the 3 newest turns fit, but the oldest of them is an assistant turn, so it goes too
    assert [m["content"].split()[1] for m in kept] == ["8", "9"]
    assert kept[0]["role"] == "user"


def test_one_long_old_message_is_dropped_not_sent():
    history = [turn("user", "w " * 5000), turn("assistant", "ok"), turn("user", "hi"), turn("assistant", "salama")]
    kept = build(history, "new", budget=200)[1:-1]
    assert [m["content"] for m in kept] == ["hi", "salama"]


def test_new_message_too_long_for_the_budget_is_refused():
    with pytest.raises(MessageTooLong):
        build([], "w " * 500, budget=100)
