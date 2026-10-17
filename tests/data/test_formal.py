import random

from fym1.data.formal import PROMPT_TEMPLATES, pick_paragraph, to_formal_chat

LONG = "Antananarivo no renivohitr'i Madagasikara. " * 6  # ~260 chars


def test_pick_paragraph_skips_short_and_long_lines():
    text = "\n".join(["Tantara", "x" * 900, LONG.strip()])

    assert pick_paragraph(text, random.Random(0)) == LONG.strip()


def test_pick_paragraph_none_when_nothing_fits():
    assert pick_paragraph("Fohy\n\nFohy koa", random.Random(0)) is None


def test_to_formal_chat_is_single_turn_about_title():
    chat = to_formal_chat("Antananarivo", LONG.strip(), random.Random(0))

    assert [t["role"] for t in chat] == ["user", "assistant"]
    assert "Antananarivo" in chat[0]["content"]
    assert chat[0]["content"] in {t.format(title="Antananarivo") for t in PROMPT_TEMPLATES}
    assert chat[1]["content"] == LONG.strip()
