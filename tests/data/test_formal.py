import random

from fym1.data.formal import (
    MAX_TEMPLATE_REPEATS,
    PROMPT_TEMPLATES,
    Candidate,
    drop_templated,
    is_stub,
    pick_paragraph,
    to_formal_chat,
)

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


def test_is_stub_below_1500_chars():
    assert is_stub("x" * 1499)
    assert not is_stub("x" * 1500)


def commune(i):
    title = f"Tanana{i}"
    return Candidate(id=str(i), title=title, paragraph=f"{title} dia kaominina ao Frantsa, misy {i} mponina.")


def test_drop_templated_removes_openings_repeated_too_often():
    stubs = [commune(i) for i in range(MAX_TEMPLATE_REPEATS + 1)]
    unique = Candidate(id="u", title="Antananarivo", paragraph=LONG.strip())

    assert drop_templated(stubs + [unique]) == [unique]


def test_drop_templated_keeps_openings_up_to_the_limit():
    stubs = [commune(i) for i in range(MAX_TEMPLATE_REPEATS)]

    assert drop_templated(stubs) == stubs
