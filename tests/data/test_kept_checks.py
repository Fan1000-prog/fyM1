import pytest

from fym1.data.kept_checks import KEPT_COUNT_KEYS, kept_counts, leak_counts, message_counts


def record(*turns):
    return {"id": "x", "messages": [{"role": role, "content": text} for role, text in turns]}


CLEAN = [
    record(("user", "salama, inona no vaovao?"), ("assistant", "tsy misy, 15 000 Ar ilay izy")),
    record(("user", "ok tsara"), ("assistant", "<TEL> sy <EMAIL> sy <URL> 😂"), ("user", "veloma")),
]


def test_message_counts_per_role():
    assert message_counts(CLEAN) == {"kept_messages": 5, "kept_messages_user": 3, "kept_messages_assistant": 2}


def test_unknown_role_is_an_error():
    with pytest.raises(ValueError, match="system"):
        message_counts([record(("system", "x"))])


def test_clean_chats_have_no_leaks():
    assert set(leak_counts(CLEAN).values()) == {0}


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("antsoy 0341234567", "leak_digit_runs"),
        ("vola 1500000 Ar", "leak_digit_runs"),  # no price excuse, unlike the gate
        ("antsoy 034 12 345 67", "leak_spaced_digit_runs"),
        ("antsoy 034.12.345.67", "leak_spaced_digit_runs"),
        ("alefaso any amin'ny rabe.sample@example.mg", "leak_emails"),
        ("alefaso amin'ny rabe @ example.com", "leak_emails"),
        ("ry @someone", "leak_at_signs"),
        ("ry ＠someone", "leak_at_signs"),
        ("jereo https://example.org/x", "leak_links"),
        ("jereo www.example", "leak_links"),
        ("jereo example.com", "leak_links"),
        ("jereo m.me/someone", "leak_links"),
    ],
)
def test_each_leak_is_counted(text, key):
    counts = leak_counts([record(("user", text))])

    assert counts[key] >= 1


def test_leaks_are_counted_per_match_across_records():
    records = [record(("user", "0341234567 sy 0321234567")), record(("assistant", "@a @b"))]

    counts = leak_counts(records)

    assert counts["leak_digit_runs"] == 2 and counts["leak_at_signs"] == 2


@pytest.mark.parametrize("text", ["12 34 56", "tamin'ny 1 jolay 08:05", "15 000 Ar", "ok.tsara be", "fr: c'est ça"])
def test_short_numbers_and_plain_words_are_not_leaks(text):
    assert set(leak_counts([record(("user", text))]).values()) == {0}


def test_kept_counts_has_exactly_the_stats_keys_and_ints():
    counts = kept_counts(CLEAN)

    assert tuple(counts) == KEPT_COUNT_KEYS
    assert all(isinstance(v, int) for v in counts.values())
