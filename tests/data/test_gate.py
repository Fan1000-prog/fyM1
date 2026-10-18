"""Strict automatic gate: synthetic text only."""

import pytest

from fym1.data.anonymize import FIRST_NAMES
from fym1.data.gate import GATE_REASONS, contact_left, has_long_number, strict_gate
from fym1.data.known_words import word_list


def record(rid, *turns):
    roles = ["user", "assistant"]
    return {"id": rid, "messages": [{"role": roles[i % 2], "content": t} for i, t in enumerate(turns)]}


@pytest.mark.parametrize(
    "text",
    [
        "034 12 345 67",
        "0341234567",
        "34 12 345 67",
        "034.12.345.67",
        "034-12-345-67",
        "034\u201312\u2013345\u201367",  # en dash
        "034\u00a012\u00a0345\u00a067",
        "034\u202f12\u202f345\u202f67",
        "ref 12/34/567",
        "nomery 1234567 io",
        "034 12 345 67 Ar",  # not a price shape, the currency does not excuse it
        "15/03/2026",  # 8 digits: dates with a full year count too
    ],
)
def test_long_digit_runs_are_caught(text):
    assert has_long_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "1 500 000 Ar",
        "1 500 000 ariary",
        "2500000Ar",
        "1.500.000 Ariary",
        "1\u202f500\u202f000 fmg",
        "3 000 000 FMG",
        "1 250 000,50 €",
        "1 250 000 euros",
        "1250000 eur",
        "123456 olona",  # only 6 digits
        "14h30 sy 12/03",
        "<TEL>",
    ],
)
def test_prices_and_short_numbers_pass(text):
    assert not has_long_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "alefaso amin'ny olona@example.mg",
        "jereo https://example.org",
        "jereo example.org/sary?id=3",
        "www.example.org",
        "@olona_iray io",
        "slot \ue0000\ue001 tsy nosoloina",  # a private-use mark left by a failed replacement
    ],
)
def test_contacts_and_failed_placeholders_are_caught(text):
    assert contact_left(text)


@pytest.mark.parametrize("text", ["<EMAIL> <URL> <HANDLE> <ADDRESS>", "example.org", "mandeha @ 3 ora"])
def test_masked_or_bare_text_is_not_a_contact(text):
    assert not contact_left(text)


TINY_LEXICON = {
    "mg_common": word_list({"eny"}),
    "mg_wiki": word_list({"zazakely"}),
    "known_names": word_list({"tana"}),
    "fr": word_list({"bonjour", "tana"}),  # "Tana" counts once, under the first list that knows it
    "en": word_list({"hello"}),
}


def texts(result):
    return [turn["content"] for r in result.kept for turn in r["messages"]]


def test_gate_drops_whole_conversations_for_numbers_and_contacts_only():
    records = [
        record("keep", "salama", "eny tsara, 1 500 000 Ar ny vidiny"),
        record("phone", "salama", "antsoy aho 034 12 345 67"),
        record("name", "aiza i Bema?", "tsy fantatro ilay olona fa tsy hitako any an-tanàna"),
        record("both", "Bema 034 12 345 67", "eny"),
        record("mail", "ity: olona@example.mg", "misaotra"),
    ]

    result = strict_gate(records, fake_names=["Soa Rabe"])

    assert [r["id"] for r in result.kept] == ["keep", "name"]
    assert {k: v for k, v in result.counts.items() if not k.startswith("cleared_")} == {
        "conversations_before_gate": 5,
        "dropped": 3,
        "dropped_long_number": 2,
        "dropped_contact": 1,
        "dropped_too_many_unknown": 1,  # "both": 1 name in 2 words, counted under every reason
        "kept": 2,
        "tokens_replaced": 1,
    }
    assert set(GATE_REASONS) == {"long_number", "contact", "too_many_unknown"}


def test_unknown_capitalised_word_becomes_a_pool_fake_the_same_one_in_a_conversation():
    records = [record("a", "aiza i Bema sy Zandry?", "tsy hitako i Bema fa tonga i Zandry omaly hariva")]

    result = strict_gate(records, fake_names=[], lexicon=TINY_LEXICON)

    first, second = texts(result)
    assert "Bema" not in first + second and "Zandry" not in first + second
    bema, zandry = first.split()[2], first.split()[4].rstrip("?")
    assert bema in FIRST_NAMES and zandry in FIRST_NAMES and bema != zandry
    assert second.split()[3] == bema and second.split()[7] == zandry
    assert result.counts["tokens_replaced"] == 4


def test_replacement_fake_never_matches_a_word_already_in_the_conversation():
    # every pool first name but one is already in the text, so the free one must be used
    taken = " ".join(FIRST_NAMES[1:])
    records = [record("a", f"{taken} ary Bema", " ".join(["eny"] * 200))]

    result = strict_gate(records, fake_names=FIRST_NAMES, lexicon=TINY_LEXICON)

    assert texts(result)[0].endswith(f"ary {FIRST_NAMES[0]}")


def test_known_words_are_cleared_and_counted_by_the_first_list_that_knows_them():
    records = [record("a", "Eny, Zazakely tonga any Tana", "Bonjour! Hello. Eny")]

    result = strict_gate(records, fake_names=[], lexicon=TINY_LEXICON)

    assert texts(result) == ["Eny, Zazakely tonga any Tana", "Bonjour! Hello. Eny"]
    assert {k: v for k, v in result.counts.items() if k.startswith("cleared_")} == {
        "cleared_mg_common": 2,
        "cleared_mg_wiki": 1,
        "cleared_known_names": 1,
        "cleared_fr": 1,
        "cleared_en": 1,
        "cleared_seen_lowercase": 0,
    }
    assert result.counts["tokens_replaced"] == 0


@pytest.mark.parametrize(
    "word", ["Bonjour", "Bonjouuur", "Salut", "Mais", "Tana", "Jesoa", "Andriamanitra", "Mvola", "Hello", "Mariée"]
)
def test_real_lists_clear_everyday_french_english_and_listed_names(word):
    records = [record("a", f"{word} ary eny tsara", "eny tsara ihany koa")]

    result = strict_gate(records, fake_names=[])

    assert texts(result)[0].startswith(word)


@pytest.mark.parametrize("name", ["Marie", "Jean", "David", "John", "Pierre", "Paul"])
def test_real_lists_do_not_clear_common_first_names(name):
    # wordfreq and mg_common_words.txt ignore case; names Wikipedia capitalises are kept out
    records = [record("a", f"tonga i {name} omaly", "eny tsara ihany koa")]

    result = strict_gate(records, fake_names=[])

    assert name not in texts(result)[0]
    assert result.counts["tokens_replaced"] == 1


def test_conversation_with_more_than_30_percent_replaced_words_is_dropped():
    # 3 of 9 words replaced (33%): dropped; 3 of 10 (30%): kept
    records = [
        record("over", "Bema Zandry Koto", "eny eny eny eny eny eny"),
        record("at", "Bema Zandry Koto", "eny eny eny eny eny eny eny"),
    ]

    result = strict_gate(records, fake_names=[], lexicon=TINY_LEXICON)

    assert [r["id"] for r in result.kept] == ["at"]
    assert result.counts["dropped_too_many_unknown"] == 1
    assert result.counts["tokens_replaced"] == 3  # counted in kept conversations only


def test_gate_keeps_fake_names_placeholders_currency_and_common_words():
    records = [record("ok", "Eny ry Soa, <TEL> ve?", "Eny Rabe, Mialy sy Olona tonga, 1 500 Ar")]

    result = strict_gate(records, fake_names=["Soa Rabe"])

    assert texts(result) == ["Eny ry Soa, <TEL> ve?", "Eny Rabe, Mialy sy Olona tonga, 1 500 Ar"]
    assert result.counts["tokens_replaced"] == 0


def test_sentence_initial_word_seen_in_lower_case_is_cleared():
    # "Vazivazy" is in no list, but it is only capitalised at a sentence start and is
    # also typed in lower case, so it is an ordinary word, not a name.
    records = [record("a", "Vazivazy io. Vazivazy koa!", "ilay vazivazy ve? eny eny eny")]

    result = strict_gate(records, fake_names=[], lexicon=TINY_LEXICON)

    assert texts(result)[0] == "Vazivazy io. Vazivazy koa!"
    assert result.counts["cleared_seen_lowercase"] == 2


def test_word_capitalised_mid_sentence_is_replaced_even_if_seen_in_lower_case():
    records = [record("a", "hoy i Vazivazy", "ilay vazivazy ve? eny eny eny")]

    result = strict_gate(records, fake_names=[], lexicon=TINY_LEXICON)

    assert "Vazivazy" not in texts(result)[0]
    assert result.counts["tokens_replaced"] == 1


def test_gate_counts_never_hold_text():
    records = [record("a", "Bema 034 12 345 67", "eny"), record("b", "hoy i Koto", "eny eny eny eny")]

    counts = strict_gate(records, fake_names=[]).counts

    assert all(isinstance(v, int) for v in counts.values())
    assert "Bema" not in repr(counts) and "Koto" not in repr(counts)
