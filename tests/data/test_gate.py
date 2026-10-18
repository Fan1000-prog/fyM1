"""Strict automatic gate: synthetic text only."""

import pytest

from fym1.data.gate import GATE_REASONS, contact_left, has_long_number, strict_gate


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


def test_gate_drops_whole_conversations_and_counts_each_reason():
    records = [
        record("keep", "salama", "eny tsara, 1 500 000 Ar ny vidiny"),
        record("phone", "salama", "antsoy aho 034 12 345 67"),
        record("name", "aiza i Bema?", "tsy fantatro"),
        record("both", "Bema 034 12 345 67", "eny"),
        record("mail", "ity: olona@example.mg", "misaotra"),
    ]

    result = strict_gate(records, fake_names=["Soa Rabe"])

    assert [r["id"] for r in result.kept] == ["keep"]
    assert result.counts == {
        "conversations_before_gate": 5,
        "dropped": 4,
        "dropped_long_number": 2,
        "dropped_leftover_name": 2,
        "dropped_contact": 1,
        "kept": 1,
    }
    assert set(GATE_REASONS) == {"long_number", "leftover_name", "contact"}


def test_gate_keeps_fake_names_placeholders_and_common_words():
    records = [record("ok", "Eny ry Soa, <TEL> ve?", "Eny Rabe, Mialy sy Olona tonga")]

    assert strict_gate(records, fake_names=["Soa Rabe"]).counts["kept"] == 1


def test_sentence_initial_word_seen_in_lower_case_is_noise():
    # "Zazakely" is not a common word, but it is only capitalised at a sentence start and is
    # also typed in lower case, so it is an ordinary word, not a name.
    records = [record("a", "Zazakely tonga. Zazakely koa!", "ilay zazakely ve?")]

    assert strict_gate(records, fake_names=[]).counts["kept"] == 1


def test_sentence_initial_name_never_typed_in_lower_case_is_dropped():
    records = [record("a", "Bema, aiza ianao?", "ato")]

    assert strict_gate(records, fake_names=[]).counts["dropped_leftover_name"] == 1


def test_word_capitalised_mid_sentence_is_dropped_even_if_seen_in_lower_case():
    records = [record("a", "hoy i Zazakely", "ilay zazakely ve?")]

    assert strict_gate(records, fake_names=[]).counts["dropped_leftover_name"] == 1


def test_gate_counts_never_hold_text():
    records = [record("a", "Bema 034 12 345 67", "eny")]

    counts = strict_gate(records, fake_names=[]).counts

    assert all(isinstance(v, int) for v in counts.values())
    assert "Bema" not in repr(counts)
