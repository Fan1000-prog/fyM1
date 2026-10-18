"""Strict gate hardening (security pass on round 4, re-checked on main): synthetic text only.

One block per item. Over-replacing or over-dropping is fine; a name or number getting through is not.
"""

import pytest

from fym1.data.anonymize import Anonymizer, assign_fake_names, name_keys, normalise
from fym1.data.gate import has_hidden_char, has_long_number, strict_gate
from fym1.data.messenger import Message, Thread
from fym1.data.pipeline import build_records

FILLER = "eny tsara ihany koa, eny tsara ihany koa, eny tsara ihany koa"


def record(rid, *turns):
    roles = ["user", "assistant"]
    return {"id": rid, "messages": [{"role": roles[i % 2], "content": t} for i, t in enumerate(turns)]}


# 1. Phone shapes: any 9-10 digit run is a phone unless it is clearly a price.


@pytest.mark.parametrize(
    "text",
    [
        "202 212 345 Ar",  # Malagasy landline 2x + 7 digits
        "20 22 123 45 Ar",
        "612 345 678 €",  # French mobile without its 0
        "712345678 euros",
        "123 456 789 Ar",  # 9 digits, not ending 000, no cents
        "1 234 567 891 Ar",  # 10 digits
    ],
)
def test_h1_nine_or_ten_digit_runs_are_phones_unless_price_shaped(text):
    assert has_long_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "150 000 000 Ar",  # grouped thousands ending 000
        "1 500 000 000 Ariary",
        "150.000.000 fmg",
        "123 456 789,00 €",  # cents
        "1 500 000 Ar",  # 7 digits: unchanged
        "25 000 000 Ar",  # 8 digits: unchanged
    ],
)
def test_h1_price_shaped_nine_or_ten_digit_runs_still_pass(text):
    assert not has_long_number(text)


@pytest.mark.parametrize("text", ["150000000 Ar", "150 000 000"])
def test_h1_a_price_still_needs_groups_and_a_currency(text):
    assert has_long_number(text)


# 2. A fake written in one chat never whitelists that word as a real name in another.


def _thread(text):
    return Thread(
        title="t",
        participants=["Nirina Rakoto", "Tojo Randria"],
        messages=[Message("Nirina Rakoto", text, 0), Message("Tojo Randria", FILLER, 60_000)],
    )


def test_h2_mention_fake_from_one_chat_is_replaced_as_a_real_name_in_another():
    mention_chat = _thread("hoy @Tsiky Bema omaly")
    probe = Anonymizer({"Nirina Rakoto": "x", "Tojo Randria": "y"})
    probe("hoy @Tsiky Bema omaly")
    fake_first = next(iter(probe.mention_fakes)).split()[0]

    result = build_records(
        [mention_chat, _thread(f"nandeha {fake_first} omaly")], owner="Tojo Randria", is_malagasy=lambda _: True
    )

    other = [r["messages"][0]["content"] for r in result.records if r["messages"][0]["content"].startswith("nandeha")]
    assert other and fake_first not in other[0].split()


# 3. Real names become match keys through the same normalise() as the text.


@pytest.mark.parametrize("hidden", ["\u200b", "\u00ad", "\u2060", "\ufeff"])
def test_h3_invisible_char_inside_a_real_name_still_matches_the_text(hidden):
    anonymize = Anonymizer({f"Ni{hidden}rina Rakoto": "Soa Rabe"})

    assert "Nirina" not in anonymize("Nirina tonga")
    assert name_keys(f"Ni{hidden}rina") == ["nirina"]


def test_h3_fake_pool_skips_a_real_name_hidden_by_an_invisible_char():
    fake = assign_fake_names(["So\u200ba Rakoto"])["So\u200ba Rakoto"]

    assert fake.split()[0] != "Soa"


# 4. Invisible or combining characters left after normalise() drop the conversation.


@pytest.mark.parametrize(
    "text",
    [
        "Nir\u0903ina tonga",  # spacing mark (Mc) splits a name
        "R\u0903a\u0903k\u0903o\u0903t\u0903o\u0903",  # moved from r6: now dropped, not replaced
        "Nir\u3164ina tonga",  # Hangul filler: a default-ignorable letter
        "Nir\uffa0ina tonga",
        "Nir\u115fina tonga",
        "a \u200dNirina",  # format char not after a letter
        "salut \u2066Nirina\u2069",  # bidi isolates
        "a \U0001d173Nirina",  # musical format char not after a letter
    ],
)
def test_h4_leftover_hidden_chars_are_found(text):
    assert has_hidden_char(normalise(text))


@pytest.mark.parametrize(
    "text",
    [
        "tsara \u2764\ufe0f",  # emoji + variation selector
        "\U0001f468\u200d\U0001f469\u200d\U0001f467 fianakaviana",  # ZWJ family
        "\U0001f3f3\ufe0f\u200d\U0001f308",  # ZWJ after a variation selector
        "laharana 1\ufe0f\u20e3 sy #\u20e3",  # keycaps
        "\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",  # subdivision flag tags
        "Hélène sy Zoë",  # real accents, NFC-composed
        "He\u0301le\u0300ne",  # decomposed accents: normalise() composes them
        "\U0001f44d\U0001f3fd",  # skin tone
    ],
)
def test_h4_emoji_sequences_and_accents_are_not_hidden_chars(text):
    assert not has_hidden_char(normalise(text))  # the gate checks normalised text


def test_h4_gate_drops_and_counts_a_conversation_with_a_hidden_char():
    result = strict_gate([record("a", "Nir\u0903ina tonga", FILLER), record("b", "tsara \u2764\ufe0f", FILLER)], [])

    assert [r["id"] for r in result.kept] == ["b"]
    assert result.counts["dropped_hidden_char"] == 1
