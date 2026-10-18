"""Strict gate fails closed (security review, round 4): synthetic text only.

One block per finding. Over-replacing is fine; a name or number getting through is not.
"""

import importlib.util
import re
from pathlib import Path

import pytest

from fym1.data import known_words
from fym1.data.anonymize import Anonymizer
from fym1.data.gate import has_long_number, pool_names_in, strict_gate
from fym1.data.known_words import KNOWN_WORDS, known_by, word_list
from fym1.data.messenger import Message, Thread
from fym1.data.pipeline import build_records

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_known_words.py"
_spec = importlib.util.spec_from_file_location("build_known_words", SCRIPT)
build_known_words = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_known_words)

FILLER = "eny tsara ihany koa, eny tsara ihany koa, eny tsara ihany koa"


def record(rid, *turns):
    roles = ["user", "assistant"]
    return {"id": rid, "messages": [{"role": roles[i % 2], "content": t} for i, t in enumerate(turns)]}


def gated_text(text, fake_names=(), **kwargs):
    result = strict_gate([record("a", text, FILLER)], fake_names=list(fake_names), **kwargs)
    assert result.kept, "conversation was dropped"
    return result.kept[0]["messages"][0]["content"]


def words(text):
    return set(re.findall(r"[^\W\d_]+", text))


# 1. Allowlist: only fakes actually assigned pass.


def test_f1_unassigned_pool_name_is_replaced():
    assert "Mialy" not in words(gated_text("nandeha Mialy omaly", fake_names=["Soa Rabe"]))


def test_f1_unassigned_pool_surname_and_fallback_are_replaced():
    out = words(gated_text("nandeha Ravelo sy Olona omaly", fake_names=["Soa Rabe"]))
    assert not {"Ravelo", "Olona"} & out


def test_f1_assigned_fakes_pass():
    assert gated_text("nandeha Soa sy Rabe omaly", fake_names=["Soa Rabe"]) == "nandeha Soa sy Rabe omaly"


def test_f1_anonymizer_reports_every_mention_fake_it_wrote():
    anonymize = Anonymizer({"Nirina Rakoto": "Soa Rabe"})

    out = anonymize("hoy @Tsiky Bema sy @Nirina")

    fake_words = {"Soa", "Rabe"} | {w for fake in anonymize.mention_fakes for w in fake.split()}
    assert set(out.split()) - {"hoy", "sy"} <= fake_words
    assert anonymize.mention_fakes - {"Soa", "Soa Rabe"}  # the stranger's fake is recorded


def test_f1_pipeline_keeps_mention_fakes_and_replaces_other_pool_names():
    thread = Thread(
        title="t",
        participants=["Nirina Rakoto", "Tojo Randria"],
        messages=[
            Message(sender="Nirina Rakoto", text="hoy @Tsiky Bema omaly", ts_ms=0),
            Message(sender="Tojo Randria", text="eny tsara ihany koa, eny tsara ihany koa", ts_ms=60_000),
        ],
    )
    result = build_records([thread], owner="Tojo Randria", is_malagasy=lambda _: True)

    assert result.records[0]["messages"][0]["content"].startswith("hoy ")
    assert result.gate["tokens_replaced"] == 0  # the mention's fake is a fake we wrote, so it passes


@pytest.mark.parametrize("raw", ["Rabe2 no tonga", "RABE_ no tonga", "tonga2Rabe"])
def test_f1_raw_pool_names_are_found_inside_digits_and_underscores(raw):
    assert "rabe" in pool_names_in([raw])


# 2. Number check: all whitespace, ( ) +, and no phone hiding as a price.


@pytest.mark.parametrize(
    "text",
    [
        "034\t12\t345\t67",
        "034 12 345 67",  # figure space
        "034 12 345 67",  # thin space
        "034 12 345 67",  # hair space
        "034\u200b12\u200b345\u200b67",  # zero-width space
        "(034) 12 345 67",
        "034 12 (345) 67",
        "+261 34 12 345 67",
        "034_12_345_67",
        "0341234567 Ar",  # plain digits, but a phone prefix
        "034 123 456 Ar",
        "341234567 Ar",  # 3[2-9] + 7 digits
        "261341234567 ariary",
        "00261341234567 Ar",
        "+261 341 234 567 Ar",
        "+33 612 345 678 €",
        "(032) 123 4567 Ar",
        "1\t500\t000 Ar",  # not a clean price shape
    ],
)
def test_f2_phone_shapes_are_long_numbers(text):
    assert has_long_number(text)


@pytest.mark.parametrize(
    "text", ["1 500 000 Ar", "2500000Ar", "12 000 000 ariary", "1.500.000 Ariary", "1 250 000,50 €"]
)
def test_f2_clean_prices_still_pass(text):
    assert not has_long_number(text)


# 3. No corpus-wide whitelist.


def test_f3_lower_case_elsewhere_in_the_corpus_never_clears_a_capitalised_word():
    records = [record("a", "Zakotoka io.", FILLER), record("b", "ilay zakotoka", FILLER)]

    result = strict_gate(records, fake_names=[])

    assert "Zakotoka" not in result.kept[0]["messages"][0]["content"]


def test_f3_sentence_initial_word_no_list_knows_is_replaced():
    tiny = {"mg_common": word_list({"eny", "tsara", "ihany", "koa", "io"})}

    assert "Vazivazy" not in gated_text("Vazivazy io. ilay vazivazy", lexicon=tiny)


# 4. Tokenizer: digits and underscores split words; only exact placeholders are skipped.


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("tonga Rakoto2 omaly", "Rakoto"),
        ("tonga 2Rakoto omaly", "Rakoto"),
        ("tonga Rakoto_ omaly", "Rakoto"),
        ("hoy jean_r omaly", "jean"),
        ("hoy <Rakoto> omaly", "Rakoto"),
        ("hoy <TEL>Rakoto omaly", "Rakoto"),
        ("salutRakoto ve", "Rakoto"),
    ],
)
def test_f4_names_glued_to_digits_underscores_or_brackets_are_checked(text, name):
    assert name.lower() not in gated_text(text).lower()


def test_f4_exact_placeholders_stay():
    text = "<TEL> <EMAIL> <URL> <HANDLE> <ADDRESS> eny"
    assert gated_text(text) == text


# 5. Word lists hold no first names; the case floor is low.


@pytest.mark.parametrize("name", ["soa", "vola", "mamy", "tiana", "fara", "hery", "rabe", "ravelo", "lalao", "noro"])
def test_f5_committed_lists_hold_no_pool_or_deny_list_names(name):
    for list_name in ("mg_wiki", "fr", "en"):
        assert name not in KNOWN_WORDS[list_name].folded, (list_name, name)


def test_f5_build_subtracts_pool_and_deny_list_names():
    cases = build_known_words.count_cases(["ny vola sy ny soa sy ny trano"] * 5)

    names = build_known_words.name_list(["sandrine"])
    words_ = build_known_words.lower_case_words(cases, min_count=3, names=names)

    assert {"vola", "soa", "rabe", "hery", "sandrine"} <= names
    assert "trano" in words_ and "vola" not in words_ and "soa" not in words_


def test_f5_name_capitalised_a_few_times_and_never_lower_looks_like_a_name():
    cases = build_known_words.count_cases(["hoy i Bema"] * 3)

    assert build_known_words.looks_like_a_name("bema", cases)


# 6. Fuzzy match only for real stretching, never into a first name.


def test_f6_double_letters_are_not_stretching():
    assert not word_list({"ane"}).has("Anne")


def test_f6_real_stretching_still_matches():
    assert word_list({"bonjour"}).has("Bonjouuur")


def test_f6_stretched_first_name_never_matches_a_word():
    assert not word_list({"mariee"}).has("Mariiiee")  # its key is "marie"
    assert known_by("Mariiie", {"fr": word_list({"marie"})}) is None


# 7. The lexicon clears sentence starts only; markers and deny list.


@pytest.mark.parametrize("word", ["Fiara", "Merci", "Trano", "Hello"])
def test_f7_lexicon_word_capitalised_mid_sentence_is_replaced(word):
    assert word not in words(gated_text(f"tonga ny {word} omaly"))


@pytest.mark.parametrize("word", ["Bonjour", "Trano", "Hello"])
def test_f7_lexicon_word_at_a_sentence_start_is_cleared(word):
    assert gated_text(f"eny. {word} tsara") == f"eny. {word} tsara"


@pytest.mark.parametrize("text", ["tonga i Neny omaly", "any Tana izy", "ny Zoma hariva", "amin'i Dada"])
def test_f7_hand_list_clears_mid_sentence_and_after_markers(text):
    assert gated_text(text) == text


@pytest.mark.parametrize(
    "name", ["Hery", "Soa", "Vola", "Mamy", "Lalao", "Fetra", "Rivo", "Tiana", "Fara", "Noro", "Voahangy"]
)
def test_f7_deny_list_names_are_replaced_mid_sentence(name):
    assert name not in words(gated_text(f"nandeha {name} omaly"))


@pytest.mark.parametrize("name", ["Hery", "Lalao", "Fetra", "Voahangy"])
def test_f7_deny_list_names_are_replaced_at_a_sentence_start(name):
    assert name not in words(gated_text(f"{name} no nandeha"))


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("ry hery, aiza?", "hery"),
        ("tonga i lalao", "lalao"),
        ("tamin'i fetra", "fetra"),
        ("amin'i noro", "noro"),
        ("hoy @ rivo", "rivo"),
        ("Ra-hery tonga", "hery"),
    ],
)
def test_f7_deny_list_names_after_markers_are_replaced(text, name):
    assert name not in {w.lower() for w in words(gated_text(text))}


@pytest.mark.parametrize("text", ["hoy @ Merci", "amin'i Trano"])
def test_f7_capitalised_word_after_a_marker_is_replaced(text):
    assert not {"Merci", "Trano"} & words(gated_text(text))


def test_f7_deny_list_has_the_reviewed_names():
    reviewed = {"hery", "soa", "vola", "mamy", "lalao", "fetra", "rivo", "tiana", "fara", "noro", "voahangy"}
    assert reviewed <= known_words.NAME_DENY_LIST


@pytest.mark.parametrize("text", ["mila vola aho", "soa ve", "ry namana"])
def test_f7_deny_list_words_in_lower_case_without_a_marker_stay(text):
    assert gated_text(text) == text


# Round-3 review list.


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("salut mialy, tonga ve ianao?", "mialy"),
        ("aiza ianao njaka?", "njaka"),
        ("veloma nirina", "nirina"),
        ("miarahaba anao voahangy", "voahangy"),
        ("hoy fanja sy lalaina", "fanja"),
        ("hoy fanja sy lalaina", "lalaina"),
        ("tonga tsiry sy koloina sy herizo", "koloina"),
    ],
)
def test_r3_lowercase_malagasy_first_names_are_replaced(text, name):
    assert name not in words(gated_text(text))


def test_r3_raw_pool_name_in_lower_case_is_disallowed():
    assert "mialy" in pool_names_in(["salut mialy"])


def test_r3_lowercase_name_in_one_conversation_never_clears_it_in_another():
    records = [record("a", "Njaka tonga.", FILLER), record("b", "aiza njaka ?", FILLER)]

    result = strict_gate(records, fake_names=[])

    out = " ".join(turn["content"] for r in result.kept for turn in r["messages"]).lower()
    assert "njaka" not in out


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Rakoto123 no username-ny", "rakoto"),
        ("Sandrine2 tonga", "sandrine"),
        ("ny insta-ko dia sandrine_rk", "sandrine"),
        ("Jean_Pierre tonga", "jean"),
        ("Jean_Pierre tonga", "pierre"),
        ("Rak0to tonga", "rak0to"),
        ("hoy n1r1na", "n1r1na"),
        ("Rakoto>>> tonga", "rakoto"),
        ("<<Njaka tonga", "njaka"),
        ("lol<3Njaka", "njaka"),
    ],
)
def test_r3_names_glued_to_digits_underscores_or_angle_brackets_are_replaced(text, name):
    assert name not in gated_text(text).lower()


@pytest.mark.parametrize(
    "text",
    ["antsoy 034;12;345;67", "antsoy 034*12*345*67", "antsoy 034|12|345|67", "antsoy 034:12:345:67", "034 (12) 345 67"],
)
def test_r3_phones_split_by_any_punctuation_are_long_numbers(text):
    assert has_long_number(text)
