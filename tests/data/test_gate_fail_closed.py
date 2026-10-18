"""Strict gate fails closed (security review, round 4): synthetic text only.

One block per finding. Over-replacing is fine; a name or number getting through is not.
"""

import re

import pytest

from fym1.data import known_words
from fym1.data.anonymize import Anonymizer, normalise
from fym1.data.gate import contact_left, has_long_number, pool_names_in, strict_gate
from fym1.data.known_words import KNOWN_WORDS, known_by, word_list
from fym1.data.messenger import Message, Thread
from fym1.data.pipeline import build_records

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


def test_f5_build_subtracts_pool_and_deny_list_names(build_known_words):
    cases = build_known_words.count_cases(["ny vola sy ny soa sy ny trano"] * 5)

    names = build_known_words.name_list(["sandrine"])
    words_ = build_known_words.lower_case_words(cases, min_count=3, names=names)

    assert {"vola", "soa", "rabe", "hery", "sandrine"} <= names
    assert "trano" in words_ and "vola" not in words_ and "soa" not in words_


def test_f5_name_capitalised_a_few_times_and_never_lower_looks_like_a_name(build_known_words):
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
        ("Ra-hery tonga", "hery"),
    ],
)
def test_f7_deny_list_names_after_markers_are_replaced(text, name):
    assert name not in {w.lower() for w in words(gated_text(text))}


def test_f7_capitalised_word_after_a_marker_is_replaced():
    assert "Trano" not in words(gated_text("amin'i Trano"))


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


# Round-4 review list.


def pipeline_text(text):
    """`text` after the anonymiser and the strict gate, or None when its conversation is dropped."""
    thread = Thread(
        title="t",
        participants=["Nirina Rakoto", "Tojo Randria"],
        messages=[
            Message(sender="Nirina Rakoto", text=text, ts_ms=0),
            Message(sender="Tojo Randria", text=FILLER, ts_ms=60_000),
        ],
    )
    result = build_records([thread], owner="Tojo Randria", is_malagasy=lambda _: True)
    return result.records[0]["messages"][0]["content"] if result.records else None


@pytest.mark.parametrize(
    "text",
    [
        "njakarabe@ gmail.com",
        "mailako njakarabe @ gmail . com",
        "njakab\uff20gmail.com",  # full-width @
        "\uff20njakarabe tonga",
        "njakarabe at gmail.com",
        "mailako: njakab at gmail dot com",
        "mailako: njakab arobase gmail point com",
        "mailako: njakab AT yahoo.fr",
    ],
)
def test_r4_email_and_handle_variants_never_leave_the_local_part(text):
    out = pipeline_text(text)
    assert out is None or "njaka" not in out.lower()


@pytest.mark.parametrize(
    "text",
    ["njaka@ <URL>", "hoy @ rivo", "hoy @ Merci", "@ <EMAIL>", "njaka at <URL>", "njaka arobase gmail", "x (at) hotmail"],
)
def test_r4_spaced_at_and_spelled_at_are_contacts(text):
    assert contact_left(text)


@pytest.mark.parametrize("text", ["tonga at 3", "<EMAIL> no nalefako", "look at that", "arobase"])
def test_r4_at_without_a_mail_host_is_not_a_contact(text):
    assert not contact_left(text)


def test_r4_full_width_at_is_a_contact_in_the_gate():
    result = strict_gate([record("a", "njakab\uff20<URL>", FILLER)], fake_names=[])
    assert result.counts["dropped_contact"] == 1


def test_r4_circled_letters_are_folded_and_checked():
    assert "rakoto" not in gated_text("salut \u24c7\u24d0\u24da\u24de\u24e3\u24de").lower()


@pytest.mark.parametrize(
    "name",
    ["tsanta", "miangaly", "tsilavina", "domoina", "nambinina", "dera", "riana", "sandrinerakoto", "kevinandria"],
)
def test_r4_lowercase_word_no_list_knows_is_replaced(name):
    assert name not in words(gated_text(f"salut {name}, {FILLER}"))


@pytest.mark.parametrize("text", ["ny fb-ko: sandrinerakoto", "ny insta-ko dia njakarabe", "ny fb-ko: kevinandria"])
def test_r4_lowercase_username_without_at_is_replaced(text):
    out = pipeline_text(f"{text}, {FILLER}")
    assert out is None or not {"sandrinerakoto", "njakarabe", "kevinandria"} & words(out)


@pytest.mark.parametrize(
    "text",
    [
        "tsara be ny trano",
        "je suis là ce soir",
        "thank you my friend",
        "lol ok mdr haha",
        "mila vola aho",
        "soa ve",
        "une rose",
        "je t'aime",
    ],
)
def test_r4_ordinary_and_texting_words_stay(text):
    assert gated_text(text) == text


def test_r4_stretched_known_word_stays():
    assert gated_text("tsaraaa be") == "tsaraaa be"


# --- round 5: a name broken into one-letter pieces ---------------------------------------

STRIKE, UNDERLINE = "\u0336", "\u0332"


def marked(word, mark):
    return "".join(c + mark for c in word)


@pytest.mark.parametrize(
    "text",
    [
        f"salut {marked('Rakoto', STRIKE)}",
        f"salut {marked('rakoto', STRIKE)}",
        f"salut {marked('Rakoto', UNDERLINE)}",
    ],
)
def test_r5_combining_marks_left_after_nfkc_are_stripped(text):
    out = gated_text(f"{text}, {FILLER}")
    assert STRIKE not in out and UNDERLINE not in out
    assert "rakoto" not in out.lower()


@pytest.mark.parametrize(
    "text",
    [
        f"salut {marked('Rakoto', STRIKE)}",
        "salut r a k o t o",
        "salut R.a.k.o.t.o",
        "salut R-a-k-o-t-o",
        "salut N j a k a R a b e",
    ],
)
def test_r5_name_through_the_anonymiser_and_gate_is_replaced(text):
    out = pipeline_text(f"{text}, {FILLER}")
    assert out is None or not re.search(r"r\W{0,2}a\W{0,2}k\W{0,2}o\W{0,2}t\W{0,2}o|n\W*j\W*a\W*k\W*a", out.lower())


@pytest.mark.parametrize(
    "text, gone",
    [
        ("salut r a k o t o", "r a k o t o"),
        ("salut R.a.k.o.t.o", "R.a.k.o.t.o"),
        ("salut R-a-k-o-t-o", "R-a-k-o-t-o"),
        ("salut N j a k a R a b e", "N j a k a R a b e"),
        ("salut R. a. k. o. t. o.", "R. a. k. o. t. o"),
    ],
)
def test_r5_single_letters_in_a_run_are_one_word(text, gone):
    out = gated_text(f"{text}, {FILLER}")
    assert gone not in out
    assert len(out.split(",")[0].split()) == 2  # "salut <one fake name>"


@pytest.mark.parametrize("text", ["salut t s a r a", "t.s.a.r.a be"])
def test_r5_spelled_out_known_word_is_kept_as_written(text):
    assert gated_text(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "il y a du monde",
        "ça a été dur",
        "a ny trano e",
        "e, tonga aho",
        "J.P. tonga",
        "à demain",
        "o a",
    ],
)
def test_r5_ordinary_single_letter_words_stay(text):
    assert gated_text(text) == text


@pytest.mark.parametrize("text", ["côté", "voilà", "été", "niño", "Ô ry namana", "à bientôt", "Ñ", "ankehitriny"])
def test_r5_normalise_keeps_accented_letters(text):
    assert normalise(text) == text


@pytest.mark.parametrize("text", ["voilà", "à demain", "tsara ô"])
def test_r5_accented_known_words_stay_through_the_gate(text):
    assert gated_text(text) == text


def test_r5_decomposed_accents_compose_and_survive():
    assert normalise("vite\u0301 o\u0302 a\u0300 n\u0303") == "vit\u00e9 \u00f4 \u00e0 \u00f1"


def test_r5_emoji_variation_selectors_and_keycaps_stay():
    text = "tsara \u2764\ufe0f be 1\ufe0f\u20e3"
    assert normalise(text) == text


def test_r5_anonymiser_drops_stray_marks_too():
    out = Anonymizer({"Nirina Rakoto": "Soa Rabe"})(f"salut {marked('Rakoto', STRIKE)}")
    assert STRIKE not in out and "rakoto" not in out.lower()


def test_r5_zero_width_space_inside_a_name_is_stripped():
    out = gated_text(f"salut R\u200bakoto, {FILLER}")
    assert "\u200b" not in out and "akoto" not in out.lower()


R6_SPELLED = [
    "R_a_k_o_t_o",
    "R,a,k,o,t,o",
    "J,E,A,N",
    "j.e.a.n",
    "R*a*k*o*t*o",
    "R/a/k/o/t/o",
    "r:a:k:o:t:o",
    "R'a'k'o't'o",
    "R|a|k|o|t|o",
    "R+a+k+o+t+o",
    "R~a~k~o~t~o",
    "R\u00b7a\u00b7k\u00b7o\u00b7t\u00b7o",
    "R\U0001f525a\U0001f525k\U0001f525o\U0001f525t\U0001f525o",
    "R . a . k . o . t . o",
    "R   a   k   o   t   o",
    # format characters: ZWJ, ZWNJ, soft hyphen, invisible times, Mongolian vowel separator, tag
    *(marked("Rakoto", c) for c in ("\u200d", "\u200c", "\u00ad", "\u2062", "\u180e", "\U000e0041")),
    # enclosing marks ("fancy text") and variation selectors after each letter
    *(marked("Rakoto", c) for c in ("\u20dd", "\u20de", "\u20e3", "\u0489", "\ufe0e", "\ufe0f")),
    "R\u0903a\u0903k\u0903o\u0903t\u0903o\u0903",
]


def _name_left(out):
    letters = "".join(re.findall(r"[^\W\d_]+", out.removesuffix(f", {FILLER}"))).lower()
    return "rakoto" in letters or "jean" in letters


@pytest.mark.parametrize("name", R6_SPELLED)
def test_r6_single_letters_split_by_any_separator_are_replaced(name):
    gated = gated_text(f"salut {name}, {FILLER}")
    assert not _name_left(gated), repr(gated)
    through = pipeline_text(f"salut {name}, {FILLER}")
    assert through is None or not _name_left(through), repr(through)


@pytest.mark.parametrize("c", ["\u200d", "\u200c", "\u00ad", "\u2062", "\u180e", "\u20dd", "\u20e3", "\u0489", "\ufe0e"])
def test_r6_normalise_drops_format_and_enclosing_marks_after_a_letter(c):
    assert normalise(marked("Rakoto", c)) == "Rakoto"


@pytest.mark.parametrize(
    "text",
    [
        "tsara \u2764\ufe0f be",  # variation selector after an emoji
        "1\ufe0f\u20e3 #\ufe0f\u20e3 *\u20e3",  # keycaps
        "\U0001f468\u200d\U0001f469\u200d\U0001f467",  # ZWJ family
        "\U0001f44d\U0001f3fd",  # skin tone
        "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",  # tag flag
        "\U0001f1f2\U0001f1ec",
        "côté à ñ",
    ],
)
def test_r6_normalise_keeps_emoji_sequences(text):
    assert normalise(text) == text
