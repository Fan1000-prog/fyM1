"""Phones typed with unicode dashes, or with a letter O for the 0 glued to a word.

All inputs are synthetic. The corpora below are generated, seeded, and contain no real chat text.
"""

import random
import unicodedata

import pytest

from fym1.data.anonymize import Anonymizer

ANON = Anonymizer({})
UNICODE_DASHES = "‐‑‒–—―−"  # ‐ ‑ ‒ – — ― −


def nfkc(text: str) -> str:
    """What the anonymiser writes for `text` it leaves alone: NFKC turns no-break spaces and "‑" plain."""
    return unicodedata.normalize("NFKC", text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("034–12–345–67", "<TEL>"),
        ("tél: 034‑12‑345‑67", "tél: <TEL>"),
        ("0327—35—17—59", "<TEL>"),
        ("15/03/2026 02‒03‒74‒01‒62", "15/03/2026 <TEL>"),
        ("antsoy 033―12―345―67 ok", "antsoy <TEL> ok"),
        ("+261−34−12−345−67", "<TEL>"),
        ("034 12 345 67", "<TEL>"),  # no-break spaces
        ("034 12 345 67", "<TEL>"),  # narrow no-break spaces
        ("laharana034–12–345–67", "laharana<TEL>"),
    ],
)
def test_phone_with_unicode_dashes_or_spaces_is_masked(text, expected):
    assert ANON(text) == expected


@pytest.mark.parametrize("dash", list(UNICODE_DASHES))
def test_date_with_unicode_dashes_is_kept(dash):
    for text in [f"15{dash}03{dash}2026 1 500 Ar", f"2026{dash}10{dash}17 14h30", f"le 15/03{dash}20/03/2026 ok"]:
        assert ANON(text) == nfkc(text)


@pytest.mark.parametrize("dash", list(UNICODE_DASHES))
def test_phone_after_a_date_with_unicode_dashes(dash):
    text = f"15{dash}03{dash}2026 034{dash}12{dash}345{dash}67"
    assert ANON(text) == nfkc(f"15{dash}03{dash}2026 <TEL>")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("numéroo32.78.833.34", "numéro<TEL>"),
        ("WhatsAppO328585520", "WhatsApp<TEL>"),
        ("contactO345 40 77 66", "contact<TEL>"),
        ("contactO345 40 77 66", "contact<TEL>"),
        ("telO34 12 345 67", "tel<TEL>"),
        ("appelleo33/78/800/37!", "appelle<TEL>!"),
        ("laharanaO20 22 123 45", "laharana<TEL>"),
        ("numO3 74 78 29 68?", "num<TEL>?"),
        ("WhatsAppo2/04/61/98/10", "WhatsApp<TEL>"),
    ],
)
def test_letter_o_glued_to_a_word_is_masked(text, expected):
    assert ANON(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Hono 3 kilao",
        "photo 2",
        "Toamasina 25",
        "manao320 000 000 Ar",  # an amount glued to a word ending in o: 3-digit groups
        "manao300 000 000 Ar",
        "photo2026 12 34",
        "kilao32 500 Ar",
        "toko3 200 000 Ar",
        "manao2 000 000 000 Ar",
        "radio320 500 250 Ar",
        "vidiny 2\u202f000\u202f000\u202f000 Ar",  # narrow no-break spaces in an amount
        "vidiny 1\u00a0000\u00a0000\u00a0000 Ar",
    ],
)
def test_everyday_words_ending_in_o_before_numbers_are_kept(text):
    assert ANON(text) == nfkc(text)


O_WORDS = ["Hono", "kilao", "photo", "manao", "toko", "radio", "vao", "loto", "moto", "Toamasina", "numero", "Lalao"]
UNITS = ["Ar", "kg", "olona", "litre", "km", "andro", "isa", "taona"]


def _number(rng: random.Random) -> str:
    amount = rng.choice([2, 3, 25, 32, 34, 150, 300, 320, 1_500, 35_000, 320_000, 3_400_000, 32_000_000, 340_000_000, 320_500_250, 2_000_000_000])
    # a 9+ digit run with no gaps is masked on purpose (a reference number leans private)
    seps = [" ", ".", " "] + ([""] if amount < 1_000_000 else [])
    return f"{amount:,}".replace(",", rng.choice(seps))


def _date(rng: random.Random) -> str:
    dash = rng.choice(UNICODE_DASHES + "-/")
    return f"{rng.randint(1, 31):02d}{dash}{rng.randint(1, 12):02d}{dash}{rng.randint(1990, 2030)}"


def o_word_line(rng: random.Random) -> str:
    """A line where a word ending in o / O sits right before a number, glued or spaced, never a phone."""
    word = rng.choice(O_WORDS)
    word = word.upper() if rng.random() < 0.2 else word
    gap = rng.choice(["", "", " "])
    tail = rng.choice([f" {rng.choice(UNITS)}", "", f" {_date(rng)}", f" {rng.randint(0, 23)}h{rng.randint(0, 59):02d}"])
    head = rng.choice(["", "ny ", "dia ", f"{_date(rng)} "])
    return f"{head}{word}{gap}{_number(rng)}{tail}"


def test_everyday_o_word_corpus_has_no_phone():
    rng = random.Random(20261018)
    lines = [o_word_line(rng) for _ in range(20_000)]

    wrong = [line for line in lines if "<TEL>" in ANON(line)]

    assert wrong == [], f"{len(wrong)} of {len(lines)} everyday lines got a <TEL>, e.g. {wrong[:5]}"


def test_unicode_dash_phone_corpus_is_fully_masked():
    rng = random.Random(18102026)
    wrong = []
    for _ in range(2_000):
        digits = rng.choice(["32", "33", "34", "37", "38", "20"]) + "".join(str(rng.randint(0, 9)) for _ in range(7))
        dash = rng.choice(UNICODE_DASHES)
        phone = rng.choice(
            [
                dash.join(["0" + digits[:2], digits[2:4], digits[4:7], digits[7:]]),
                dash.join(["0" + digits[0], *(digits[i : i + 2] for i in range(1, 9, 2))]),
                dash.join(["0" + digits[:3], digits[3:5], digits[5:7], digits[7:]]),
            ]
        )
        before = rng.choice(["", "tél: ", "antsoy ", f"{_date(rng)} ", "14h30 ", "1 500 Ar "])
        if ANON(f"{before}{phone} ok") != nfkc(f"{before}<TEL> ok"):
            wrong.append(f"{before}{phone} ok")
    assert wrong == [], f"{len(wrong)} unicode-dash phones not masked, e.g. {wrong[:5]}"
