"""Real dates are found in the whole text first; a phone match never starts inside one.

All inputs are synthetic. The corpora below are generated, seeded, and contain no real chat text.
"""

import random
import re
from itertools import pairwise

import pytest

from fym1.data.anonymize import Anonymizer

ANON = Anonymizer({})


@pytest.mark.parametrize(
    "text",
    [
        # date + price
        "15/03/2026 1 500 Ar",
        "Ny 15/03/2026 1 500 Ar no nividianako azy",
        "05/09/2026 2 500 000 Ar",
        "15-03-2026 1 500 Ar",
        "15.03.2026 1 500 Ar",
        "15/03/26 1 500 Ar",
        "15/03 1 500 Ar",
        # date + time
        "le 15/03/2026 14h30",
        "15/03/2026 14 30",
        "15/03/2026 14:30",
        "01/02/2026 09 15 no fotoana",
        # date ranges
        "15/03/2026-20/03/2026",
        "15/03/2026 - 20/03/2026",
        "15/03/2026–20/03/2026",
        "15/03-20/03/2026",
        "du 15/03/2026 au 20/03/2026",
        "15-20/03/2026",
        # two dates in a row
        "15/03/2026 20/03/2026",
        "01/02/2026 03/04/2026 05/06/2026",
        "01/02/2026 03/04",
        # yyyy-mm-dd next to numbers
        "2026-03-15 1 500 Ar",
    ],
)
def test_date_next_to_numbers_is_kept(text):
    assert ANON(text) == text


@pytest.mark.parametrize("month", [f"{m:02d}" for m in range(1, 10)])
@pytest.mark.parametrize("tail", ["1 500 Ar", "14 30", "2 500 000 Ar", "20/{m}/2026"])
def test_months_01_to_09_are_kept(month, tail):
    text = f"15/{month}/2026 {tail.format(m=month)}"
    assert ANON(text) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("15/03/2026 034 12 345 67", "15/03/2026 <TEL>"),
        ("15/03/2026 0341234567", "15/03/2026 <TEL>"),
        ("15/03/2026 +261 34 12 345 67", "15/03/2026 <TEL>"),
        ("15/03/2026 00261 34 12 345 67", "15/03/2026 <TEL>"),
        ("15/03/2026 261 34 12 345 67", "15/03/2026 <TEL>"),
        ("15/03/2026 (034) 12 345 67", "15/03/2026 <TEL>"),
        ("le 15/03/2026 14h30, antsoy 033 11 222 33", "le 15/03/2026 14h30, antsoy <TEL>"),
        ("034 12 345 67 ny 15/03/2026", "<TEL> ny 15/03/2026"),
    ],
)
def test_phone_right_next_to_a_date_is_masked(text, expected):
    assert ANON(text) == expected


@pytest.mark.parametrize("text", ["034 11/12 345", "+261 34 01/02 345", "034 01/02/26 7"])
def test_strong_phone_that_spans_a_date_shape_still_wins(text):
    assert ANON(text) == "<TEL>"


# --- generated corpora --------------------------------------------------------------------

WORDS = ["Ny", "amin'ny", "dia", "ho avy", "aho", "isika", "rahampitso", "ok", "mety", "le", "du", "au", "tamin'ny"]
UNITS = ["Ar", "kg", "olona", "litre", "km", "andro"]


def _date(rng: random.Random) -> str:
    d, m, y = rng.randint(1, 31), rng.randint(1, 12), rng.randint(1990, 2030)
    sep = rng.choice("/-.")
    dd = f"{d:02d}" if rng.random() < 0.8 else str(d)
    mm = f"{m:02d}" if rng.random() < 0.8 else str(m)
    shape = rng.randrange(5)
    if shape == 0:
        return f"{dd}{sep}{mm}{sep}{y}"
    if shape == 1:
        return f"{dd}{sep}{mm}{sep}{y % 100:02d}"
    if shape == 2:
        return f"{dd}/{mm}"
    if shape == 3:
        return f"{y}-{m:02d}-{d:02d}"
    return f"{_date_full(rng)}{rng.choice(['-', ' - ', '–'])}{_date_full(rng)}"  # range


def _date_full(rng: random.Random) -> str:
    return f"{rng.randint(1, 31):02d}/{rng.randint(1, 12):02d}/{rng.randint(2000, 2030)}"


def _price(rng: random.Random) -> str:
    amount = rng.choice([500, 1_500, 2_500, 15_000, 35_000, 150_000, 1_500_000, 25_000_000])
    grouped = f"{amount:,}".replace(",", rng.choice([" ", ".", ""]))
    return rng.choice([f"{grouped} Ar", f"Ar {grouped}", f"{grouped} Ariary", f"{grouped}Ar"])


def _time(rng: random.Random) -> str:
    h, m = rng.randint(0, 23), rng.randint(0, 59)
    return rng.choice([f"{h}h{m:02d}", f"{h:02d}h", f"{h:02d}:{m:02d}", f"{h:02d} {m:02d}", f"{h}h"])


def _quantity(rng: random.Random) -> str:
    return f"{rng.randint(1, 999)} {rng.choice(UNITS)}"


def everyday_line(rng: random.Random) -> str:
    """A line with a date and a few prices, times or quantities, never a phone number.

    Numbers sit right next to a date; two non-date numbers get a word between them, since
    "08 18 150 000" is a 10-digit number whatever it meant (a pre-existing, date-free limit).
    """
    makers = [_price, _time, _quantity, _date]
    parts = [(True, _date(rng))]
    for _ in range(rng.randint(1, 3)):
        maker = rng.choice(makers)
        parts.append((maker is _date, maker(rng)))
    rng.shuffle(parts)
    line = parts[0][1]
    for (was_date, _), (is_date, text) in pairwise(parts):
        line += " " if was_date or is_date else f" {rng.choice(WORDS)} "
        line += text
    return f"{rng.choice(WORDS)} {line}" if rng.random() < 0.5 else line


def phone_number(rng: random.Random) -> str:
    """A Malagasy or international number in a form the anonymiser promises to catch."""
    prefix = rng.choice(["32", "33", "34", "37", "38", "39", "20"])
    a, b, c = f"{rng.randint(0, 99):02d}", f"{rng.randint(0, 999):03d}", f"{rng.randint(0, 99):02d}"
    sep = rng.choice([" ", "", ".", "-"])
    local = sep.join([prefix, a, b, c])
    return rng.choice(
        [
            f"0{local}",
            f"(0{prefix}) {a} {b} {c}",
            f"+261 {local}",
            f"+261{prefix}{a}{b}{c}",
            f"00261 {local}",
            f"261 {local}",
            f"+261 (0){prefix} {a} {b} {c}",
            f"+33 6 {a} {c} {a} {c}",
        ]
    )


def test_everyday_corpus_has_no_phone():
    rng = random.Random(20261017)
    lines = [everyday_line(rng) for _ in range(5_000)]

    wrong = [line for line in lines if "<TEL>" in ANON(line)]

    assert wrong == [], f"{len(wrong)} of {len(lines)} everyday lines got a <TEL>, e.g. {wrong[:5]}"


def test_phone_corpus_is_fully_masked():
    rng = random.Random(17102026)
    cases = []
    for _ in range(2_000):
        phone = phone_number(rng)
        before = rng.choice(["Antsoy", "laharana:", _date_full(rng), f"ny {_date_full(rng)}", "le"])
        after = rng.choice(
            ["rahampitso", "ok", f"ny {_date_full(rng)}", f"amin'ny {_time(rng)}", f"sady {_price(rng)}"]
        )
        cases.append((f"{before} {phone} {after}", f"{before} <TEL> {after}"))

    wrong = [(text, ANON(text)) for text, expected in cases if ANON(text) != expected]

    assert wrong == [], f"{len(wrong)} of {len(cases)} phones not masked cleanly, e.g. {wrong[:5]}"


def test_corpora_are_not_trivial():
    rng = random.Random(1)
    lines = [everyday_line(rng) for _ in range(200)]
    assert sum(bool(re.search(r"\d{1,2}/\d{1,2}/\d{4}", line)) for line in lines) > 50
