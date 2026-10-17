"""Replace personal details in chat text with placeholders or fake Malagasy names."""

import hashlib
import re
from collections.abc import Iterable

FAKE_NAMES = (
    "Soa", "Hasina", "Tiana", "Mamy", "Lova", "Fara", "Haja", "Rindra", "Mialy", "Toky",
    "Kanto", "Onja", "Faly", "Sitraka", "Tahina", "Andry", "Hanitra", "Volana", "Miora", "Fitia",
    "Aina", "Fenitra", "Herizo", "Lalaina", "Mahery", "Nomena", "Ravo", "Santatra", "Tsiry", "Vola",
    "Zoary", "Holy", "Ando", "Bodo", "Dera", "Elia", "Finaritra", "Hoby", "Iony", "Koloina",
)  # fmt: skip

# A name token matches a word when they share a prefix this long (catches "nirin", "Nirinaaa").
MIN_NAME_PREFIX = 4

URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
SEP = r"[\s.-]?"
PHONE = re.compile(
    r"(?<![\w+])(?:"
    rf"(?:\+|00)?261{SEP}\d{{2}}{SEP}\d{{2}}{SEP}\d{{3}}{SEP}\d{{2}}"  # Madagascar, international
    rf"|0{SEP}3\d{SEP}\d{{2}}{SEP}\d{{3}}{SEP}\d{{2}}"  # Madagascar mobile, 03x xx xxx xx
    r"|\+\d(?:[\s.-]?\d){7,14}"  # any other international number
    r"|\d{9,15}"  # any other long run of digits
    r")(?!\w)"
)
HANDLE = re.compile(r"(?<![\w.])@(\w[\w.]*\w|\w)")
WORD = re.compile(r"\w+")


def _shares_prefix(word: str, token: str) -> bool:
    shorter, longer = sorted((word.lower(), token.lower()), key=len)
    return len(shorter) >= MIN_NAME_PREFIX and longer.startswith(shorter)


def _fake_name_for(key: str) -> str:
    digest = hashlib.sha256(key.lower().encode("utf-8")).digest()
    return FAKE_NAMES[int.from_bytes(digest[:4], "big") % len(FAKE_NAMES)]


def _replace_names(text: str, names: dict[str, str]) -> str:
    for real in sorted(names, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(real)}(?!\w)", names[real], text, flags=re.IGNORECASE)
    fakes = set(names.values())
    tokens = [(token, fake) for real, fake in names.items() for token in WORD.findall(real)]

    def replace_word(match: re.Match) -> str:
        word = match.group()
        if word in fakes:
            return word
        return next((fake for token, fake in tokens if _shares_prefix(word, token)), word)

    return WORD.sub(replace_word, text)


def anonymize(text: str, names: dict[str, str]) -> str:
    """Replace URLs, emails, phone numbers, @handles and the given names in `text`.

    `names` maps each real participant name to its fake name. Full names and any
    single word of a name (case-insensitive, sharing a prefix of at least 4 letters)
    become the fake name. Other @handles become a fake name derived from the handle,
    so the same handle always reads the same.
    """
    text = URL.sub("<URL>", text)
    text = EMAIL.sub("<EMAIL>", text)
    text = PHONE.sub("<TEL>", text)
    text = _replace_names(text, names)
    fakes = set(names.values())
    # an @mention of a participant is already their fake name by now; just drop the "@"
    return HANDLE.sub(lambda m: m.group(1) if m.group(1) in fakes else _fake_name_for(m.group(1)), text)


def assign_fake_names(real_names: Iterable[str]) -> dict[str, str]:
    """Give each real name a distinct fake name that shares no word with any real name.

    Stable: the same set of names always gets the same mapping, whatever the order.
    """
    real = sorted(set(real_names))
    real_words = {w.lower() for name in real for w in WORD.findall(name)}
    pool = [f for f in FAKE_NAMES if not any(_shares_prefix(f, w) or f.lower() == w for w in real_words)]
    fakes = {}
    for i, name in enumerate(real):
        round_number, index = divmod(i, len(pool))
        fakes[name] = pool[index] + (str(round_number + 1) if round_number else "")
    return fakes
