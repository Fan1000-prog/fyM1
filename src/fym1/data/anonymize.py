"""Replace personal details in chat text with placeholders or fake Malagasy names."""

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from importlib.resources import files

FIRST_NAMES = (
    "Soa", "Hasina", "Tiana", "Mamy", "Lova", "Fara", "Haja", "Rindra", "Mialy", "Toky",
    "Kanto", "Onja", "Faly", "Sitraka", "Tahina", "Andry", "Hanitra", "Volana", "Miora", "Fitia",
    "Aina", "Fenitra", "Herizo", "Lalaina", "Mahery", "Nomena", "Ravo", "Santatra", "Tsiry", "Vola",
    "Zoary", "Holy", "Ando", "Bodo", "Dera", "Elia", "Finaritra", "Hoby", "Iony", "Koloina",
)  # fmt: skip
SURNAMES = (
    "Rabe", "Ravelo", "Rasoa", "Rabemananjara", "Andrianina", "Razafy", "Rajaonarison",
    "Ramanantsoa", "Randrianarisoa", "Rasolofo", "Ratsimba", "Rafidy", "Rahaga", "Ralay",
    "Ranaivo", "Raveloson", "Rakotondrabe", "Razanakoto", "Rabarison", "Rajoelina",
)  # fmt: skip
# Only used if every pool name clashes with a real contact.
FALLBACK_NAME = "Olona"

# Words found in the most Malagasy Wikipedia articles, most widespread first
# (scripts/build_common_words.py). A name part never prefix-matches one of these.
COMMON_WORDS = tuple(files("fym1.data").joinpath("mg_common_words.txt").read_text("utf-8").split())
# Short words a short name part ("Ny", "Fa") must never replace: the most widespread
# Wikipedia words plus texting words Wikipedia lacks.
PROTECTED_SHORT_WORDS = set(COMMON_WORDS[:300]) | {"ve", "eh", "e", "ah", "ahn", "aho", "ry", "o", "ok"}

# A name part matches a longer or shorter word when they share a prefix this long.
MIN_NAME_PREFIX = 4

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
URL = re.compile(
    r"\bhttps?://\S+|\bwww\.\S+"
    r"|\b(?:[\w-]+\.)+(?:com|mg|fr|net|org|me|ly|be|io|co|gl)\b(?:/\S*)?",
    re.IGNORECASE,
)
_SEP = r"[\s.-]?"
PHONE = re.compile(
    r"(?<![\w+(])(?:"
    rf"(?:\(\s*(?:\+|00)261\s*\)|(?:\+|00)?261){_SEP}(?:\(0\){_SEP})?\d(?:{_SEP}\d){{8}}"  # Madagascar, intl
    rf"|(?:\+|00)\d(?:{_SEP}\d){{7,14}}"  # other international
    rf"|0(?:{_SEP}\d){{9}}"  # national, 10 digits however grouped: 034 12 345 67, 020 22 345 67
    r"|\d{9,15}"  # any other long run of digits
    r")(?!\w)"
)
# Malagasy plot addresses: "Lot II M 85 Ter", "Lot IVG 12 Bis". Kept only if it holds a number.
_LOT_PART = r"(?:[IVX]+[A-Z]*|\d+[A-Za-z]*|[A-Z]{1,3}|[Bb]is|[Tt]er|[Qq]uater)\b"
LOT_ADDRESS = re.compile(rf"\b[Ll][Oo][Tt]\.?\s+{_LOT_PART}(?:[\s-]+{_LOT_PART})*")
HANDLE = re.compile(r"(?<![\w.])@(\w[\w.]*\w|\w)")
WORD = re.compile(r"\w+")


def fold(s: str) -> str:
    """Lower-case and strip accents, so "Hélène" and "helene" compare equal."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).casefold()


def _shares_prefix(a: str, b: str) -> bool:
    """`a` and `b` are folded words, one a prefix of the other of at least MIN_NAME_PREFIX letters."""
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= MIN_NAME_PREFIX and longer.startswith(shorter)


def _fake_name_for(handle: str) -> str:
    digest = hashlib.sha256(fold(handle).encode("utf-8")).digest()
    return FIRST_NAMES[int.from_bytes(digest[:4], "big") % len(FIRST_NAMES)]


def _lot_address(match: re.Match) -> str:
    return "<ADDRESS>" if any(c.isdigit() for c in match.group()) else match.group()


class Anonymizer:
    """Anonymise many texts with one set of names; counts common words it had to replace."""

    def __init__(self, names: dict[str, str]):
        self._full_names = []  # (folded words, fake) for multi-word names, longest first
        self._exact = {}  # folded name part -> fake part
        for real in sorted(names):
            real_parts, fake_parts = WORD.findall(real), names[real].split()
            if len(real_parts) > 1:
                self._full_names.append((tuple(fold(p) for p in real_parts), names[real]))
            for i, part in enumerate(real_parts):
                # first name -> fake first name, any other part -> fake surname
                self._exact.setdefault(fold(part), fake_parts[min(i, 1, len(fake_parts) - 1)])
        self._full_names.sort(key=lambda item: len(item[0]), reverse=True)
        self._prefixable = [(part, fake) for part, fake in self._exact.items() if len(part) >= MIN_NAME_PREFIX]
        self._fake_words = {fold(w) for fake in names.values() for w in fake.split()}
        self._common = set(COMMON_WORDS)
        self.common_word_replacements = 0

    def __call__(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = EMAIL.sub("<EMAIL>", text)
        text = URL.sub("<URL>", text)
        text = PHONE.sub("<TEL>", text)
        text = LOT_ADDRESS.sub(_lot_address, text)
        text = self._replace_names(text)
        # an @mention of a participant is already their fake name by now; just drop the "@"
        return HANDLE.sub(lambda m: m.group(1) if fold(m.group(1)) in self._fake_words else _fake_name_for(m.group(1)), text)

    def _replace_names(self, text: str) -> str:
        words = list(WORD.finditer(text))
        pieces, done_until, i = [], 0, 0
        while i < len(words):
            length, fake = self._full_name_at(text, words, i)
            if not fake:
                length, fake = 1, self._fake_for_word(words[i].group())
            if fake:
                pieces += [text[done_until : words[i].start()], fake]
                done_until = words[i + length - 1].end()
            i += length
        return "".join(pieces) + text[done_until:]

    def _full_name_at(self, text: str, words: list[re.Match], i: int) -> tuple[int, str | None]:
        for parts, fake in self._full_names:
            window = words[i : i + len(parts)]
            if (
                len(window) == len(parts)
                and all(fold(w.group()) == p for w, p in zip(window, parts))
                and all(text[a.end() : b.start()].isspace() for a, b in zip(window, window[1:]))
            ):
                return len(parts), fake
        return 1, None

    def _fake_for_word(self, word: str) -> str | None:
        folded = fold(word)
        if folded in self._fake_words:
            return None
        if folded in self._exact:
            if len(folded) < MIN_NAME_PREFIX and folded in PROTECTED_SHORT_WORDS:
                return None
            self.common_word_replacements += folded in self._common
            return self._exact[folded]
        if len(folded) < MIN_NAME_PREFIX or folded in self._common:
            return None
        return next((fake for part, fake in self._prefixable if _shares_prefix(folded, part)), None)


def anonymize(text: str, names: dict[str, str]) -> str:
    """Replace emails, URLs, phone numbers, plot addresses, @handles and the given names in `text`.

    `names` maps each real participant name to its fake name ("First Last"). The full
    name, or any single part of it, becomes the fake (first name -> fake first name,
    other parts -> fake surname). Matching ignores case and accents. A word that shares
    a 4+ letter prefix with a name part ("nirin", "Nirinaaa") is replaced too, unless it
    is a common Malagasy word. Other @handles become a fake name derived from the
    handle, so the same handle always reads the same.
    """
    return Anonymizer(names)(text)


def assign_fake_names(real_names: Iterable[str]) -> dict[str, str]:
    """Give each real name a distinct "First Last" fake that shares no word with any real name.

    Stable: the same set of names always gets the same mapping, whatever the order.
    """
    real = sorted(set(real_names))
    real_words = {fold(w) for name in real for w in WORD.findall(name)}

    def is_free(fake: str) -> bool:
        return not any(fold(fake) == w or _shares_prefix(fold(fake), w) for w in real_words)

    firsts = [f for f in FIRST_NAMES if is_free(f)]
    lasts = [s for s in SURNAMES if is_free(s)]
    # surname-major order, so the first len(firsts) people all get different first names
    pool = [f"{first} {last}" for last in lasts for first in firsts]
    if not pool:
        return {name: f"{FALLBACK_NAME} {i + 1}" for i, name in enumerate(real)}
    fakes = {}
    for i, name in enumerate(real):
        round_number, index = divmod(i, len(pool))
        fakes[name] = pool[index] + (str(round_number + 1) if round_number else "")
    return fakes
