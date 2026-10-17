"""Replace personal details in chat text with placeholders or fake Malagasy names.

Known limit: a third party's name typed as plain text (no "@", not a participant or a
nickname) is not detected — that needs named-entity recognition. The owner's review
sheet is the backstop. See docs/anonymisation.md.
"""

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
# Display names Messenger shows for deleted accounts: not a person's name, never mapped.
DELETED_ACCOUNT_NAMES = {"facebook user", "utilisateur facebook"}

# A name part matches a longer or shorter word when they share a prefix this long.
MIN_NAME_PREFIX = 4
# Malagasy possessive / pronoun endings glued to names: "Soako" (my Soa), "Fyntsika".
NAME_SUFFIXES = ("ntsika", "tsika", "nao", "nay", "ko", "ny")

EMAIL = re.compile(
    r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"
    r"|\b[\w.+-]+@(?:gmail|yahoo|hotmail|outlook|live|ymail|icloud)\b"  # domain cut short
    r"|\b[\w.+-]+\s*(?:\[at\]|\(at\)|\s@\s)\s*[\w-]+(?:\s*(?:\.|\[dot\]|\(dot\))\s*[\w-]+)*"
    r"\s*(?:\.|\[dot\]|\(dot\))\s*(?:com|fr|mg|net|org|io|me|co)\b",  # obfuscated: x [at] y . com
    re.IGNORECASE,
)
URL = re.compile(
    r"\bhttps?://\S+|\bwww\.\S+"
    r"|\b(?:[\w-]+\.)+[a-z]{2,}/\S*"  # any dotted host with a path: linktr.ee/x
    r"|\b(?:[\w-]+\.)+(?:com|mg|fr|net|org|me|ly|be|io|co|gl)\b",
    re.IGNORECASE,
)
_SEP = r"\s*[./-]?\s*"
PHONE = re.compile(
    r"(?<![\w+(])(?:"
    rf"(?:\(\s*(?:\+|00)261\s*\)|(?:\+|00)?261){_SEP}(?:\(0\){_SEP})?\d(?:{_SEP}\d){{8}}"  # Madagascar, intl
    rf"|(?:\+|00)\d{{1,3}}(?:\s*\(\d{{1,4}}\))?(?:{_SEP}\d){{6,13}}"  # other intl: +33 6 …, +1 (555) …
    rf"|[0Oo](?:{_SEP}\d){{9}}"  # national, 10 digits however grouped; "O34" typed with a letter O
    rf"|3[2-9]{_SEP}\d\d{_SEP}\d{{3}}{_SEP}\d\d"  # mobile without the 0: 34 12 345 67
    r"|\d{9,15}"  # any other long run of digits
    r")(?!\w)"
)
# Malagasy plot addresses: "Lot II M 85 Ter", "Lot IVG 12 Bis". Kept only if it holds a number.
_LOT_PART = r"(?:[IVX]+[A-Z]*|\d+[A-Za-z]*|[A-Z]{1,3}|[Bb]is|[Tt]er|[Qq]uater)\b"
LOT_ADDRESS = re.compile(rf"\b[Ll][Oo][Tt]\.?\s+{_LOT_PART}(?:[\s-]+{_LOT_PART})*")
# "ig: nirina.rk", "insta nirina_rk", "snap: nirinark"
LABELLED_HANDLE = re.compile(
    r"\b(?P<label>ig|insta(?:gram)?|snap(?:chat)?|tiktok|tg|telegram|twitter|fb|facebook|viber|wa|whatsapp)"
    r"(?P<sep>\s*[:=]\s*@?|\s*@|\s+)(?P<handle>[\w.]*\w)",
    re.IGNORECASE,
)
# "@nirina.rk", or a display-name mention "@Haja Rabe" (up to 3 capitalised words)
MENTION = re.compile(r"(?<![\w.])@(\w[\w.]*\w|\w)((?:[ \t]+[A-ZÀ-ÖØ-Þ]\w*){0,2})")
WORD = re.compile(r"\w+")
_PLACEHOLDER_SLOT = "{}"  # private-use marks: never part of a \w word
_SLOT = re.compile("(\\d+)")


def fold(s: str) -> str:
    """Lower-case and strip accents, so "Hélène" and "helene" compare equal."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).casefold()


def compare_key(word: str) -> str:
    """How words are compared: folded, with stretched letters collapsed ("SOAAA" -> "soa")."""
    return re.sub(r"(.)\1+", r"\1", fold(word))


# Words found in the most Malagasy Wikipedia articles, most widespread first
# (scripts/build_common_words.py). A name part never prefix-matches one of these.
COMMON_WORDS = tuple(files("fym1.data").joinpath("mg_common_words.txt").read_text("utf-8").split())
_COMMON_KEYS = {compare_key(w) for w in COMMON_WORDS}
# Short words a short name part ("Ny", "Fa") must never replace: the most widespread
# Wikipedia words plus texting words Wikipedia lacks.
PROTECTED_SHORT_WORDS = {compare_key(w) for w in COMMON_WORDS[:300]} | {"ve", "eh", "e", "ah", "ahn", "aho", "ry", "o", "ok"}


def _shares_prefix(a: str, b: str) -> bool:
    """`a` and `b` are compare keys, one a prefix of the other of at least MIN_NAME_PREFIX letters."""
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= MIN_NAME_PREFIX and longer.startswith(shorter)


def _hashed_fake(text: str, words: int) -> str:
    """A fake name derived from `text`, so the same handle always reads the same."""
    number = int.from_bytes(hashlib.sha256(compare_key(text).encode("utf-8")).digest()[:4], "big")
    first = FIRST_NAMES[number % len(FIRST_NAMES)]
    return first if words == 1 else f"{first} {SURNAMES[number // len(FIRST_NAMES) % len(SURNAMES)]}"


def _lot_address(match: re.Match) -> str:
    return "<ADDRESS>" if any(c.isdigit() for c in match.group()) else match.group()


def _labelled_handle(match: re.Match) -> str:
    sep, handle = match["sep"], match["handle"]
    looks_like_handle = sep.strip() or re.search(r"[._\d]", handle)  # "snap ve?" is just a word
    return f"{match['label']}{sep.rstrip('@')}<HANDLE>" if looks_like_handle else match.group()


class Anonymizer:
    """Anonymise many texts with one set of names; counts common words it had to replace."""

    def __init__(self, names: dict[str, str]):
        self._full_names = {}  # compare keys of a multi-word name -> fake
        self._exact = {}  # compare key of a name part -> fake part
        for real in sorted(names):
            real_parts, fake_parts = WORD.findall(real), names[real].split()
            if len(real_parts) > 1:
                self._full_names[tuple(compare_key(p) for p in real_parts)] = names[real]
            for i, part in enumerate(real_parts):
                # first name -> fake first name, any other part -> fake surname
                self._exact.setdefault(compare_key(part), fake_parts[min(i, 1, len(fake_parts) - 1)])
        self._longest_name = max(map(len, self._full_names), default=1)
        self._prefixable = [(part, fake) for part, fake in self._exact.items() if len(part) >= MIN_NAME_PREFIX]
        self._fake_words = {compare_key(w) for fake in names.values() for w in fake.split()}
        self.common_word_replacements = 0

    def __call__(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = EMAIL.sub("<EMAIL>", text)
        text = URL.sub("<URL>", text)
        text = PHONE.sub("<TEL>", text)
        text = LOT_ADDRESS.sub(_lot_address, text)
        text = LABELLED_HANDLE.sub(_labelled_handle, text)
        mentions: list[str] = []
        text = MENTION.sub(lambda m: self._park(mentions, self._fake_for_mention(m)), text)
        text = self._replace_names(text)
        return _SLOT.sub(lambda m: mentions[int(m.group(1))], text)

    @staticmethod
    def _park(parked: list[str], fake: str) -> str:
        """Hold `fake` aside so the name pass below cannot touch it."""
        parked.append(fake)
        return _PLACEHOLDER_SLOT.format(len(parked) - 1)

    def _fake_for_mention(self, match: re.Match) -> str:
        words = [match.group(1), *match.group(2).split()]
        keys = tuple(compare_key(w) for w in words)
        if keys in self._full_names:
            return self._full_names[keys]
        if len(words) == 1 and (fake := self._fake_for_word(words[0])):
            return fake
        return _hashed_fake(" ".join(words), len(words))

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
        for length in range(min(self._longest_name, len(words) - i), 1, -1):
            window = words[i : i + length]
            keys = tuple(compare_key(w.group()) for w in window)
            if keys in self._full_names and all(
                text[a.end() : b.start()].isspace() for a, b in zip(window, window[1:])
            ):
                return length, self._full_names[keys]
        return 1, None

    def _fake_for_word(self, word: str) -> str | None:
        key = compare_key(word)
        if key in self._fake_words:
            return None
        if key in self._exact:
            return self._exact_fake(key)
        for suffix in NAME_SUFFIXES:
            stem = key[: -len(suffix)]
            if key.endswith(suffix) and stem in self._exact and (fake := self._exact_fake(stem)):
                return fake + suffix
        if len(key) < MIN_NAME_PREFIX or key in _COMMON_KEYS:
            return None
        return next((fake for part, fake in self._prefixable if _shares_prefix(key, part)), None)

    def _exact_fake(self, key: str) -> str | None:
        if len(key) < MIN_NAME_PREFIX and key in PROTECTED_SHORT_WORDS:
            return None
        self.common_word_replacements += key in _COMMON_KEYS
        return self._exact[key]


def anonymize(text: str, names: dict[str, str]) -> str:
    """Replace emails, URLs, phone numbers, plot addresses, handles and the given names in `text`.

    `names` maps each real participant name to its fake name ("First Last"). The full
    name, or any single part of it, becomes the fake (first name -> fake first name,
    other parts -> fake surname). Matching ignores case, accents and stretched letters,
    and also catches a part with a possessive ending ("Soako"). A word that shares a
    4+ letter prefix with a name part ("nirin") is replaced too, unless it is a common
    Malagasy word. Labelled handles ("ig: x") become <HANDLE>; other @handles and
    @mentions become a fake name derived from the handle.
    """
    return Anonymizer(names)(text)


def assign_fake_names(real_names: Iterable[str]) -> dict[str, str]:
    """Give each real name a distinct "First Last" fake that shares no word with any real name.

    Stable: the same set of names always gets the same mapping, whatever the order.
    Deleted-account display names ("Facebook User") are left out.
    """
    real = sorted({name for name in real_names if name.casefold() not in DELETED_ACCOUNT_NAMES})
    real_words = {compare_key(w) for name in real for w in WORD.findall(name)}

    def is_free(fake: str) -> bool:
        key = compare_key(fake)
        return not any(key == w or _shares_prefix(key, w) for w in real_words)

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
