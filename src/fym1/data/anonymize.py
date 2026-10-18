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
    # bare domain: lower-case only, and no "be"/"me" (Malagasy "be", "tsara.Be" typos)
    r"|\b(?-i:(?:[\w-]+\.)+(?:com|mg|fr|net|org|ly|io|co|gl))\b",
    re.IGNORECASE,
)
_SEP = r"\s*[./,_-]?\s*"
_INTL = r"(?:\+|(?<!\d[ .,])00)"  # "00" not inside a long amount: "1 000 000 000 Ar"
# A Malagasy number, written once so the spaced and glued rules accept the same forms.
_MG_COUNTRY = rf"(?:\(\s*{_INTL}?261\s*\)|{_INTL}?261)"  # +261, 00261, 261, (+261)
_ZERO_TRUNK = rf"(?:\(\s*0\s*\)|0){_SEP}"  # 0 or (0), then any gap


def _mg_phone(trunk: str) -> str:
    """A Malagasy number with the given leading-0 form."""
    return (
        rf"(?:{_MG_COUNTRY}{_SEP}(?:{trunk})?\d(?:{_SEP}\d){{8}}"  # +261 34 …, +261 (0)34 …, (+261)34 …
        # 034 12 345 67; the 0 may stand apart ("0 34 …") but "34" must touch, so "2h03 50 …" isn't a phone
        rf"|{trunk}[23]\d(?:{_SEP}\d){{7}}"
        rf"|\(\s*[0Oo]\d\d\s*\)(?:{_SEP}\d){{7}}(?!\w))"  # area code in brackets: (034) 12 345 67, (089) …
    )


# Spaced, a letter O typed for the 0 counts when it touches the digits ("O34…"). Glued to a word it doesn't:
# the word's own last "o" would be read as the 0 ("numero261 034…", "manao 300…").
_MG_PHONE_SPACED = _mg_phone(rf"(?:{_ZERO_TRUNK}|[Oo])")
_MG_PHONE_GLUED = _mg_phone(_ZERO_TRUNK)
PHONE = re.compile(
    r"(?<![\w+])(?:"
    rf"{_MG_PHONE_SPACED}"
    rf"|{_INTL}\d{{1,3}}(?:\s*\(\d{{1,4}}\))?(?:{_SEP}\d){{6,13}}"  # other intl: +33 6 …, +1 (555) …
    rf"|[0Oo](?:{_SEP}\d){{9}}"  # national, 10 digits however grouped; "O34" typed with a letter O
    # mobile without the 0: "34 12" as separate pairs, then any grouping ("34 12 2019/5"); a pair
    # after "3x" keeps price pairs out ("35 000, 3 500")
    rf"|3[2-9](?=[\s./,_-]){_SEP}\d\d(?=[\s./,_-]){_SEP}\d(?:{_SEP}\d){{4}}"
    rf"|3[2-9]{_SEP}\d\d{_SEP}\d{{3}}{_SEP}\d\d"  # 34 12 345 67 grouped or glued: "3412 345 67"
    r"|\d{9,15}"  # any other long run of digits
    r")(?!\w)"
    rf"|(?<![\d+]){_MG_PHONE_GLUED}(?!\d)"  # glued to a word: "laharana034 …", "wa+26134…", "tel(034) …"
)
# "01/02/2026 03/04" is dates, not a phone: the whole match must be one real day/month/year,
# plus at most one time or number group. A phone with a year-like group ("038 27 2000/5") or
# one that merely contains a date ("+261 34/12/2019/5") is still a phone; a match starting
# with a country code can never fit this shape.
_WHOLE_DATE = re.compile(
    r"(?:(?P<day>\d{1,2})[/-](?P<month>\d{1,2})[/-](?:19|20)\d\d"
    r"|(?:19|20)\d\d[/-](?P<month2>\d{1,2})[/-](?P<day2>\d{1,2}))(?:\s+\d{1,4})?"
)


def _is_whole_date(text: str) -> bool:
    match = _WHOLE_DATE.fullmatch(text.strip())
    if not match:
        return False
    day, month = int(match["day"] or match["day2"]), int(match["month"] or match["month2"])
    return 1 <= day <= 31 and 1 <= month <= 12
# Malagasy plot addresses: "Lot II M 85 Ter", "Lot IVG 12 Bis". Kept only if it holds a number.
_LOT_PART = r"(?:[IVX]+[A-Z]*|\d+[A-Za-z]*|[A-Z]{1,3}|[Bb]is|[Tt]er|[Qq]uater)\b"
LOT_ADDRESS = re.compile(rf"\b[Ll][Oo][Tt]\.?\s+{_LOT_PART}(?:[\s-]+{_LOT_PART})*")
# "ig: nirina.rk", "insta nirina_rk", "snap: nirinark" (not "fb 2 andro", "Ig: tsy mampiasa")
LABELLED_HANDLE = re.compile(
    r"\b(?P<label>ig|insta(?:gram)?|snap(?:chat)?|tiktok|tg|telegram|twitter|fb|facebook|viber|wa|whatsapp)"
    r"(?P<sep>\s*[:=]\s*@?|\s*@|\s+)(?P<handle>[\w.]*\w)",
    re.IGNORECASE,
)
# "@nirina.rk", or a display-name mention "@Marie-Claire Rasoanaivo": up to 3 words, read
# by Anonymizer._mention; it decides how many belong to the name. Never "@15h".
MENTION_MAX_WORDS = 3
_LETTER_START = r"[^\W\d_]"
_MENTION_GAP = r"[^\S\n\r\v\f\x1c-\x1e\x85\u2028\u2029]"  # any space on the same line
MENTION = re.compile(
    rf"(?<![\w.])@({_LETTER_START}(?:[\w.-]*\w)?)((?:{_MENTION_GAP}+{_LETTER_START}(?:[\w-]*\w)?){{0,{MENTION_MAX_WORDS - 1}}})"
)
_HANDLE_CHARS = re.compile(r"[._\d]")
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


def is_everyday_word(word: str) -> bool:
    """A common Malagasy word (or short texting word) that is never taken for a name."""
    key = compare_key(word)
    return key in _COMMON_KEYS or key in PROTECTED_SHORT_WORDS


def _shares_prefix(a: str, b: str) -> bool:
    """`a` and `b` are compare keys, one a prefix of the other of at least MIN_NAME_PREFIX letters."""
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= MIN_NAME_PREFIX and longer.startswith(shorter)


def _is_free(fake: str, real_keys: set[str]) -> bool:
    """`fake` shares no word or 4+ letter prefix with any real name part."""
    key = compare_key(fake)
    return not any(key == w or _shares_prefix(key, w) for w in real_keys)


def _letters_only(key: str) -> str:
    """`key` with every non-letter dropped, so "mia_ly" and "lov_va" read "mialy" and "lovva"."""
    return re.sub(r"[\W\d_]", "", key)


def _is_free_of_mention(name: str, own_keys: set[str]) -> bool:
    """`name` is free of the mention's words and sits inside none of them ("@itsmialy" -> not "Mialy")."""
    key = compare_key(name)
    return _is_free(name, own_keys) and not any(key in w for w in own_keys)


def _hashed_fake(text: str, words: int, firsts: list[str], lasts: list[str]) -> str:
    """A fake name derived from `text`, so the same handle always reads the same.

    Pool names that share a word or 4+ letter start with `text` itself, or sit inside one of
    its words or its letters joined up, are skipped, so "@Mialy", "@itsmialy" and "@mia_ly"
    never come out as "Mialy".
    """
    own_keys = {compare_key(p) for p in WORD.findall(text)} | {compare_key(_letters_only(fold(text)))}
    firsts = [f for f in firsts if _is_free_of_mention(f, own_keys)] or [FALLBACK_NAME]
    lasts = [s for s in lasts if _is_free_of_mention(s, own_keys)] or [FALLBACK_NAME]
    number = int.from_bytes(hashlib.sha256(compare_key(text).encode("utf-8")).digest()[:4], "big")
    first = firsts[number % len(firsts)]
    return first if words == 1 else f"{first} {lasts[number // len(firsts) % len(lasts)]}"


def _phone(match: re.Match) -> str:
    return match.group() if _is_whole_date(match.group()) else "<TEL>"


def _lot_address(match: re.Match) -> str:
    return "<ADDRESS>" if any(c.isdigit() for c in match.group()) else match.group()


def _labelled_handle(match: re.Match) -> str:
    sep, handle = match["sep"], match["handle"]
    is_word = handle[0].isdigit() or is_everyday_word(handle)
    # after a bare space, only something handle-shaped counts: "snap ve?" is just a word
    looks_like_handle = not is_word and (sep.strip() or _HANDLE_CHARS.search(handle))
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
        real_keys = set(self._exact)
        self._free_firsts = [f for f in FIRST_NAMES if _is_free(f, real_keys)] or list(FIRST_NAMES)
        self._free_lasts = [s for s in SURNAMES if _is_free(s, real_keys)] or list(SURNAMES)
        self.common_word_replacements = 0

    def __call__(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = EMAIL.sub("<EMAIL>", text)
        text = URL.sub("<URL>", text)
        text = PHONE.sub(_phone, text)
        text = LOT_ADDRESS.sub(_lot_address, text)
        text = LABELLED_HANDLE.sub(_labelled_handle, text)
        mentions: list[str] = []
        text = MENTION.sub(lambda m: self._mention(m, mentions), text)
        text = self._replace_names(text)
        return _SLOT.sub(lambda m: mentions[int(m.group(1))], text)

    @staticmethod
    def _park(parked: list[str], fake: str) -> str:
        """Hold `fake` aside so the name pass below cannot touch it."""
        parked.append(fake)
        return _PLACEHOLDER_SLOT.format(len(parked) - 1)

    def _mention(self, match: re.Match, parked: list[str]) -> str:
        """The mention's name part parked as a fake; words after the name stay as text."""
        body = match.group()[1:]
        spans = list(re.finditer(r"\S+", body))
        taken, fake = self._fake_for_mention([s.group() for s in spans])
        return self._park(parked, fake) + body[spans[taken - 1].end() :]

    def _fake_for_mention(self, words: list[str]) -> tuple[int, str]:
        """(how many leading `words` the mention covers, the one fake name replacing them).

        A capitalised mention covers every capitalised word after it ("@Nirina Mbola" ->
        one fake): the owner chose losing a word over leaking a surname. A lower-case
        one ("@haja rabe ok") runs while words are not everyday words.
        """
        if _HANDLE_CHARS.search(words[0]):  # @nirina.rk: a handle, one word
            known = self._fake_for_word(words[0])
            return 1, known or _hashed_fake(words[0], 1, self._free_firsts, self._free_lasts)
        taken = 1
        for word in words[1:]:
            if words[0][0].isupper() and not word[0].isupper():
                break
            if not words[0][0].isupper() and is_everyday_word(word):
                break
            taken += 1
        name = words[:taken]
        for length in range(taken, 1, -1):  # a participant's full name
            keys = tuple(compare_key(p) for w in name[:length] for p in WORD.findall(w))
            if keys in self._full_names:
                return taken, self._full_names[keys]
        known = []  # leading run of known name parts
        for word in name:
            fakes = [self._fake_for_word(p) for p in WORD.findall(word)]
            if not all(fakes):
                break
            known.append("-".join(fakes))
        if known:
            return taken, " ".join(known)
        return taken, _hashed_fake(" ".join(name), min(taken, 2), self._free_firsts, self._free_lasts)

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
        if key in _COMMON_KEYS:  # "farany" is not "Fara" + "-ny"
            return None
        for suffix in NAME_SUFFIXES:
            stem = key[: -len(suffix)]
            if key.endswith(suffix) and stem in self._exact and (fake := self._exact_fake(stem)):
                return fake + suffix
        if len(key) < MIN_NAME_PREFIX:
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
    real_keys = {compare_key(w) for name in real for w in WORD.findall(name)}
    firsts = [f for f in FIRST_NAMES if _is_free(f, real_keys)]
    lasts = [s for s in SURNAMES if _is_free(s, real_keys)]
    # surname-major order, so the first len(firsts) people all get different first names
    pool = [f"{first} {last}" for last in lasts for first in firsts]
    if not pool:
        return {name: f"{FALLBACK_NAME} {i + 1}" for i, name in enumerate(real)}
    fakes = {}
    for i, name in enumerate(real):
        round_number, index = divmod(i, len(pool))
        fakes[name] = pool[index] + (str(round_number + 1) if round_number else "")
    return fakes
