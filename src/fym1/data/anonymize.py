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
# "00" not inside a long amount ("1 000 000 000 Ar"), unless it reads "00261": "14h42 00261 34 …"
_INTL = r"(?:\+|(?<!\d[ .,])00|00(?=261))"
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
# Real dates are found in the whole text first, so a phone match can't start inside one:
# "15/03/2026 1 500 Ar" would otherwise read "03/2026 1 500" as a 10-digit number.
# dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy, dd/mm/yy, dd/mm, yyyy-mm-dd, and ranges of them
# ("15/03-20/03/2026"). Not after "digit/" or "digit.", nor before "/digit", ".digit" or
# "-digit", so "34/12/2019/5" and "038 27 2000/5" are not dates.
_DAY = r"(?:0?[1-9]|[12]\d|3[01])"
_MONTH = r"(?:0?[1-9]|1[0-2])"
_YEAR = r"(?:19|20)\d\d"
_ONE_DATE = "|".join(
    [rf"{_DAY}{sep}{_MONTH}{sep}(?:{_YEAR}|\d\d)" for sep in ("/", "-", r"\.")]
    + [rf"{_YEAR}{sep}{_MONTH}{sep}{_DAY}" for sep in ("/", "-", r"\.")]
    + [rf"{_DAY}/{_MONTH}"]
)
DATE = re.compile(rf"(?<!\d)(?<!\d[/.])(?:{_ONE_DATE})(?:\s*[-–]\s*(?:{_ONE_DATE}))*(?![/.-]?\d)")
# Clock times are set aside the same way: "03:02 034 …" must not read "02 034 …" as the phone.
TIME = re.compile(r"(?<![\w:])(?:[01]?\d|2[0-3])[:hH][0-5]\d(?![\w:])")
# A phone starting outside a date with one of these still wins over a date shape inside it,
# unless it continues an amount ("1 261 340 000 19/04", "1 020 000 15/03"). "03-24-…": the
# 3 may stand apart from the next digit by a mark, not by a space ("03 45" is a time).
_MOBILE_START = r"[0Oo]\s*\)?\s*[./,_-]?\s*[23](?:\s*[./,_-]\s*)?\d"
_STRONG_PHONE_START = re.compile(rf"\(?\s*(?:\+|00[1-9]|261|{_MOBILE_START})")
# A strong Malagasy start, then every digit after it: "034 12 01/02/26" and "261 32 01/11/12 12"
# carry a stray digit, so the exact-length PHONE misses them once a date shape sits inside.
_STRONG_RUN = re.compile(
    rf"(?<![\w+])(?:{_MG_COUNTRY}{_SEP}(?:{_ZERO_TRUNK})?|(?:\(\s*)?(?={_MOBILE_START})[0Oo]\s*\)?{_SEP})"
    rf"\d(?:{_SEP}\d){{8,}}"
)
# A grouped amount right before: "1 ", "25 000 " ("Ar 2500 261 …" and "35.000 261 …" are not one)
_AMOUNT_BEFORE = re.compile(r"(?<!\d)(?<!\d[ .,])\d{1,3}(?P<groups>(?:[ .,]\d{3})*)(?P<sep>[ .,])$")
_THOUSANDS_GROUP = re.compile(r"\d{3}(?!\d)")
# Digit groups still glued on by a mark after a phone match ("05 58 0326-59-58-41": the match
# takes "05 58 0326-59") belong to it, so no tail of the number is left in clear.
_PHONE_TAIL = re.compile(r"(?:\s*[./_-]\s*\d+)+")
_DATE_FILLER = "\ue010"  # private-use: not a digit, word, space or separator


def _continues_amount(text: str, blanked: str, start: int) -> bool:
    """A 3-digit group right after digits ("1 261 340 …"); dates and times before it are blanked,
    so "15/03/2026 034 …" and "12h16 261 …" are not amounts, nor "3 500 03-47-…" (a 2-digit group)."""
    before = _AMOUNT_BEFORE.search(blanked, 0, start)
    return bool(
        before
        and set(before["groups"][::4]) <= {before["sep"]}  # one separator all through
        and _THOUSANDS_GROUP.match(text, start)
    )


def _strong_spans_through_dates(text: str, blanked: str, dates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Strong phones (or strong digit runs) that start outside a date or time and run through one."""
    spans = []
    for pattern in (PHONE, _STRONG_RUN):
        pos = 0
        while match := pattern.search(text, pos):
            start, end = match.span()
            date_end = next((e for s, e in dates if s <= start < e), None)
            if date_end is not None:
                pos = date_end
                continue
            if (
                any(s < end and start < e for s, e in dates)
                and _STRONG_PHONE_START.match(match.group())
                and not _continues_amount(text, blanked, start)
            ):
                spans.append((start, end))
            pos = start + 1
    return spans


def _all_matches(pattern: re.Pattern, text: str) -> list[tuple[int, int]]:
    """Every match, overlapping ones too: in "35.000 00261 37 76 034 84" the leftmost match
    ("000 00261 3") hides the real phone; masking both leaves nothing of it."""
    spans, pos = [], 0
    while match := pattern.search(text, pos):
        spans.append(match.span())
        pos = match.start() + 1
    return spans


def _mask_phones(text: str) -> str:
    """Every phone becomes <TEL>; real dates and clock times are never read as part of one.

    Phones are matched on a copy where each date and time is blanked out, so "15/03/2026 1 500 Ar"
    keeps its price and "15/03/2026 034 12 345 67" masks only the phone. A strong phone
    (+, 00x, 261, 0[23]x, 0-3-x) that starts outside a date and runs through a date shape is
    matched on the real text and still wins, stray extra digits included: "034 01/02/26 7",
    "034 12 01/02/26". Digit groups glued on by a mark after a phone are masked with it.
    """
    dates = sorted(m.span() for pattern in (DATE, TIME) for m in pattern.finditer(text))
    chars = list(text)
    for start, end in dates:
        chars[start:end] = _DATE_FILLER * (end - start)
    blanked = "".join(chars)
    spans = _all_matches(PHONE, blanked)
    if dates:
        spans += _strong_spans_through_dates(text, blanked, dates)
    pieces, done = [], 0
    for start, end in sorted(spans):
        if tail := _PHONE_TAIL.match(blanked, end):
            end = tail.end()
        if start >= done:
            pieces += [text[done:start], "<TEL>"]
        done = max(done, end)  # an overlapping match widens the phone before it
    return "".join(pieces) + text[done:]


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
        text = _mask_phones(text)
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
