"""Replace personal details in chat text with placeholders or fake Malagasy names.

Known limit: a third party's name typed as plain text (no "@", not a participant or a
nickname) is not detected — that needs named-entity recognition. The owner's review
sheet is the backstop. See docs/anonymisation.md.
"""

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from itertools import pairwise
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


def _clean_amount(sep_name: str) -> str:
    """A price in clean 3-digit groups, one separator all through: "1 500", "25.000", "2 500 000"."""
    return rf"\d{{1,3}}(?P<{sep_name}>[ .,])\d{{3}}(?:(?P={sep_name})\d{{3}})*"


# A price that ends there (no digit after its own separator), or is followed by another price:
# "35 000, 3 500", "35.000 2015", "35 000 3 500"; not "34 123 45 67"
_PRICE_AHEAD = rf"{_clean_amount('price_sep')}(?:(?!\d|(?P=price_sep)\d)|[ .,]\s*\d{{1,3}}[ .,]\d{{3}}(?!\d))"


def _mg_phone(trunk: str) -> str:
    """A Malagasy number with the given leading-0 form."""
    return (
        rf"(?:{_MG_COUNTRY}{_SEP}(?:{trunk})?\d(?:{_SEP}\d){{8}}"  # +261 34 …, +261 (0)34 …, (+261)34 …
        # 034 12 345 67; the 0 may stand apart ("0 34 …") but "34" must touch, so "2h03 50 …" isn't a phone
        rf"|{trunk}[23]\d(?:{_SEP}\d){{7}}"
        rf"|\(\s*[0Oo]\d\d\s*\)(?:{_SEP}\d){{7}}(?!\w))"  # area code in brackets: (034) 12 345 67, (089) …
    )


# Spaced, a letter O typed for the 0 counts when it touches the digits ("O34…"). Glued to a word only
# the narrow rule below applies: the word's own last "o" would be read as the 0 ("numero261 034…").
_MG_PHONE_SPACED = _mg_phone(rf"(?:{_ZERO_TRUNK}|[Oo])")
# Right after a word ("laharana03 41 …", "tel03.41.…") a pair-style number may stand apart after
# its "03": two letters before it rule out a clock time ("2h03 50 …").
_AFTER_A_WORD = r"(?<=[^\W\d_]{2})"
_MG_PHONE_GLUED = rf"(?:{_mg_phone(_ZERO_TRUNK)}|{_AFTER_A_WORD}0[23]{_SEP}\d(?:{_SEP}\d){{7}})"
PHONE = re.compile(
    r"(?<![\w+])(?:"
    rf"{_MG_PHONE_SPACED}"
    rf"|{_INTL}\d{{1,3}}(?:\s*\(\d{{1,4}}\))?(?:{_SEP}\d){{6,13}}"  # other intl: +33 6 …, +1 (555) …
    rf"|[0Oo](?:{_SEP}\d){{9}}"  # national, 10 digits however grouped; "O34" typed with a letter O
    # mobile without the 0, any grouping ("34 12 345 67", "34 123 45 67", "34 1234567"), unless it
    # reads as a price in clean 3-digit groups ("35 000, 3 500", "32 500 000 Ar")
    rf"|(?!{_PRICE_AHEAD})3[2-9](?:{_SEP}\d){{7}}"
    r"|\d{9,15}"  # any other long run of digits
    r")(?!\w)"
    rf"|(?<![\d+]){_MG_PHONE_GLUED}(?!\d)"  # glued to a word: "laharana034 …", "wa+26134…", "tel(034) …"
)
# A letter O for the 0 right after a word ("WhatsAppO34…", "numéroo32…", "numO3 74 …"): only before a
# mobile (3, then 2-9) or Antananarivo (2, then 0) prefix, and not before an amount ("manao320 000 000",
# "io200000000"). Matched on the blanked copy only, so it never wins over a date the way a real 0 can.
_AMOUNT_AHEAD = r"\d{1,3}(?:[\s.,]\d{3})+(?!\d)|\d+0000(?!\d)"
_LETTER_O_PHONE = re.compile(
    rf"(?<=[^\W\d_])[Oo](?!{_AMOUNT_AHEAD})(?:3{_SEP}[2-9]|2{_SEP}0)(?:{_SEP}\d){{7}}(?!\d)"
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
# A Malagasy number has 9 digits after its 0 or its 261.
_SUBSCRIBER_DIGITS = 9
# A strong Malagasy start, then every digit after it: "034 12 01/02/26" and "261 32 01/11/12 12"
# carry a stray digit, so the exact-length PHONE misses them once a date shape sits inside.
_STRONG_RUN = re.compile(
    rf"(?<![\w+])(?:{_MG_COUNTRY}{_SEP}(?:{_ZERO_TRUNK})?|(?:\(\s*)?(?={_MOBILE_START})[0Oo]\s*\)?{_SEP})"
    rf"(?P<body>\d(?:{_SEP}\d){{{_SUBSCRIBER_DIGITS - 1},}})"
)
# A grouped amount right before: "1 ", "25 000 " ("Ar 2500 261 …" and "35.000 261 …" are not one)
_AMOUNT_BEFORE = re.compile(r"(?<!\d)(?<!\d[ .,])\d{1,3}(?P<groups>(?:[ .,]\d{3})*)(?P<sep>[ .,])$")
_DATE_FILLER = "\ue010"  # private-use: not a digit, word, space or separator
# The amount's groups end here: no digit follows, and no date glued on by a mark ("382-08-11-65"
# with the date blanked). A date after a space is its own word: "1 261 340 000 19/04".
_AMOUNT_ENDS = re.compile(rf"(?!\s*\d|\s*[./,_-]\s*[\d{_DATE_FILLER}])")
# Digit groups still glued on by a mark after a phone match ("05 58 0326-59-58-41": the match
# takes "05 58 0326-59") belong to it, so no tail of the number is left in clear.
_PHONE_TAIL = re.compile(r"(?:\s*[./_-]\s*\d+)+")
# Phones and dates are matched with each unicode dash read as "-" (‐ ‑ ‒ – — ― −) and each no-break
# space as " ", one character for one: "[ .,]" amount rules then see "2\u202f000\u202f000" as an amount.
_ASCII_MARKS = str.maketrans(
    dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-") | dict.fromkeys("\u00a0\u202f", " ")
)


def _amount_end(blanked: str, start: int) -> int | None:
    """Where the amount right before `start` ends, if the digits from `start` are more of its 3-digit groups.

    Dates and times are blanked, so "15/03/2026 034 …" and "12h16 261 …" have no amount before
    them. The amount must go on to its end in clean 3-digit groups with its own separator:
    "1 261 340 000 19/04" is one, but "Ar 25 000 000 261-341-26-06-08" and "5 261-3-76-…" are a
    phone after a number (a missed phone is worse than a masked price).
    """
    before = _AMOUNT_BEFORE.search(blanked, 0, start)
    if not before or not set(before["groups"][::4]) <= {before["sep"]}:  # one separator all through
        return None
    groups = re.compile(rf"\d{{3}}(?:{re.escape(before['sep'])}\d{{3}})*")
    rest = groups.match(blanked, start)
    return rest.end() if rest and _AMOUNT_ENDS.match(blanked, rest.end()) else None


def _stop_before_trailing_date(match: re.Match, dates: list[tuple[int, int]]) -> int:
    """Where a strong run ends: at the first date that starts once its subscriber digits are in.

    The run takes stray digits through a date shape ("034 12 01/02/26"), but a whole phone followed
    by a date keeps the date: "034 12 345 67 15/03/2026" -> "<TEL> 15/03/2026".
    """
    body_start, end = match.span("body")
    for date_start, _ in dates:
        digits_before_date = sum(c.isdigit() for c in match.string[body_start:date_start])
        if body_start < date_start < end and digits_before_date >= _SUBSCRIBER_DIGITS:
            return len(match.string[:date_start].rstrip())
    return end


# Numbers a date or time may stand next to without being part of a phone, each a whole word: a clock
# time with a space ("14 30"), a price in clean 3-digit groups ("1 500", "25.000"), a year ("2026"),
# a round number ("2500000"), or one short number ("150", "1500"). Several may follow each other
# ("2026 150 olona", "2025 150,000Ar"), but only one short number, not next to a price, and none
# with a time: "15/03 41 23 45 67", "15/03 412 345 67", "15/03 352 83 538", "15/03 22 34 5678" are phones.
_CLEAR_PIECE = re.compile(
    rf"(?<!\d)(?:(?P<time>(?:[01]?\d|2[0-3]) [0-5]\d)|(?P<amount>{_clean_amount('amount_sep')})"
    rf"|(?P<year>(?:19|20)\d\d)|(?P<round>\d+000)|(?P<short>\d{{1,4}}))(?![.,]?\d)"
)
_NEXT_PIECE = re.compile(r" +(?=\d)")
_NUMBER_BEFORE = re.compile(r"\d[ .,]?$")  # a number goes on right before: not a word start
# Numbers right before a phone shape, which may start the clear numbers it is part of: "02 " in
# "02 01 2029-02-02" starts the time "02 01", "350," in "350,000,000 15-03-2026" the price
_NUMBERS_BEFORE = re.compile(r"(?:\d+[ .,] *)+$")
# Digits that run on after a phone shape (any gap, no date or time in between)
_RUN_ON = re.compile(r"(?:\s*[./,_-]?\s*\d)+")


def _clear_number_end(blanked: str, start: int) -> int | None:
    """Where the clear numbers starting at `start` end, if they are clear (see _CLEAR_PIECE)."""
    if _NUMBER_BEFORE.search(blanked, 0, start):
        return None  # not a word start: "5 034", "12,5"
    kinds, pos = [], start
    while piece := _CLEAR_PIECE.match(blanked, pos):
        kinds.append(piece.lastgroup)
        pos = piece.end()
        if not (gap := _NEXT_PIECE.match(blanked, pos)):
            break
        pos = gap.end()
    else:
        return None  # a word that is no clear number
    short_by_a_price = any({a, b} == {"amount", "short"} for a, b in pairwise(kinds))
    if kinds.count("short") > 1 or short_by_a_price or ("time" in kinds and len(kinds) > 1):
        return None
    return piece.end() if _AMOUNT_ENDS.match(blanked, piece.end()) else None


def _explained_by_dates(blanked: str, start: int, end: int, strong: bool, found: list[tuple[int, int]]) -> bool:
    """Every digit from `start` to `end` is in a date or time, a phone already found, or a clear number.

    `blanked` has dates and times blanked out. A strong start ("034", "+261", "261") counts only as
    more groups of an amount right before it ("1 034 000 15/03"), and nothing else of it may lie
    outside a date. Elsewhere, digits outside dates must be clear numbers (see _CLEAR_PIECE), which
    may start a little before a weak start ("02 01 2029-02-02"): "15/03/2026 1 500 Ar" and
    "15/03/2026 14 30" are explained, "12/03 41 23 45 67" and "15/03/2026 1 500 34 12" are not.
    """
    pos = start
    while pos < end:
        if not blanked[pos].isdigit() or any(s <= pos < e for s, e in found):
            pos += 1
            continue
        if strong:
            number_end = _amount_end(blanked, pos) if pos == start else None
        elif pos == start:
            number_end = _amount_end(blanked, pos) or _clear_number_end(blanked, pos)
            if number_end is None and (before := _NUMBERS_BEFORE.search(blanked, 0, pos)):
                word_starts = [m.start() for m in re.finditer(r"(?<![\d.,])\d", blanked[:pos])]
                ends = (_clear_number_end(blanked, w) for w in word_starts if w >= before.start())
                number_end = next((e for e in ends if e is not None and e > pos), None)
        else:
            number_end = _clear_number_end(blanked, pos)
        if number_end is not None and number_end <= pos:
            number_end = None
        if number_end is None:
            return False
        pos = number_end
    return True


def _date_overlapping_spans(
    marks: str, blanked: str, dates: list[tuple[int, int]], found: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Phone shapes that touch a date or time and are not fully explained by it: masked, fail closed.

    Every phone shape is tried from every start, inside a date too ("12/03 41 23 45 67" hides
    "03 41 23 45 67"). It is spared only when its digits lie in dates and times, plus a clear
    price or time next to them. A strong start (+, 00x, 261, 0[23]x, 0-3-x) outside a date is judged
    first: it runs through stray digits ("034 12 01/02/26") but stops before a date that follows a
    whole phone ("034 12 345 67 15/03/2026"). Any other shape is also spared when its digits past
    the date are a phone masked anyway ("15/03/2026 034 12 345 67" keeps its date); `found` are the
    phones matched with dates blanked.
    """
    candidates = [(m, m.end()) for m in _all_match_objects(PHONE, marks)]
    candidates += [(m, _stop_before_trailing_date(m, dates)) for m in _all_match_objects(_STRONG_RUN, marks)]
    touching = [
        (m.start(), end, m)
        for m, end in candidates
        if any(s < end and m.start() < e for s, e in dates) and not _starts_with_date_digits(m, dates)
    ]
    strong = [
        (start, end)
        for start, end, match in touching
        if not any(s <= start < e for s, e in dates) and _STRONG_PHONE_START.match(match.group())
    ]
    spans = [
        _with_run_on(blanked, start, end)
        for start, end in strong
        if not _explained_by_dates(blanked, start, end, True, [])
    ]
    masked = found + spans
    for start, end, _ in touching:
        if (start, end) not in strong and not _explained_by_dates(blanked, start, end, False, masked):
            spans.append(_with_run_on(blanked, start, end))
    return spans


def _starts_with_date_digits(match: re.Match, dates: list[tuple[int, int]]) -> bool:
    """A shape starting inside a date or time on a digit other than 0: "39 00 12 617" in "09:39 00 12 617".

    A phone carries its 0 (or +, 261) into a date; a no-0 mobile's "3x" read off a date's last
    digits is the date's own.
    """
    inside = any(s <= match.start() < e for s, e in dates)
    return inside and match.group()[0] in "123456789"


def _with_run_on(blanked: str, start: int, end: int) -> tuple[int, int]:
    """The span widened over digits that run on after it, so no tail of the number is left."""
    run_on = _RUN_ON.match(blanked, end)
    return start, run_on.end() if run_on else end


def _all_match_objects(pattern: re.Pattern, text: str) -> list[re.Match]:
    """Every match from every start, overlapping ones too."""
    matches, pos = [], 0
    while match := pattern.search(text, pos):
        matches.append(match)
        pos = match.start() + 1
    return matches


def _all_matches(pattern: re.Pattern, text: str) -> list[tuple[int, int]]:
    """Every match, overlapping ones too: in "35.000 00261 37 76 034 84" the leftmost match
    ("000 00261 3") hides the real phone; masking both leaves nothing of it."""
    return [match.span() for match in _all_match_objects(pattern, text)]


def _mask_phones(text: str) -> str:
    """Every phone becomes <TEL>; real dates and clock times are never read as part of one.

    Phones are matched on a copy where each date and time is blanked out, so "15/03/2026 1 500 Ar"
    keeps its price and "15/03/2026 034 12 345 67" masks only the phone. Then every phone shape
    that touches a date or time is matched on the real text and masked unless the date explains
    all its digits (fail closed): "12/03 41 23 45 67" -> "12/<TEL>", "034 01/02/26 7" -> "<TEL>".
    Digit groups glued on by a mark after a phone are masked with it.
    Unicode dashes are read as "-" and no-break spaces as " ": "034–12–345–67" is a phone,
    "15–03–2026" a date.
    """
    marks = text.translate(_ASCII_MARKS)  # same length as `text`, so its spans fit `text`
    dates = sorted(m.span() for pattern in (DATE, TIME) for m in pattern.finditer(marks))
    chars = list(marks)
    for start, end in dates:
        chars[start:end] = _DATE_FILLER * (end - start)
    blanked = "".join(chars)
    spans = _all_matches(PHONE, blanked) + _all_matches(_LETTER_O_PHONE, blanked)
    if dates:
        spans += _date_overlapping_spans(marks, blanked, dates, spans)
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
# "ig: hanta.rk", "insta hanta_rk", "snap: hantark" (not "fb 2 andro", "Ig: tsy mampiasa")
LABELLED_HANDLE = re.compile(
    r"\b(?P<label>ig|insta(?:gram)?|snap(?:chat)?|tiktok|tg|telegram|twitter|fb|facebook|viber|wa|whatsapp)"
    r"(?P<sep>\s*[:=]\s*@?|\s*@|\s+)(?P<handle>[\w.]*\w)",
    re.IGNORECASE,
)
# "@hanta.rk", or a display-name mention "@Marie-Claire Rasoanaivo": up to 3 words, read
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


# Left after NFKC: combining marks no letter absorbed ("fancy text" strike-through U+0336, underline U+0332)
# and zero-width spaces. Both split a name into one-letter pieces. Variation selectors (emoji) are kept here.
_ZERO_WIDTH = frozenset("\u200b\u2060\ufeff")
# Dropped only right after a letter, where they only hide a name ("R\u200da\u200dk..."): format characters
# (ZWJ, ZWNJ, soft hyphen, U+2062, tags), enclosing marks (U+20DD, keycap U+20E3) and variation selectors.
# After an emoji, digit, "#" or "*" they build emoji sequences (families, "\u2764\ufe0f", keycaps) and stay.
_HIDING_CATEGORIES = frozenset({"Cf", "Me"})


def _is_stray(c: str) -> bool:
    if c in _ZERO_WIDTH:
        return True
    return unicodedata.category(c) == "Mn" and not is_variation_selector(c)


def is_variation_selector(c: str) -> bool:
    return "\ufe00" <= c <= "\ufe0f" or "\U000e0100" <= c <= "\U000e01ef"


def _hides_a_letter(c: str) -> bool:
    return unicodedata.category(c) in _HIDING_CATEGORIES or is_variation_selector(c)


def normalise(text: str) -> str:
    """`text` NFKC-normalised (full-width, circled forms read as plain), stray marks and zero-width spaces dropped.

    NFKC composes accents first ("e" + U+0301 -> "é"), so only marks no letter absorbed are dropped. Format
    characters, enclosing marks and variation selectors are dropped right after a letter; emoji keep theirs.
    """
    kept: list[str] = []
    for c in unicodedata.normalize("NFKC", text):
        if _is_stray(c) or (kept and kept[-1].isalpha() and _hides_a_letter(c)):
            continue
        kept.append(c)
    return "".join(kept)


def fold(s: str) -> str:
    """Lower-case and strip accents, so "Hélène" and "helene" compare equal."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).casefold()


def compare_key(word: str) -> str:
    """How words are compared: folded, with stretched letters collapsed ("SOAAA" -> "soa")."""
    return re.sub(r"(.)\1+", r"\1", fold(word))


def name_keys(name: str) -> list[str]:
    """Compare keys of a real name's words, read through the same `normalise()` as the text.

    The one entry point for turning a real name into match keys: a zero-width space or soft hyphen
    inside a display name ("Han\u200bta") must not split it where the normalised text does not.
    """
    return [compare_key(part) for part in WORD.findall(normalise(name))]


# Words found in the most Malagasy Wikipedia articles, most widespread first
# (scripts/build_common_words.py). A name part never prefix-matches one of these.
COMMON_WORDS = tuple(files("fym1.data").joinpath("mg_common_words.txt").read_text("utf-8").split())
_COMMON_KEYS = {compare_key(w) for w in COMMON_WORDS}
# Short words a short name part ("Ny", "Fa") must never replace: the most widespread
# Wikipedia words plus texting words Wikipedia lacks.
TEXTING_WORDS = frozenset({"ve", "eh", "e", "ah", "ahn", "aho", "ry", "o", "ok"})
PROTECTED_SHORT_WORDS = {compare_key(w) for w in COMMON_WORDS[:300]} | TEXTING_WORDS


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
            keys, fake_parts = name_keys(real), names[real].split()
            if len(keys) > 1:
                self._full_names[tuple(keys)] = names[real]
            for i, key in enumerate(keys):
                # first name -> fake first name, any other part -> fake surname
                self._exact.setdefault(key, fake_parts[min(i, 1, len(fake_parts) - 1)])
        self._longest_name = max(map(len, self._full_names), default=1)
        self._prefixable = [(part, fake) for part, fake in self._exact.items() if len(part) >= MIN_NAME_PREFIX]
        self._fake_words = {compare_key(w) for fake in names.values() for w in fake.split()}
        real_keys = set(self._exact)
        self._free_firsts = [f for f in FIRST_NAMES if _is_free(f, real_keys)] or list(FIRST_NAMES)
        self._free_lasts = [s for s in SURNAMES if _is_free(s, real_keys)] or list(SURNAMES)
        self.common_word_replacements = 0
        self.mention_fakes: set[str] = set()  # every fake written for an @mention, for the strict gate

    def __call__(self, text: str) -> str:
        text = normalise(text)  # full-width "＠", circled "Ⓡ" read as plain; stray marks dropped
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
        self.mention_fakes.add(fake)
        return self._park(parked, fake) + body[spans[taken - 1].end() :]

    def _fake_for_mention(self, words: list[str]) -> tuple[int, str]:
        """(how many leading `words` the mention covers, the one fake name replacing them).

        A capitalised mention covers every capitalised word after it ("@Hanta Mbola" ->
        one fake): the owner chose losing a word over leaking a surname. A lower-case
        one ("@haja rabe ok") runs while words are not everyday words.
        """
        if _HANDLE_CHARS.search(words[0]):  # @hanta.rk: a handle, one word
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
    4+ letter prefix with a name part ("hant") is replaced too, unless it is a common
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
    real_keys = {key for name in real for key in name_keys(name)}
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
