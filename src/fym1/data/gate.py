"""Strict automatic gate run after anonymising: drop any conversation that may still leak.

No person reads the anonymised text, so the gate fails closed: replacing or dropping too much is
fine, letting a name or number through is not. A whole conversation is dropped if any turn holds:

- `long_number`: 7+ digits in one run, counting across any non-letter characters (every kind of
  space, `. , / \\ - _ ( ) + ; * | :`, unicode dashes). A price is kept only when it
  is plain digits or clean 3-digit groups, is followed by a currency word (`1 500 000 Ar`,
  `2500000 ariary`, `1 250 000,50 €`), and does not look like a phone: no `+` before it and its
  digits do not start with `0`, `261` or read `3[2-9]` + 7 digits. A run of 9-10 digits is a phone's
  length, so it is a price only with grouped thousands ending `000` or cents ("150 000 000 Ar",
  "123 456 789,00 €"). A full date (8 digits) counts too.
- `contact`: an email, a link with a path or query, an `@` before a word or a placeholder (spaces
  allowed: "@ njaka", "njaka@ <URL>"), an `at`/`arobase`/`(at)`/`[at]` before a placeholder or a
  mail host ("njaka at gmail"), or a private-use mark the anonymiser leaves only when a
  replacement failed.
- `too_many_unknown`: more than 30% of its words would be replaced as names (below).
- `too_many_fakes`: more than MAX_FAKES_PER_CONVERSATION (5) words would be replaced as names. A
  model trained on chats full of fakes learns to sprinkle random names into its replies.
- `hidden_char`: an invisible or combining character normalise() left (a spacing mark inside a name,
  a Hangul filler, a format character not after a letter), unless it builds an emoji (variation
  selector, ZWJ between emoji, keycap after a digit/#/*, subdivision-flag tags).

Every turn is NFKC-normalised first, so full-width (`＠`, `Ｒ`) and circled (`Ⓡ`) letters are
checked as plain ones; combining marks NFKC leaves loose (strike-through, underline) and zero-width
spaces are dropped, and so are format characters, enclosing marks and variation selectors right after a
letter (emoji keep theirs). Kept turns are returned normalised.

Words are letter runs, split at digits, underscores, punctuation, `<`/`>` and a lower-to-upper case change
("Rakoto2", "jean_r", "salutRakoto", "<<Njaka"); only the exact placeholders (`<TEL>`, `<EMAIL>`, ...)
are skipped. Letters with digits inside ("Rak0to", "h4nt4") are one word and always a name. A run of
3+ single letters joined by 1-3 non-letter, non-digit characters ("r a k o t o", "R.a.k.o.t.o", "R_a_k",
"J,E,A,N", "R . a . k", "R🔥a🔥k") is one word ("rakoto"), checked by the rules below and replaced whole.

Capitalised words that are not fakes the anonymiser wrote (`fake_names`) or currency words are:

- cleared at a sentence start if a known-word list has them (`fym1.data.known_words`), and anywhere
  else (mid-sentence, after a person marker) only if the curated hand list `known_names` has them.
  Person names (first names, the Malagasy name deny list "Hery", "Soa", the fake-name pool) never clear.
- replaced otherwise by a fake first name from the anonymiser's pool, the same fake for the same
  word within one conversation, never a word already in that conversation.

A pool name the raw conversation already held ("Rabe no nandeha") is a real person there, not
our fake: the caller passes it in `disallowed` and it is always replaced, word or not.

Lower-case words fail closed too. Name lists cannot hold every name, so any lower-case word that no
known-word list knows is replaced like an unknown capitalised word ("salut tsanta", "rakoto",
"sandrinerakoto"). Kept without a marker: words a list knows ("trano", "lol", "mdr"), and names that
are also everyday words (`name_deny_list.txt`: "soa", "vola"; `wordlike_first_names.txt`: "une
rose"). Also replaced:

- after a person marker (`i`, `ry`, `ra`, `n'i`, `an'i`, `amin'i`, `tamin'i`) or an `@`: any word that
  is a person name or that no list but `mg_corpus` knows ("ry hery"); "ry namana", "i dada" stay. Web
  text types names in lower case too, so `mg_corpus` alone never clears a name position; in running
  text and at a sentence start it does.
- any `Ra-`/`ra-` hyphenated word ("Ra-be"), and any first name hyphen-joined to another
  ("jean-pierre", "marie-claire").

Only counts leave this module; never text or words.
"""

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping
from statistics import median_low
from typing import NamedTuple

from fym1.data.anonymize import (
    EMAIL,
    FALLBACK_NAME,
    FIRST_NAMES,
    SURNAMES,
    compare_key,
    fold,
    is_variation_selector,
    normalise,
)
from fym1.data.known_words import (
    FIRST_NAME_WORDS,
    KNOWN_WORDS,
    MG_FIRST_NAMES,
    NAME_DENY_LIST,
    WORDLIKE_FIRST_NAMES,
    Lexicon,
    known_by,
)
from fym1.data.leftovers import MIN_LETTERS, fake_name_keys

GATE_REASONS = ("long_number", "contact", "too_many_unknown", "too_many_fakes", "hidden_char")
HAND_LIST = "known_names"  # the only list that clears a word mid-sentence or after a marker
# Web text holds names typed in lower case, so this list alone never clears a name position (after a
# person marker or "@"); elsewhere it does. Lead decision: corpus-only name slots.
CORPUS_LIST = "mg_corpus"
MAX_REPLACED_SHARE = 0.3
MAX_FAKES_PER_CONVERSATION = 5  # lead decision: AUTO-DECISIONS 22
# Gate counts copied into results/data_stats.json, so the training data records how many fakes it holds.
FAKE_STATS_KEYS = ("dropped_too_many_fakes", "fakes_per_kept_median", "fakes_per_kept_max")

MIN_NUMBER_DIGITS = 7
# Any run of non-letter, non-digit characters joins digits: spaces of every kind, ". , / - _ ( ) + ; * | :"...
DIGIT_RUN = re.compile(r"\d(?:[\W_]*\d)*")
CURRENCY_AFTER = re.compile(r"\s*(?:€|(?:Ar|AR|Ariary|ariary|fmg|FMG|euros?|eur)(?!\w))")
CURRENCY_WORDS = ("Ar", "Ariary", "Fmg", "Euro", "Euros", "Eur")  # never a leftover name
# Clean price shapes: "1 500 000", "1.500.000", "2500000", each with optional cents ",50".
PRICE = re.compile(r"\d{1,3}(?:([ \u00a0\u202f.,])\d{3})?(?:\1\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?")
# Digits of a Malagasy phone: trunk 0, country code 261 (or 00261), or a 9-digit mobile without the 0.
PHONE_DIGITS = re.compile(r"(?:0|261|3[2-9]\d{7}$)")
# A phone's length without its trunk 0 or with it (Malagasy mobiles 3x and landlines 2x, French mobiles
# [67] without the 0, ...). Too many shapes to list, so a run this long fails closed: it is a price only
# with grouped thousands ending 000 ("150 000 000") or cents ("123 456 789,00").
PHONE_LENGTHS = (9, 10)
CENTS = re.compile(r"[.,]\d{1,2}$")

LINK_WITH_PATH = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:[\w-]+\.)+[a-z]{2,}[/?]\S*", re.IGNORECASE)
_PLACEHOLDER_AFTER = r"<(?:URL|EMAIL|HANDLE)>"
# "@njaka", "@ njaka", "njaka@ <URL>"; not "@ 3 ora"
HANDLE = re.compile(rf"@(?:\w|\s*(?:[^\W\d_]|{_PLACEHOLDER_AFTER}))")
MAIL_HOSTS = r"(?:gmail|yahoo|hotmail|outlook|live|ymail|icloud)\b"
SPELLED_AT = re.compile(
    rf"\w\s*(?:\b(?:at|arobase)\b|\(at\)|\[at\])\s*(?:{_PLACEHOLDER_AFTER}|{MAIL_HOSTS})", re.IGNORECASE
)
PRIVATE_USE = re.compile("[\ue000-\uf8ff]")
SENTENCE_END = ".!?…\n"

PLACEHOLDERS = ("<TEL>", "<EMAIL>", "<URL>", "<HANDLE>", "<ADDRESS>")  # what the anonymiser writes
# An exact placeholder, letters with digits inside ("Rak0to", "h4nt4": one word, always a name), or a
# letter run ("Ra-be" taken whole); otherwise digits and "_" end a run.
_LEET = r"[^\W\d_]+(?:\d+[^\W\d_]+)+"
# 3+ single letters joined by 1-3 non-letter, non-digit characters ("r a k o t o", "R_a_k", "R . a . k",
# "R🔥a🔥k", combining marks normalise() keeps): one word.
_ONE = r"(?<![^\W\d_])[^\W\d_](?![^\W\d_])"
_SPELLED = rf"(?P<spelled>{_ONE}(?:[\W_]{{1,3}}{_ONE}){{2,}})"
_PIECE = re.compile("|".join(map(re.escape, PLACEHOLDERS)) + rf"|{_LEET}|{_SPELLED}|(?:[Rr]a-)?[^\W\d_]+")
LETTER_RUN = re.compile(r"[^\W\d_]+")
# Words that put a person's name after them ("i Soa", "ry Bodo", "ra Fara"); also typed glued on ("iSoa").
PERSON_MARKER_WORDS = ("i", "ry", "ra")
PERSON_MARKER = re.compile(rf"(?:(?<![\w'’])(?:{'|'.join(PERSON_MARKER_WORDS)})|\w['’]i)\s+$", re.IGNORECASE)
MARKER_WINDOW = 12  # characters looked back for a marker
AT_BEFORE = re.compile(r"@\s*$")  # "@njaka", "@ njaka": a handle, a name position
# Pool names (and the fallback) written in raw text: real people there, never our fakes.
RAW_POOL_KEYS = frozenset(compare_key(name) for name in (*FIRST_NAMES, *SURNAMES, FALLBACK_NAME))
ANY_FIRST_NAME = FIRST_NAME_WORDS | WORDLIKE_FIRST_NAMES
# Names that are also everyday words: kept in lower case unless after a person marker ("soa ve", "une rose").
# The fallback fake "Olona" is the everyday word for "person".
EVERYDAY_NAMES = (NAME_DENY_LIST | WORDLIKE_FIRST_NAMES | {fold(FALLBACK_NAME)}) - MG_FIRST_NAMES
NAME_BEFORE_HYPHEN = re.compile(r"([^\W\d_]+)-$")
NAME_AFTER_HYPHEN = re.compile(r"-([^\W\d_]+)")
HYPHEN_WINDOW = 30  # characters looked back for "jean-" before "pierre"


class GateResult(NamedTuple):
    kept: list[dict]
    counts: dict[str, int]  # counts only, safe to print and copy into results/


class Token(NamedTuple):
    start: int
    end: int
    word: str


def tokens(text: str) -> Iterator[Token]:
    """Words of `text`: letter runs split at a lower-to-upper case change; placeholders skipped."""
    for match in _PIECE.finditer(text):
        piece, start = match.group(), match.start()
        if piece in PLACEHOLDERS:
            continue
        if match.group("spelled"):  # "N j a k a R a b e": checked whole as "NjakaRabe"
            yield Token(start, match.end(), "".join(LETTER_RUN.findall(piece)))
            continue
        if any(c.isdigit() for c in piece):  # "Rak0to": kept whole
            yield Token(start, match.end(), piece)
            continue
        for i in range(1, len(piece)):
            if piece[i - 1].islower() and piece[i].isupper():  # "salutRakoto"
                yield Token(start, match.start() + i, text[start : match.start() + i])
                start = match.start() + i
        yield Token(start, match.end(), text[start : match.end()])


def _looks_like_a_phone(text: str, run: re.Match) -> bool:
    digits = re.sub(r"\D", "", run.group())
    after_plus = text[: run.start()].rstrip().rstrip("(").rstrip().endswith("+")
    return after_plus or bool(PHONE_DIGITS.match(digits))


def _is_price(text: str, run: re.Match) -> bool:
    shape = PRICE.fullmatch(run.group())
    if not shape or not CURRENCY_AFTER.match(text, run.end()) or _looks_like_a_phone(text, run):
        return False
    whole = CENTS.sub("", run.group())
    if sum(c.isdigit() for c in whole) not in PHONE_LENGTHS:
        return True
    return whole != run.group() or (shape.group(1) is not None and whole.endswith("000"))


def has_long_number(text: str) -> bool:
    """`text` holds 7+ digits in one run that is not a clean price followed by a currency word."""
    for run in DIGIT_RUN.finditer(text):
        if sum(c.isdigit() for c in run.group()) >= MIN_NUMBER_DIGITS and not _is_price(text, run):
            return True
    return False


# Default-ignorable code points no category check below catches: Hangul fillers are letters (Lo),
# U+FFF0-FFF8 are unassigned.
_IGNORABLE_LETTERS = frozenset("\u115f\u1160\u3164\uffa0") | frozenset(map(chr, range(0xFFF0, 0xFFF9)))
_HIDDEN_CATEGORIES = frozenset({"Cf", "Mn", "Me", "Mc"})
ZWJ, KEYCAP, BLACK_FLAG = "\u200d", "\u20e3", "\U0001f3f4"
KEYCAP_BASES = frozenset("0123456789#*")


def _is_tag(c: str) -> bool:
    return "\U000e0020" <= c <= "\U000e007f"


def _is_emoji_part(c: str) -> bool:
    return unicodedata.category(c) in ("So", "Sk") or is_variation_selector(c)


def _in_emoji_sequence(text: str, i: int) -> bool:
    """The invisible character at `text[i]` builds an emoji: VS, ZWJ between emoji, keycap, flag tag."""
    c = text[i]
    before = text[i - 1] if i else ""
    if is_variation_selector(c):  # normalise() already dropped those after a letter
        return True
    if c == ZWJ:
        after = text[i + 1] if i + 1 < len(text) else ""
        return bool(before and after) and _is_emoji_part(before) and _is_emoji_part(after)
    if c == KEYCAP:  # "1\u20e3", "1\ufe0f\u20e3"
        base = text[i - 2] if before == "\ufe0f" and i >= 2 else before
        return base in KEYCAP_BASES
    if _is_tag(c):  # subdivision flags: black flag + tag letters + cancel tag
        j = i - 1
        while j >= 0 and _is_tag(text[j]):
            j -= 1
        return j >= 0 and text[j] == BLACK_FLAG
    return False


def is_hidden_char(c: str) -> bool:
    """`c` is invisible or combining: a format character, a combining mark or a default-ignorable letter."""
    return c in _IGNORABLE_LETTERS or unicodedata.category(c) in _HIDDEN_CATEGORIES


def has_hidden_char(text: str) -> bool:
    """`text` (normalised) still holds an invisible or combining character that is not part of an emoji.

    normalise() drops the ones that hide a letter; anything left (a spacing mark inside a name, a Hangul
    filler, a format character after a space) fails closed. NFC-composed accents are letters, not marks.
    """
    return any(is_hidden_char(c) and not _in_emoji_sequence(text, i) for i, c in enumerate(text))


def contact_left(text: str) -> bool:
    """An email, link with a path or query, @handle, or failed-replacement mark is left in `text`."""
    return any(p.search(text) for p in (EMAIL, LINK_WITH_PATH, HANDLE, SPELLED_AT, PRIVATE_USE))


def _sentence_initial(text: str, start: int) -> bool:
    before = text[:start].rstrip(" \t\u00a0\u202f\"'«(")
    return not before or before[-1] in SENTENCE_END


def _in_name_slot(text: str, start: int) -> bool:
    """The word at `start` follows a person marker ("i", "ry", "amin'i") or an "@"."""
    window = max(0, start - MARKER_WINDOW)
    return bool(PERSON_MARKER.search(text, window, start) or AT_BEFORE.search(text, window, start))


def pool_names_in(texts: Iterable[str]) -> set[str]:
    """Keys of the fake-name pool's names written in `texts` in any case (raw, before anonymising)."""
    parts = (part for text in map(normalise, texts) for token in tokens(text) for part in LETTER_RUN.findall(token.word))
    return {key for part in parts if (key := compare_key(part)) in RAW_POOL_KEYS}


def _is_capitalised(word: str, fake_keys: set[str]) -> bool:
    return word[0].isupper() and len(word) >= MIN_LETTERS and compare_key(word) not in fake_keys


def _in_compound_first_name(text: str, token: Token) -> bool:
    """`token` is a first name hyphen-joined to another first name ("jean-pierre")."""
    if fold(token.word) not in ANY_FIRST_NAME:
        return False
    before = NAME_BEFORE_HYPHEN.search(text, max(0, token.start - HYPHEN_WINDOW), token.start)
    after = NAME_AFTER_HYPHEN.match(text, token.end)
    return any(m and fold(m.group(1)) in ANY_FIRST_NAME for m in (before, after))


def _is_lowercase_name(text: str, token: Token, lexicon: Lexicon) -> bool:
    """A lower-case word that is a name by its place or shape (see the module docstring)."""
    word = token.word
    if not word[0].islower() or len(word) < MIN_LETTERS:
        return False
    if "-" in word or any(c.isdigit() for c in word):  # "ra-koto", "h4nt4"
        return True
    if _in_compound_first_name(text, token):
        return True
    if _in_name_slot(text, token.start):
        return known_by(word, _without_corpus(lexicon)) is None
    unknown = known_by(word, lexicon) is None  # person names are never known
    return unknown and fold(word) not in EVERYDAY_NAMES


def _clearing_list(text: str, token: Token, lexicon: Lexicon) -> str | None:
    """The list that clears a capitalised word here: any list at a sentence start, else the hand list."""
    if _sentence_initial(text, token.start):
        return known_by(token.word, lexicon)
    return known_by(token.word, {HAND_LIST: lexicon[HAND_LIST]}) if HAND_LIST in lexicon else None


def _fake_for(key: str, taken: set[str]) -> str:
    """A pool first name picked from `key`, so the same word gets the same fake; none of `taken`."""
    free = [name for name in FIRST_NAMES if compare_key(name) not in taken] or [FALLBACK_NAME]
    number = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")
    return free[number % len(free)]


class _Names(NamedTuple):
    texts: list[str]  # with unknown capitalised words replaced
    words: int
    replaced: int
    cleared: Counter  # list name -> capitalised words it cleared


def _clear_or_replace(texts: list[str], fake_keys: set[str], lexicon: Lexicon, disallowed: set[str]) -> _Names:
    """One conversation's turns with every unknown or disallowed name swapped for a fake first name."""
    taken = {compare_key(t.word) for text in texts for t in tokens(text)} | disallowed
    fakes: dict[str, str] = {}
    cleared, replaced = Counter(), 0

    def swap(text: str, token: Token) -> str:
        nonlocal replaced
        word, key = token.word, compare_key(token.word)
        if not _is_lowercase_name(text, token, lexicon):
            if not _is_capitalised(word, fake_keys):
                return text[token.start : token.end]
            source = None if key in disallowed else _clearing_list(text, token, lexicon)
            if source:
                cleared[source] += 1
                return text[token.start : token.end]
        if key not in fakes:
            fakes[key] = _fake_for(key, taken)
            taken.add(compare_key(fakes[key]))
        replaced += 1
        return fakes[key]

    new_texts, words = [], 0
    for text in texts:
        pieces, done = [], 0
        for token in tokens(text):
            pieces += [text[done : token.start], swap(text, token)]
            done = token.end
            words += 1
        new_texts.append("".join(pieces) + text[done:])
    return _Names(new_texts, words, replaced, cleared)


def _without_corpus(lexicon: Lexicon) -> Lexicon:
    """`lexicon` minus the web-corpus list, which alone never clears a name position."""
    return {name: words for name, words in lexicon.items() if name != CORPUS_LIST}


def unknown_capitalised_words(texts: Iterable[str], fake_names: Iterable[str], lexicon: Lexicon = KNOWN_WORDS) -> int:
    """How many capitalised words of one conversation's turns no list clears. A count only.

    For synthetic chats (lead decision 30): their few-shot examples hold only fakes, so lower-case words are not
    checked. A capitalised word passes where strict_gate clears it, and also mid-sentence (not after a person
    marker or "@") when a list other than `mg_corpus` knows it. `fake_names` and currency words pass.
    """
    fake_keys = fake_name_keys([*fake_names, *CURRENCY_WORDS])
    without_corpus = _without_corpus(lexicon)
    unknown = 0
    for text in map(normalise, texts):
        for token in tokens(text):
            if not _is_capitalised(token.word, fake_keys) or _clearing_list(text, token, lexicon):
                continue
            in_running_text = not _sentence_initial(text, token.start) and not _in_name_slot(text, token.start)
            if not (in_running_text and known_by(token.word, without_corpus)):
                unknown += 1
    return unknown


def _with_texts(record: dict, texts: list[str]) -> dict:
    messages = [{**turn, "content": text} for turn, text in zip(record["messages"], texts, strict=True)]
    return {**record, "messages": messages}


def strict_gate(
    records: list[dict],
    fake_names: Iterable[str],
    lexicon: Lexicon = KNOWN_WORDS,
    disallowed: Mapping[str, Collection[str]] | None = None,
) -> GateResult:
    """Records with nothing unresolved and unknown names replaced, plus counts.

    `fake_names` must be every fake the anonymiser wrote (participants' fakes and @mention fakes):
    only those and currency words ("Ar") pass as they are; any other pool name is replaced.
    `disallowed` maps a record id to pool names its raw text already held: those are real there, so
    they are replaced (a participant's fake that equals one is replaced too, the same way). A
    conversation dropped for two reasons counts under both. `cleared_*` counts capitalised words in
    every conversation before the gate; `tokens_replaced` counts words replaced in kept conversations,
    `words_kept` all their words, `fakes_per_kept_median`/`_max` replaced words per kept conversation.
    """
    fake_keys = fake_name_keys([*fake_names, *CURRENCY_WORDS])
    disallowed = disallowed or {}
    kept, dropped, cleared, replaced, words = [], dict.fromkeys(GATE_REASONS, 0), Counter(), [], 0
    for record in records:
        texts = [normalise(turn["content"]) for turn in record["messages"]]
        real_pool = {compare_key(name) for name in disallowed.get(record["id"], ())}
        names = _clear_or_replace(texts, fake_keys - real_pool, lexicon, real_pool)
        cleared += names.cleared
        reasons = _reasons(texts)
        if names.replaced > MAX_REPLACED_SHARE * names.words:
            reasons.add("too_many_unknown")
        if names.replaced > MAX_FAKES_PER_CONVERSATION:
            reasons.add("too_many_fakes")
        for reason in reasons:
            dropped[reason] += 1
        if not reasons:
            kept.append(_with_texts(record, names.texts))
            replaced.append(names.replaced)
            words += names.words
    counts = {
        "conversations_before_gate": len(records),
        "dropped": len(records) - len(kept),
        **{f"dropped_{reason}": n for reason, n in dropped.items()},
        "kept": len(kept),
        **{f"cleared_{source}": cleared[source] for source in lexicon},
        "tokens_replaced": sum(replaced),
        "words_kept": words,
        "fakes_per_kept_median": median_low(replaced) if replaced else 0,
        "fakes_per_kept_max": max(replaced, default=0),
    }
    return GateResult(kept, counts)


def _reasons(texts: list[str]) -> set[str]:
    reasons = set()
    for text in texts:
        if has_long_number(text):
            reasons.add("long_number")
        if contact_left(text):
            reasons.add("contact")
        if has_hidden_char(text):
            reasons.add("hidden_char")
    return reasons
