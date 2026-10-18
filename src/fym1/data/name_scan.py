"""Name and chat-text scan shared by the release snapshot and the synth sample gate.

Finds the raw export's sender names (as typed, glued to markers or endings, hidden behind escapes, invisible
characters or look-alike digits) and CHAT_TEXT_NGRAM-word runs of the export's message texts in any text. Everything
here returns a count, a bool or an index; nothing returns a sender name or a message text.
"""

import html
import re
import unicodedata
import urllib.parse
from collections.abc import Iterable
from pathlib import Path

from fym1.data.anonymize import MIN_NAME_PREFIX, NAME_SUFFIXES, compare_key, fold, is_everyday_word, normalise
from fym1.data.gate import PERSON_MARKER_WORDS, is_hidden_char
from fym1.data.gate import tokens as gate_tokens
from fym1.data.known_words import KNOWN_WORDS, known_by
from fym1.data.messenger import Thread, load_threads
from fym1.eval.metrics import NgramIndex

MIN_NAME_TOKEN = 3  # shorter tokens ("ny", initials) are everyday words or letters; counted as skipped
# Meta's label for a deleted or unknown account: not a person's name.
PLACEHOLDER_SENDERS = frozenset({"facebook user", "utilisateur de facebook", "instagram user", "meta user"})
# Word-run length that counts as copied chat text: the same n as the training-data leak gate (fym1.eval.metrics).
CHAT_TEXT_NGRAM = 8
_LETTERS = re.compile(r"[^\W\d_]+")
_LETTERS_AND_DIGITS = re.compile(r"[^\W_]+")
# Digits typed for the letters they look like ("H4nt4", "Rak0to"): the name scan also reads them as letters.
DIGIT_LETTERS = str.maketrans("140357", "iaoest")
_ESCAPES = re.compile(r"\\(?:x([0-9a-fA-F]{2})|u([0-9a-fA-F]{4})|U([0-9a-fA-F]{8})|[nrt\"'\\])")
MAX_UNESCAPE_ROUNDS = 5


class RefusedError(Exception):
    """An input cannot be read or checked safely; the caller stops and writes nothing."""


def readings(text: str) -> tuple[str, str]:
    """`text` with escapes decoded (`_unescape`) and normalised as the anonymiser does, read two ways: every
    invisible or combining character the gate flags taken as a word break ("Bodo\u200bzafy" -> "Bodo zafy"), and
    taken out ("Za\u0903fy" -> "Zafy"). A name found in either reading counts."""
    text = _unescape(text)
    split = normalise("".join(" " if is_hidden_char(c) else c for c in unicodedata.normalize("NFC", text)))
    joined = "".join(c for c in normalise(text) if not is_hidden_char(c))
    return split, joined


def _forms(word: str) -> list[str]:
    """Letter runs of one gate token; a token with digits inside ("Rak0to") is also read with DIGIT_LETTERS."""
    forms = _LETTERS.findall(word)
    if any(c.isdigit() for c in word):
        forms += _LETTERS.findall(word.translate(DIGIT_LETTERS))
    return forms


def _reading_words(reading: str) -> list[str]:
    """Words of one reading, split by the strict gate's own tokeniser (`fym1.data.gate.tokens`)."""
    return [form for token in gate_tokens(reading) for form in _forms(token.word)]


def words(text: str) -> list[str]:
    """Compare keys (folded, stretched letters collapsed) of the gate's words in both `readings` of `text`."""
    return [compare_key(word) for reading in dict.fromkeys(readings(text)) for word in _reading_words(reading)]


def is_name_word(word: str, names: frozenset[str], allowed: frozenset[str] = frozenset()) -> bool:
    """`word` (as typed) is a name, alone or with a glued person marker ("izafy", "n'izafy" reads "izafy", "ryzafy")
    and/or a glued possessive ending ("Zafyko"), the anonymiser's suffix list, or holds a 4+ letter name: at its start
    ("zafybodo") or, when no known-word list has the word and its compare key is not in `allowed` (everyday words
    the caller spares for this text), anywhere inside ("xzafyx"). Like the anonymiser, only an exact name match counts
    for an everyday word. A word under MIN_NAME_TOKEN letters is never a name, so a short name's collapsed key
    ("Ann" -> "an") does not match the everyday word "an"."""
    if len(fold(word)) < MIN_NAME_TOKEN:
        return False
    key = compare_key(word)
    if key in names:
        return True
    if is_everyday_word(key):  # checked first, as the anonymiser does: "farany" is not "Fara" + "-ny"
        return False
    stems = {key} | {key[len(m):] for m in PERSON_MARKER_WORDS if key.startswith(m)}
    stems |= {stem[: -len(s)] for stem in stems for s in NAME_SUFFIXES if stem.endswith(s)}
    if not stems.isdisjoint(names):
        return True
    if any(stem.startswith(n) for stem in stems for n in names if len(n) >= MIN_NAME_PREFIX):
        return True
    inner = [n for n in names if len(n) >= MIN_NAME_PREFIX and n in key]
    return bool(inner) and key not in allowed and known_by(word, KNOWN_WORDS) is None


def name_tokens(senders: Iterable[str]) -> tuple[frozenset[str], int, int]:
    """Name tokens of `senders` (compare keys of the gate's words in both readings), plus how many short tokens
    (under MIN_NAME_TOKEN letters as typed, before stretched letters collapse) and placeholder senders were skipped.
    """
    tokens, short, placeholders = set(), set(), 0
    placeholder_keys = {" ".join(words(label)) for label in PLACEHOLDER_SENDERS}
    for sender in set(senders):
        if " ".join(words(sender)) in placeholder_keys:
            placeholders += 1
            continue
        for reading in readings(sender):
            for word in _reading_words(reading):
                (tokens if len(fold(word)) >= MIN_NAME_TOKEN else short).add(compare_key(word))
    return frozenset(tokens), len(short - tokens), placeholders


def read_export(export: Path) -> list[Thread]:
    """The export's conversations. Errors never carry the export's paths or text."""
    try:
        threads = load_threads(export)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise RefusedError(f"export could not be read ({type(error).__name__})") from None
    if not threads:
        raise RefusedError("export holds no conversations")
    return threads


def export_senders(threads: list[Thread]) -> list[str]:
    """Every sender and listed participant in the export."""
    return [p for thread in threads for p in [*thread.participants, *(m.sender for m in thread.messages)]]


def export_texts(threads: list[Thread]) -> list[str]:
    """Every non-empty message text in the export."""
    return [m.text for thread in threads for m in thread.messages if m.text]


def chat_text_index(texts: list[str], source: str) -> NgramIndex:
    """CHAT_TEXT_NGRAM-word runs of `texts`, as written and as the anonymiser `normalise`s them. Refused when no
    text from `source` is that long: an empty index would pass every file unchecked."""
    try:
        return NgramIndex([*texts, *map(normalise, texts)], n=CHAT_TEXT_NGRAM)
    except ValueError:
        raise RefusedError(f"{source} has no {CHAT_TEXT_NGRAM}-word message to check copied chat text against") \
            from None


def has_chat_text(text: str, indexes: Iterable[NgramIndex]) -> bool:
    """`text` shares a word run with any of `indexes`, read as written and in both name-scan `readings`
    (escapes decoded, hidden characters as a break and taken out)."""
    forms = (text, *readings(text))
    return any(index.shares_with(form) for index in indexes for form in forms)


def count_name_hits(text: str, names: frozenset[str], allowed: frozenset[str] = frozenset()) -> int:
    """Case/accent-insensitive name matches (`is_name_word`, with `allowed` compare keys spared the inside-a-word
    rule) in the gate's words of `text`, in whichever of its two `readings` finds more. Escapes are decoded first; a
    run of letters and digits with no name in its gate words is read again with DIGIT_LETTERS ("B0d0", "Z4fy2")."""
    return max(_reading_hits(reading, names, allowed) for reading in readings(text))


def _token_hits(text: str, names: frozenset[str], allowed: frozenset[str]) -> list[tuple[int, int]]:
    """Spans of the gate tokens in `text` that hold a name."""
    return [(t.start, t.end) for t in gate_tokens(text)
            if any(is_name_word(f, names, allowed) for f in _forms(t.word))]


def _reading_hits(reading: str, names: frozenset[str], allowed: frozenset[str]) -> int:
    spans = _token_hits(reading, names, allowed)
    hits = len(spans)
    for run in _LETTERS_AND_DIGITS.finditer(reading):
        word = run.group()
        if word.isalpha() or not any(c.isalpha() for c in word):
            continue
        if not any(start < run.end() and end > run.start() for start, end in spans):
            hits += len(_token_hits(word.translate(DIGIT_LETTERS), names, allowed))
    return hits


def _unescape(text: str) -> str:
    """`text` with escapes read as the characters they stand for, repeated until nothing changes (escapes inside
    escapes): JSON/Python `\\n`, `\\xXX`, `\\uXXXX`, `\\UXXXXXXXX`, HTML entities (`&#90;`, `&amp;`) and URL
    percent escapes (`%5A`)."""
    for _ in range(MAX_UNESCAPE_ROUNDS):
        decoded = _ESCAPES.sub(_escape_char, text)
        decoded = html.unescape(decoded)
        decoded = urllib.parse.unquote(decoded, errors="replace")
        if decoded == text:
            break
        text = decoded
    return text


def _escape_char(match: re.Match) -> str:
    code = next((g for g in match.groups() if g), None)
    if code is None:  # \n \r \t \" \' \\: a break between words
        return " "
    try:
        return chr(int(code, 16))
    except (ValueError, OverflowError):  # past U+10FFFF: not a character
        return " "
