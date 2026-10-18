"""Name and chat-text scan shared by the release snapshot and the synth sample gate.

Finds the raw export's sender names (as typed, glued to markers or endings, hidden behind escapes, invisible
characters or look-alike digits) and copied chat text in any text: CHAT_TEXT_NGRAM-word runs of a conversation's
messages (runs may cross from one message into the next) and whole short messages with SHORT_MESSAGE_DISTINCTIVE+
distinctive words, words compared as the name scan compares them (`ChatIndex`). The export is read from every
folder of thread pages and every one-file export at any depth; a file no reader consumes refuses it. Everything
here returns a count, a bool or an index; nothing returns a sender name or a message text.
"""

import fnmatch
import html
import re
import unicodedata
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from fym1.data.anonymize import MIN_NAME_PREFIX, NAME_SUFFIXES, compare_key, fold, is_everyday_word, normalise
from fym1.data.gate import PERSON_MARKER_WORDS, is_hidden_char
from fym1.data.gate import tokens as gate_tokens
from fym1.data.known_words import KNOWN_WORDS, known_by
from fym1.data.messenger import Thread, load_dyi_threads, load_flat_threads

MIN_NAME_TOKEN = 3  # shorter tokens ("ny", initials) are everyday words or letters; counted as skipped
# Meta's label for a deleted or unknown account: not a person's name.
PLACEHOLDER_SENDERS = frozenset({"facebook user", "utilisateur de facebook", "instagram user", "meta user"})
# Word-run length that counts as copied chat text: the same n as the training-data leak gate (fym1.eval.metrics).
CHAT_TEXT_NGRAM = 8
# A shorter message is copied chat text when a shipped text holds all its words in a row and it has this many
# different distinctive words (`is_distinctive`): "ok merci tsara" is everyday, four rare words are a fingerprint.
SHORT_MESSAGE_DISTINCTIVE = 4
# The known-word lists whose words are common enough to say nothing about a message's author.
COMMON_LISTS = ("mg_common", "fr", "en")
# Typographic apostrophes, quotes and hyphens, read as their ASCII forms by the chat-text check.
ASCII_PUNCTUATION = str.maketrans({**dict.fromkeys("\u2018\u2019\u201a\u201b\u2032\u02bc\u02b9\u0060\u00b4", "'"),
                                   **dict.fromkeys("\u201c\u201d\u201e\u201f\u2033\u00ab\u00bb", '"'),
                                   **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe63\uff0d", "-")})
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


# Every file under --export must be read by one of these, or the export is refused (review finding 14): a file no
# reader consumes ("message_1.html", a one-file export renamed or nested, a hidden ".DS_Store") would hold names
# nobody checks.
DYI_PAGE = "message_*.json"  # a "Download your information" thread page, in any folder at any depth
FLAT_SUFFIX = ".json"  # any other JSON file, at any depth: must be a one-file export


@dataclass(frozen=True)
class ExportFiles:
    pages: list[Path]  # DYI thread pages
    flat: list[Path]  # one-file exports


def export_files(export: Path) -> ExportFiles:
    """Every file under `export` (or `export` itself), sorted by reader. Any file no reader consumes refuses the
    export, counted, never named."""
    # A symlinked folder is not walked into, so it counts as a file no reader consumes.
    if export.is_file():
        files = [export]
    else:
        files = sorted(p for p in export.rglob("*") if p.is_symlink() or not p.is_dir())
    pages = [p for p in files if fnmatch.fnmatch(p.name, DYI_PAGE)]
    flat = [p for p in files if p.suffix.lower() == FLAT_SUFFIX and p not in pages]
    unread = len(files) - len(pages) - len(flat)
    if unread:
        raise RefusedError(f"export holds {unread} files no reader consumes (want only thread pages "
                           f"`{DYI_PAGE}` and one-file `*{FLAT_SUFFIX}` exports; hidden files such as .DS_Store "
                           f"count too)")
    return ExportFiles(pages, flat)


# Errors a malformed export raises while it is read; RecursionError is JSON nested too deep to parse.
_READ_ERRORS = (ValueError, OSError, KeyError, TypeError, IndexError, AttributeError, RecursionError)


def read_export(export: Path, found: ExportFiles | None = None) -> list[Thread]:
    """The export's conversations, from every folder of thread pages (message requests and filtered threads too)
    and every one-file export at any depth (`found`: `export_files(export)`, read here when not given). Errors never
    carry the export's paths or text."""
    found = found or export_files(export)
    try:
        threads = [*(load_dyi_threads(export, folders=None) if found.pages else []),
                   *(t for path in found.flat for t in load_flat_threads(path))]
    except _READ_ERRORS as error:
        raise RefusedError(f"export could not be read ({type(error).__name__})") from None
    if not threads:
        raise RefusedError("export holds no conversations")
    return threads


def export_senders(threads: list[Thread]) -> list[str]:
    """Every sender and listed participant in the export."""
    return [p for thread in threads for p in [*thread.participants, *(m.sender for m in thread.messages)]]


def export_conversations(threads: list[Thread]) -> list[list[str]]:
    """Each thread's non-empty message texts, in order."""
    return [texts for thread in threads if (texts := [m.text for m in thread.messages if m.text])]


def chat_words(text: str) -> list[str]:
    """The words the chat-text check compares, keyed as the name scan keys a word: typographic quotes and hyphens
    read as ASCII, split into runs of letters and digits, digits typed for letters read as letters (DIGIT_LETTERS),
    then `compare_key` (case and accents folded, stretched letters collapsed). "Amin’ny", "AMIN'NY" and "4min'nyyy"
    all read "amin ny"."""
    return [compare_key(word.translate(DIGIT_LETTERS))
            for word in _LETTERS_AND_DIGITS.findall(fold(text.translate(ASCII_PUNCTUATION)))]


def _chat_forms(text: str) -> tuple[str, ...]:
    """`text` as written and in both name-scan `readings` (escapes decoded, hidden characters as a break and taken
    out)."""
    return (text, *readings(text))


def is_distinctive(word: str) -> bool:
    """A `chat_words` word that says something about who typed it: letters only, MIN_NAME_TOKEN+ long, and not an
    everyday Malagasy word nor one of wordfreq's most frequent French or English words."""
    return (word.isalpha() and len(word) >= MIN_NAME_TOKEN and not is_everyday_word(word)
            and not any(KNOWN_WORDS[common].has(word) for common in COMMON_LISTS))


class ChatIndex:
    """Copied chat text in one source (lead decision 34, review findings 17/18). Each conversation's messages are
    joined in order, so a CHAT_TEXT_NGRAM-word run that crosses from one message into the next is indexed too. A
    message shorter than that with SHORT_MESSAGE_DISTINCTIVE+ different `is_distinctive` words is indexed whole: a
    shipped text holding all its words in a row matches. Every text is read in its `_chat_forms`, as `chat_words`."""

    def __init__(self, conversations: Iterable[list[str]], source: str):
        self.runs: set[tuple[str, ...]] = set()
        self.short: dict[int, set[tuple[str, ...]]] = {}
        for texts in conversations:
            joined: dict[int, list[str]] = {}  # each of the `_chat_forms`, by position, joined across messages
            for text in texts:
                for position, form in enumerate(_chat_forms(text)):
                    message = chat_words(form)
                    joined.setdefault(position, []).extend(message)
                    if len(message) < CHAT_TEXT_NGRAM and len({w for w in message if is_distinctive(w)}) \
                            >= SHORT_MESSAGE_DISTINCTIVE:
                        self.short.setdefault(len(message), set()).add(tuple(message))
            for words in joined.values():
                self.runs.update(_runs(words, CHAT_TEXT_NGRAM))
        if not self.runs:  # an empty index would pass every file unchecked
            raise RefusedError(f"{source} has no {CHAT_TEXT_NGRAM}-word run of messages to check copied chat text "
                               f"against")

    @property
    def short_messages(self) -> int:
        return sum(map(len, self.short.values()))

    def shares_with(self, text: str) -> bool:
        """`text`, in any of its `_chat_forms`, holds an indexed run or a whole indexed short message."""
        for form in _chat_forms(text):
            words = chat_words(form)
            if not self.runs.isdisjoint(_runs(words, CHAT_TEXT_NGRAM)):
                return True
            if any(not messages.isdisjoint(_runs(words, n)) for n, messages in self.short.items()):
                return True
        return False


def _runs(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def has_chat_text(text: str, indexes: Iterable[ChatIndex]) -> bool:
    """`text` holds copied chat text from any of `indexes`."""
    return any(index.shares_with(text) for index in indexes)



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
