"""Public release snapshot: tracked files of one clean commit, minus private paths, checked for names and secrets.

The public repo is a fresh single-commit repo built from this snapshot, so private git history never ships.
Release blocker (lead decision 21): the export's sender names are read at run time, removed from the hand-written
name lists, and searched for in every file except the lists a script rebuilds from public data alone. The lists
scripts/build_known_words.py builds from public data minus the hand lists do not ship (REBUILT_LISTS). Release
blocker (lead decision 34): any shipped file or path sharing a CHAT_TEXT_NGRAM-word run with the export's messages
or the anonymised chats' turns (fake names, joined messages; runs may cross from one message into the next), or
holding a whole short message with distinctive words, fails the release; words are compared as the name scan
compares them. The release ships text files only (lead decision 36). Everything this module
reports is a count or a repo path; it never returns a sender name, a message text, a private id or a matched
secret value.
"""

import fnmatch
import hashlib
import html
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import unicodedata
import urllib.parse
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from fym1.data.anonymize import MIN_NAME_PREFIX, NAME_SUFFIXES, compare_key, fold, is_everyday_word, normalise
from fym1.data.gate import PERSON_MARKER_WORDS, is_hidden_char
from fym1.data.gate import tokens as gate_tokens
from fym1.data.known_words import KNOWN_WORDS, known_by
from fym1.data.messenger import Thread, fix_meta_text, load_dyi_threads, load_flat_threads


@dataclass(frozen=True)
class Exclude:
    path: str  # exact repo path, or a folder prefix ending in "/"
    reason: str


# Lists scripts/build_known_words.py writes. It builds them from public data MINUS the hand lists' names, and the
# repo's copies were built from the unstripped hand lists: a rebuild from the shipped (stripped) hand lists and a diff
# would show exactly the names the snapshot removed. So they never ship; anyone rebuilds them from the shipped hand
# lists with REBUILD_COMMAND (large downloads: Wikipedia, INSEE, HPLT).
REBUILD_COMMAND = "uv sync --group data && uv run python scripts/build_known_words.py"
_NOT_SHIPPED = f"built from the unstripped hand lists (a rebuild + diff would show the removed names); rebuild with " \
               f"`{REBUILD_COMMAND}`"
REBUILT_LISTS: tuple[str, ...] = tuple(f"src/fym1/data/{name}" for name in (
    "en_words.txt", "fr_words.txt", "first_names.txt", "wordlike_first_names.txt", "mg_wiki_words.txt",
    "mg_corpus_words.txt",
))

# The release tooling itself: private build tooling, and its exact (file, word) allow-lists are a private build
# record (lead decision 32). Nothing that ships imports it (tests/test_release.py checks this).
RELEASE_TOOLING: tuple[str, ...] = ("src/fym1/release.py", "scripts/release_snapshot.py", "tests/test_release.py")
_RELEASE_TOOLING = "private release tooling: builds and checks this snapshot; its allow-lists are a private build record"

EXCLUDES: tuple[Exclude, ...] = (
    Exclude("AGENTS.md", "agent-team rules for the private build; points at the owner's private notes"),
    Exclude("CLAUDE.md", "same agent-team rules as AGENTS.md, for one tool"),
    Exclude("docs/project-update.md", "internal status and team pickup notes; points at the private operator notes "
                                      "that name accounts and machines"),
    Exclude("docs/colab_operator.md", "operator-only: the owner's personal Google account rules and the HF token "
                                      "hand-off steps for Playwright"),
    Exclude("docs/kaggle_operator.md", "operator-only: the owner's Kaggle account setup and run steps against "
                                       "private repos"),
    Exclude("docs/superpowers/", "internal design spec and build plan: owner background, accounts and team process; "
                                 "the public account is README, MODEL_CARD, paper and docs/anonymisation.md"),
    Exclude("scripts/serve_hf_token_once.py", "operator-only: hands the local HF token to the browser agent"),
    Exclude("tests/test_serve_hf_token_once.py", "tests the excluded scripts/serve_hf_token_once.py"),
    Exclude("paper/fym1.pdf", "built from paper/fym1.tex (CI / arXiv); the release ships text files only, so every "
                              "byte it ships is read (lead decision 36)"),
    Exclude("results/release_check.json", "this tool's own counts report: a private build record of what the "
                                          "snapshot checked"),
    *(Exclude(path, _RELEASE_TOOLING) for path in RELEASE_TOOLING),
    *(Exclude(path, _NOT_SHIPPED) for path in REBUILT_LISTS),
)

# Word/name lists, shipped under their own licences (see NOTICE).
LIST_GLOBS: tuple[str, ...] = ("src/fym1/data/*.txt",)
# Hand-edited lists: the export's sender names are removed from these. Nobody can rebuild them, so a removal
# leaves no trace.
HAND_LISTS: dict[str, str] = {
    "src/fym1/data/mg_first_names.txt": "hand-edited Malagasy given names",
    "src/fym1/data/mg_known_names.txt": "hand-edited places, brands and other non-person words",
    "src/fym1/data/name_deny_list.txt": "hand-edited Malagasy given names that are also everyday words",
}
# Shipped lists a script writes from public data alone (no hand list goes in). They are shipped exactly as built
# and the name scan skips them: anyone can rebuild one and diff it, so a removed entry would point at a sender, and
# a common word in one reveals nothing about the export. Secrets and private ids are still checked in them.
GENERATED_LISTS: dict[str, str] = {
    "src/fym1/data/mg_common_words.txt": "reproducible from public data by scripts/build_common_words.py "
                                         "(Malagasy Wikipedia)",
}
DATA_LICENCES: dict[str, str] = {
    "src/fym1/data/mg_common_words.txt": "derived from Malagasy Wikipedia (wikimedia/wikipedia 20231101.mg); "
                                         "CC BY-SA 4.0",
    "src/fym1/data/mg_first_names.txt": "hand list, this repo (Apache-2.0)",
    "src/fym1/data/mg_known_names.txt": "hand list, this repo (Apache-2.0)",
    "src/fym1/data/name_deny_list.txt": "hand list, this repo (Apache-2.0)",
}
NOTICE_NAME = "NOTICE"

# The release ships text files only (lead decision 36): these types (or names), decoded in full, with no control
# character but tab, LF and CR. Any other file is refused before anything is written: a binary's bytes (images, a
# PDF's streams, data hidden after its end) are not read, so nothing in them is checked.
TEXT_SUFFIXES = frozenset({".py", ".md", ".txt", ".json", ".jsonl", ".ipynb", ".toml", ".yml", ".yaml", ".lock",
                           ".jinja", ".tex", ".bib", ".cfg", ".ini", ".csv", ".tsv", ".sh"})
TEXT_NAMES = frozenset({"LICENSE", "NOTICE", ".gitignore", ".gitkeep", ".gitattributes", ".python-version"})
BOMS = ((b"\xef\xbb\xbf", "utf-8"), (b"\xff\xfe\x00\x00", "utf-32-le"), (b"\x00\x00\xfe\xff", "utf-32-be"),
        (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"))
# The public tree holds no archives, so the release refuses any file with one of these suffixes before anything is
# written (lead call on review findings 10/11): an archive's member names, comments and gaps are not read.
ARCHIVE_SUFFIXES = frozenset({".zip", ".gz", ".tar", ".tgz", ".7z", ".rar", ".whl", ".jar", ".bz2", ".xz", ".zst",
                              ".lz", ".lzma", ".tbz2", ".txz", ".egg", ".war"})
# A renamed archive ("notes.md" that is really a zip) is caught by its bytes, not its name (review findings 10/11).
ARCHIVE_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"7z\xbc\xaf\x27\x1c",
                 b"Rar!\x1a\x07", b"\x28\xb5\x2f\xfd")
TAR_MAGIC, TAR_MAGIC_OFFSET = b"ustar", 257
# Zip readers find the archive from its end, so a zip appended to a PDF or a text file still opens as a zip.
ZIP_MARKERS = (b"PK\x03\x04", b"PK\x05\x06")

# The owner chose to publish the paper and the citations under their own name (lead decision 31), and the owner is a
# sender in the export. So exactly that author string, two words, one space, as written, passes the name scan, and
# only in these files; any other form, any other file, or either word alone is still a hit. Kept as a hash so this
# file does not hold the name.
AUTHOR_SHA256 = "44c9ff23a0c635faa8d70a28269daac1dcffa04e70ee5156c06ef7cbf3e453de"
AUTHOR_FILES = frozenset({"README.md", "MODEL_CARD.md", "paper/fym1.tex", "tests/test_public_name.py"})
_WORD_START = re.compile(r"(?<![^\W\d_])[^\W\d_]")
_TWO_WORDS = re.compile(r"(?<![^\W\d_])[^\W\d_]+ [^\W\d_]+(?![^\W\d_])")

MIN_NAME_TOKEN = 3  # shorter tokens ("ny", initials) are everyday words or letters; counted as skipped
# A MIN_NAME_PREFIX+ letter name is a hit anywhere inside a word no known-word list has ("xbodozafyx"). The only
# exceptions are these exact (file, word) pairs of everyday words the inner rule would otherwise flag, each with its
# reason. A pair covers that word (as compare_key reads it) in that file's text only, never in a path or another
# file. This module does not ship (RELEASE_TOOLING), so no entry here names a file of the release tooling.
REASON_RANDOM = "the everyday English word 'random' (random seeds, random.Random); no person is named"
INNER_NAME_ALLOWED: dict[tuple[str, str], str] = {
    ("docs/anonymisation.md", "random"): REASON_RANDOM,
    ("notebooks/00_baseline.ipynb", "random"): REASON_RANDOM,
    ("notebooks/01_train.ipynb", "random"): REASON_RANDOM,
    ("notebooks/03_trim.ipynb", "random"): REASON_RANDOM,
    ("paper/fym1.tex", "random"): REASON_RANDOM,
    ("scripts/build_formal.py", "random"): REASON_RANDOM,
    ("scripts/build_notebooks.py", "random"): REASON_RANDOM,
    ("scripts/build_pairs.py", "random"): REASON_RANDOM,
    ("spaces/arena/README.md", "random"): REASON_RANDOM,
    ("spaces/arena/app.py", "random"): REASON_RANDOM,
    ("spaces/arena/votes.py", "random"): REASON_RANDOM,
    ("src/fym1/data/formal.py", "random"): REASON_RANDOM,
    ("src/fym1/data/gate.py", "random"): REASON_RANDOM,
    ("src/fym1/data/review_sheet.py", "random"): REASON_RANDOM,
    ("src/fym1/data/synth.py", "random"): REASON_RANDOM,
    ("src/fym1/eval/arena.py", "random"): REASON_RANDOM,
    ("src/fym1/eval/scoring.py", "random"): REASON_RANDOM,
    ("tests/data/test_anonymize_dates.py", "random"): REASON_RANDOM,
    ("tests/data/test_anonymize_marks.py", "random"): REASON_RANDOM,
    ("tests/data/test_formal.py", "random"): REASON_RANDOM,
    ("tests/eval/test_arena.py", "random"): REASON_RANDOM,
    ("tests/spaces/test_votes.py", "random"): REASON_RANDOM,
}
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
PRIVATE_HF_ID = "Fan1000-Prog/" + "fyM1-"  # split so test_public_name does not see a Hub id

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "hf_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "kaggle_key_json": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "kaggle_api_token": re.compile(r"KGAT_[A-Za-z0-9]{16,}"),
    "private_key": re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),
    "sk_api_key": re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    "slack_token": re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"),
    "google_oauth_secret": re.compile(r"\bGOCSPX-[A-Za-z0-9_\-]{20,}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    "telegram_bot_token": re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{35}\b"),
    "assigned_secret": re.compile(
        r"""(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b["']?\s*[:=]\s*["'][A-Za-z0-9_\-+/=.]{16,}["']"""
    ),
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
# An address on a reserved test domain (RFC 2606/6761) is never a person's, anywhere in the snapshot.
RESERVED_EMAIL_TLDS = (".example", ".invalid", ".test")
# Every other address must be listed here, exactly, for the one file that holds it, with its reason. Any other
# address in any file is a secret hit. Stored as (local part, domain), case-folded, so this file holds no address.
EMAIL_PROBE_FILES: dict[str, frozenset[tuple[str, str]]] = {
    # "x@y.mg": a placeholder in a comment showing the shape of one non-space run; one-letter host, no person.
    "scripts/build_known_words.py": frozenset({("x", "y.mg")}),
    # the same placeholder as a synthetic chat line that the synth filter must drop.
    "tests/data/test_synth.py": frozenset({("x", "y.mg")}),
    # synthetic address in a synthetic chat line the anonymiser must mask; host "example.mg".
    "tests/data/test_anonymize.py": frozenset({("hanta.r", "example.mg")}),
    # synthetic address the kept-checks leak scan must flag; host "example.mg".
    "tests/data/test_kept_checks.py": frozenset({("rabe.sample", "example.mg")}),
    # "olona" is the Malagasy word for "person": a synthetic address the gate must refuse; host "example.mg".
    "tests/data/test_gate.py": frozenset({("olona", "example.mg")}),
    # synthetic address in a link-and-handle line that known_words must not read as words.
    "tests/data/test_known_words.py": frozenset({("fanja", "mail.mg")}),
}
SECRET_FILE_NAMES = frozenset({".env", "kaggle.json", "access_token", "id_rsa", "id_ed25519", ".netrc"})
SECRET_FILE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")

_LETTERS = re.compile(r"[^\W\d_]+")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")
# Text a reader decodes before it can be read (review finding 14 nits): a base64 `data:` URI hides bytes the scan
# never sees, and Unicode tag characters (U+E0000-E007F) spell invisible ASCII. A text file holding either is refused.
_DATA_URI = re.compile(r"(?i)\bdata:[^,\s]*;\s*base64\s*,")
_TAG_CHARS = re.compile("[\U000E0000-\U000E007F]")
_LETTERS_AND_DIGITS = re.compile(r"[^\W_]+")
# Digits typed for the letters they look like ("H4nt4", "Rak0to"): the name scan also reads them as letters.
DIGIT_LETTERS = str.maketrans("140357", "iaoest")
_ESCAPES = re.compile(r"\\(?:x([0-9a-fA-F]{2})|u([0-9a-fA-F]{4})|U([0-9a-fA-F]{8})|[nrt\"'\\])")
MAX_UNESCAPE_ROUNDS = 5


class RefusedError(Exception):
    """The snapshot cannot be made safely; nothing was copied."""


@dataclass
class Report:
    ref: str
    files_copied: int = 0
    excluded: list[dict] = field(default_factory=list)
    excludes_unmatched: list[str] = field(default_factory=list)
    name_tokens: int = 0
    short_tokens_skipped: int = 0
    placeholder_senders_skipped: int = 0
    author_strings_allowed: int = 0
    inner_name_words_allowed: int = 0
    list_entries_removed: dict[str, int] = field(default_factory=dict)
    name_hits: dict[str, int] = field(default_factory=dict)
    private_ids: int = 0
    private_id_hits: dict[str, int] = field(default_factory=dict)
    chat_messages: int = 0
    chat_turns: int = 0
    chat_short_messages: int = 0
    chat_text_hits: list[str] = field(default_factory=list)
    secret_hits: dict[str, dict[str, int]] = field(default_factory=dict)
    emails_allowed: int = 0
    private_hf_ids: dict[str, int] = field(default_factory=dict)
    references_to_excluded: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.name_hits or self.private_id_hits or self.secret_hits or self.chat_text_hits)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "ref": self.ref,
            "files_copied": self.files_copied,
            "excluded": self.excluded,
            "excludes_unmatched": self.excludes_unmatched,
            "names": {
                "name_tokens": self.name_tokens,
                "short_tokens_skipped": self.short_tokens_skipped,
                "placeholder_senders_skipped": self.placeholder_senders_skipped,
                "author_strings_allowed": self.author_strings_allowed,
                "inner_name_words_allowed": self.inner_name_words_allowed,
                "list_entries_removed": self.list_entries_removed,
                "list_entries_removed_total": sum(self.list_entries_removed.values()),
                "hits_per_file": self.name_hits,
                "hits_total": sum(self.name_hits.values()),
            },
            "private_ids": {
                "ids_checked": self.private_ids,
                "hits_per_file": self.private_id_hits,
                "hits_total": sum(self.private_id_hits.values()),
            },
            "chat_text": {
                "messages_indexed": self.chat_messages,
                "anonymised_turns_indexed": self.chat_turns,
                "short_messages_indexed": self.chat_short_messages,
                "short_message_distinctive_words": SHORT_MESSAGE_DISTINCTIVE,
                "ngram_words": CHAT_TEXT_NGRAM,
                "files_with_hits": self.chat_text_hits,
                "files_total": len(self.chat_text_hits),
            },
            "secrets": {
                "hits_per_file": self.secret_hits,
                "hits_total": sum(sum(c.values()) for c in self.secret_hits.values()),
                "emails_allowed": self.emails_allowed,
            },
            "private_hf_ids_per_file": self.private_hf_ids,
            "references_to_excluded": self.references_to_excluded,
        }


# ---- names ----------------------------------------------------------------------------------------------------


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
    ("zafybodo") or, when no known-word list has the word and its compare key is not in `allowed` (the file's
    INNER_NAME_ALLOWED words), anywhere inside ("xzafyx"). Like the anonymiser, only an exact name match counts for
    an everyday word. A word under MIN_NAME_TOKEN letters is never a name, so a short name's collapsed key
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
# reader consumes ("message_1.html", a one-file export renamed or nested) would hold names nobody checks.
DYI_PAGE = "message_*.json"  # a "Download your information" thread page, in any folder at any depth
FLAT_SUFFIX = ".json"  # any other JSON file, at any depth: must be a one-file export
# Keys whose text value is a person's name, anywhere in a thread page (participants, senders, reaction actors, the
# users of an add/remove-member event, ...), plus any key ending in NAME_KEY_SUFFIX.
NAME_KEYS = frozenset({"name", "sender_name", "actor", "sender"})
NAME_KEY_SUFFIX = "_name"


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
                           f"`{DYI_PAGE}` and one-file `*{FLAT_SUFFIX}` exports)")
    return ExportFiles(pages, flat)


def read_export(export: Path) -> list[Thread]:
    """The export's conversations, from every folder of thread pages (message requests and filtered threads too)
    and every one-file export at any depth. Errors never carry the export's paths or text."""
    found = export_files(export)
    try:
        threads = [*(load_dyi_threads(export, folders=None) if found.pages else []),
                   *(t for path in found.flat for t in load_flat_threads(path))]
    except (ValueError, OSError, KeyError, TypeError, IndexError, AttributeError) as error:
        raise RefusedError(f"export could not be read ({type(error).__name__})") from None
    if not threads:
        raise RefusedError("export holds no conversations")
    return threads


def export_senders(export: Path) -> list[str]:
    """Every person name the export lists, read from its raw files rather than the loaded conversations (which drop
    unsent, call and event messages): in every thread page (`message_*.json`) anywhere under `export`, whatever its
    folder (inbox, archived_threads, e2ee_cutover, message_requests, filtered_threads, ...), every `_page_names`
    value; in every one-file export, the listed participants and every sender; in both, the distinctive words of the
    conversation title (`title_words`). A page of any other shape is refused. Errors never carry the export's paths
    or text."""
    found = export_files(export)
    try:
        names = []
        for path in found.pages:
            page = json.loads(path.read_text(encoding="utf-8"))
            names += [*_page_names(page), *title_words(fix_meta_text(page.get("title") or ""))]
        for path in found.flat:
            for thread in load_flat_threads(path):
                names += [*thread.participants, *title_words(thread.title)]
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        raise RefusedError(f"export could not be read ({type(error).__name__})") from None
    return names


def title_words(title: str) -> list[str]:
    """The `is_distinctive` words of a conversation title: a group title may hold a person's name ("Hanta's 30th"),
    while its everyday words ("famille", "tsara") name nobody and would fail every file that uses them."""
    if not isinstance(title, str):
        raise TypeError("a title is not text")
    # Checked as typed (case and accents folded), before `compare_key` collapses letters: "famille" is a common
    # French word, "famile" is not.
    typed = _LETTERS_AND_DIGITS.findall(fold(title.translate(ASCII_PUNCTUATION)))
    return [word for word in typed if is_distinctive(word.translate(DIGIT_LETTERS))]


def _page_names(page: dict) -> list[str]:
    """Every name in one thread page, as Meta-encoded text fixed: the text value of every NAME_KEYS key or
    `*_name` key at any depth (participants, senders, reaction actors, `users` of a member event, ...). The page
    must have participants and messages."""
    if not isinstance(page["participants"], list) or not isinstance(page["messages"], list):
        raise TypeError("a thread page needs participant and message lists")
    raw: list[object] = []
    stack: list[object] = [page]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            raw += [value for key, value in node.items() if key in NAME_KEYS or key.endswith(NAME_KEY_SUFFIX)]
            stack += node.values()
        elif isinstance(node, list):
            stack += node
    if not all(isinstance(name, str) for name in raw):
        raise TypeError("a name is not text")
    return [fix_meta_text(name) for name in raw]


# ---- chat text ------------------------------------------------------------------------------------------------


def export_conversations(threads: list[Thread]) -> list[list[str]]:
    """Each thread's non-empty message texts, in order."""
    return [texts for thread in threads if (texts := [m.text for m in thread.messages if m.text])]


def read_chats(path: Path, trees: Iterable[Path]) -> list[list[str]]:
    """Each conversation's turn texts, in order, from the anonymised chats file at `path` (JSON lines,
    `{"messages": [{"role", "content"}]}`), system turns left out (the system prompt is ours, not chat text). The file
    must sit outside the repo and its worktrees. Refused when unreadable, malformed or holding no turns; errors never
    carry the file's path or contents."""
    target = path.expanduser().resolve()
    _require_outside(target, trees, "chats file")
    try:
        with target.open(encoding="utf-8") as lines:
            records = [json.loads(line) for line in lines if line.strip()]
        conversations = [[turn["content"] for turn in record["messages"] if turn["role"] != "system"]
                         for record in records]
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as error:
        raise RefusedError(f"chats file could not be read ({type(error).__name__})") from None
    if not all(isinstance(text, str) for texts in conversations for text in texts):
        raise RefusedError("chats file could not be read (a turn's content is not text)")
    conversations = [texts for c in conversations if (texts := [text for text in c if text])]
    if not conversations:
        raise RefusedError("chats file holds no turns")
    return conversations


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


def inner_allowed(path: str) -> frozenset[str]:
    """Compare keys of the INNER_NAME_ALLOWED words for the file at `path` (its text only, never its path)."""
    return frozenset(compare_key(word) for file, word in INNER_NAME_ALLOWED if file == path)


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


def strip_list(text: str, names: frozenset[str]) -> tuple[str, int]:
    """`text` (one entry per line, `#` comments) without entries holding a name token, and how many went."""
    kept, removed = [], 0
    for line in text.splitlines(keepends=True):
        if not line.startswith("#") and count_name_hits(line, names):
            removed += 1
        else:
            kept.append(line)
    return "".join(kept), removed


def without_author(path: str, text: str) -> tuple[str, int]:
    """`text` with each exact author string (AUTHOR_SHA256) blanked when `path` is in AUTHOR_FILES, and how many."""
    if path not in AUTHOR_FILES:
        return text, 0
    pieces, pos, found = [], 0, 0
    for start in (m.start() for m in _WORD_START.finditer(text)):  # every word start: "Dr A B" also tries "A B"
        if start < pos:
            continue
        match = _TWO_WORDS.match(text, start)
        if match and hashlib.sha256(match.group().encode("utf-8")).hexdigest() == AUTHOR_SHA256:
            pieces += [text[pos:start], " " * len(match.group())]
            pos, found = match.end(), found + 1
    return "".join(pieces) + text[pos:], found


# ---- private ids ----------------------------------------------------------------------------------------------


def load_private_ids(path: Path, trees: Iterable[Path]) -> tuple[re.Pattern[str], int]:
    """Whole-word, case-insensitive pattern for the ids in `path` (one per line, `#` comments), and their count.

    The file must sit outside the repo and its worktrees, so the ids never get committed. Errors never carry
    the file's contents.
    """
    target = path.expanduser().resolve()
    _require_outside(target, trees, "private ids file")
    try:
        lines = target.read_text("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RefusedError(f"private ids file could not be read ({type(error).__name__})") from None
    ids = sorted({line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")}, key=len,
                 reverse=True)
    if not ids:
        raise RefusedError("private ids file holds no ids")
    pattern = re.compile(rf"(?<![A-Za-z0-9_])(?:{'|'.join(map(re.escape, ids))})(?![A-Za-z0-9_])", re.IGNORECASE)
    return pattern, len(ids)


def _require_outside(target: Path, trees: Iterable[Path], what: str) -> None:
    """Refuse a private input file (`target`, resolved) that sits inside the repo or one of its worktrees."""
    if any(target == tree or tree in target.parents for tree in trees):
        raise RefusedError(f"{what} is inside the repo or one of its worktrees")


def count_private_ids(text: str, ids: re.Pattern[str]) -> int:
    """Whole-word matches of the private ids in `text`, JSON/Python escapes read as characters."""
    return len(ids.findall(_unescape(text)))


# ---- secrets --------------------------------------------------------------------------------------------------


def secret_hits(path: str, text: str) -> Counter:
    """Pattern name -> match count for one file. Never returns a matched value."""
    hits: Counter = Counter()
    name = PurePosixPath(path).name
    if name in SECRET_FILE_NAMES or name.startswith(".env.") or name.endswith(SECRET_FILE_SUFFIXES):
        hits["secret_file"] += 1
    for label, pattern in SECRET_PATTERNS.items():
        if n := len(pattern.findall(text)):
            hits[label] += n
    if n := sum(1 for email in EMAIL.findall(text) if not email_allowed(path, email)):
        hits["email"] += n
    return hits


def email_allowed(path: str, email: str) -> bool:
    """`email` is on a reserved test domain, or listed exactly for `path` in EMAIL_PROBE_FILES."""
    local, domain = email.casefold().rsplit("@", 1)
    return domain.endswith(RESERVED_EMAIL_TLDS) or (local, domain) in EMAIL_PROBE_FILES.get(path, frozenset())


def allowed_email_count(path: str, text: str) -> int:
    return sum(1 for email in EMAIL.findall(text) if email_allowed(path, email))


# ---- copy -----------------------------------------------------------------------------------------------------


def exclusion_entry(path: str) -> str | None:
    """The EXCLUDES path that covers `path`, if any."""
    for entry in EXCLUDES:
        if path == entry.path or (entry.path.endswith("/") and path.startswith(entry.path)):
            return entry.path
    return None


def exclusion_reason(path: str) -> str | None:
    return next((entry.reason for entry in EXCLUDES if entry.path == exclusion_entry(path)), None)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def worktrees(root: Path) -> set[Path]:
    """The repo root and every worktree of it."""
    return {root, *(Path(line[len("worktree "):]).resolve()
                    for line in _git(root, "worktree", "list", "--porcelain").splitlines()
                    if line.startswith("worktree "))}


def check_inputs(repo: Path, ref: str, out: Path) -> tuple[Path, str]:
    """The repo root and the commit sha, or RefusedError: dirty tree, bad ref, output inside a worktree or used."""
    try:
        root = Path(_git(repo, "rev-parse", "--show-toplevel").strip()).resolve()
        sha = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    except subprocess.CalledProcessError:
        raise RefusedError(f"not a git repo or unknown ref: {ref}") from None
    if _git(root, "status", "--porcelain").strip():
        raise RefusedError("working tree is not clean (local changes or untracked files)")
    target = out.resolve()
    for tree in worktrees(root):
        if target == tree or tree in target.parents:
            raise RefusedError("output dir is inside the repo or one of its worktrees")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise RefusedError("output dir exists and is not empty")
    return root, sha


def tracked_files(root: Path, sha: str) -> tuple[dict[str, tuple[bytes, int]], dict[str, str], list[dict]]:
    """The commit's regular files minus EXCLUDES, as path -> (bytes, mode), their texts (`text_of`), and the
    excluded entries.

    Every refusal is raised here, before anything is written.
    """
    archive = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", sha],
                             check=True, capture_output=True).stdout
    files, excluded, unsafe = {}, [], 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar.getmembers():
            if member.isdir() or member.name == "pax_global_header":
                continue
            if reason := exclusion_reason(member.name):
                excluded.append({"path": member.name, "reason": reason})
                continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():  # symlinks too: fail closed
                unsafe += 1
                continue
            files[member.name] = (tar.extractfile(member).read(), member.mode)
    if unsafe:  # counted, not named: a path may hold a name
        raise RefusedError(f"{unsafe} unsafe or non-regular archive entries (symlink, absolute or ../ path)")
    if archives := sum(is_archive(path) or has_archive_magic(data) for path, (data, _) in files.items()):  # counted
        raise RefusedError(f"{archives} archive files; the release ships no archives (their members are not read)")
    texts = {path: text_of(path, data) for path, (data, _) in files.items()}
    if non_text := sum(text is None for text in texts.values()):  # counted, not named
        raise RefusedError(f"{non_text} non-text files; the release ships text files only (lead decision 36)")
    if NOTICE_NAME in files:
        raise RefusedError(f"{NOTICE_NAME} already tracked; merge the licence notes by hand")
    return files, texts, excluded


def is_archive(path: str) -> bool:
    """`path` has an ARCHIVE_SUFFIXES suffix anywhere in its file name ("a.tar.gz", "x.zip"), any case."""
    return not ARCHIVE_SUFFIXES.isdisjoint(s.lower() for s in PurePosixPath(path).suffixes)


def has_archive_magic(data: bytes) -> bool:
    """`data` starts like an archive or a compressed stream, is a tar, or holds a zip record anywhere."""
    return (data.startswith(ARCHIVE_MAGIC)
            or data[TAR_MAGIC_OFFSET:TAR_MAGIC_OFFSET + len(TAR_MAGIC)] == TAR_MAGIC
            or any(marker in data for marker in ZIP_MARKERS))


def write_files(files: dict[str, tuple[bytes, int]], out: Path) -> list[str]:
    """Write `files` under `out`; returns their paths, sorted."""
    for name, (data, mode) in files.items():
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        dest.chmod(0o755 if mode & 0o111 else 0o644)
    return sorted(files)


def notice_text() -> str:
    lines = [
        "fy-m1: third-party data in this repository",
        "",
        "Code is Apache-2.0 (LICENSE). These data files keep their own licences:",
        "",
        *(f"- {path}: {licence}" for path, licence in sorted(DATA_LICENCES.items())),
        "",
        "The strict gate's other word lists are not included. Build them before using fym1.data.gate or",
        f"fym1.data.known_words: `{REBUILD_COMMAND}`.",
        "It writes:",
        "",
        *(f"- {path}" for path in REBUILT_LISTS),
        "",
        "from Malagasy, French and English Wikipedia (CC BY-SA 4.0), wordfreq by Robyn Speer",
        "(https://github.com/rspeer/wordfreq; word data CC BY-SA 4.0), the INSEE fichier des prenoms 2022",
        "(Licence Ouverte / Etalab 2.0) and HPLT 2.0 cleaned (CC0 1.0). Lists derived from CC BY-SA data are",
        "shared under the same licence (https://creativecommons.org/licenses/by-sa/4.0/).",
        "",
    ]
    return "\n".join(lines)


# ---- scan -----------------------------------------------------------------------------------------------------


def text_of(name: str, data: bytes) -> str | None:
    """Text of a file called `name`: a TEXT_SUFFIXES/TEXT_NAMES file in UTF-8 (UTF-8/16/32 with a BOM) with no
    control character but tab, LF and CR, no base64 `data:` URI and no Unicode tag character. Anything else is None:
    an unknown file type is never guessed at."""
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in TEXT_SUFFIXES and PurePosixPath(name).name not in TEXT_NAMES:
        return None
    for bom, codec in BOMS:
        if data.startswith(bom):
            data, encoding = data[len(bom):], codec
            break
    else:
        encoding = "utf-8"
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError:
        return None
    if _CONTROL.search(text):  # NUL, bell, escape...: binary posing as text
        return None
    return None if _DATA_URI.search(text) or _TAG_CHARS.search(text) else text


def is_list_file(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in LIST_GLOBS)


def build_snapshot(repo: Path, ref: str, out: Path, export: Path, private_ids: Path, chats: Path) -> Report:
    """Make the snapshot and check it. `out` is written only when every check passes. `chats` is the private
    anonymised chats.jsonl: its turns (fake names, joined messages) are checked as copied chat text too.

    Every RefusedError is raised before anything is written. The snapshot is built in a staging folder next to
    `out` and moved into place only when the report is ok; on a failed check or any error the staging folder is
    deleted, so `out` never holds a partial or failing snapshot.
    """
    root, sha = check_inputs(repo, ref, out)
    trees = worktrees(root)
    ids, id_count = load_private_ids(private_ids, trees)
    turns = read_chats(chats, trees)
    threads = read_export(export)
    names, short, placeholders = name_tokens(export_senders(export))
    if not names:
        raise RefusedError("export gave no sender name tokens")
    messages = export_conversations(threads)
    chat = (ChatIndex(messages, "export"), ChatIndex(turns, "chats file"))
    files, texts, excluded = tracked_files(root, sha)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=out.parent))
    try:
        report = _write_and_check(files, texts, excluded, staging, names, ids, chat)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report.ref, report.name_tokens, report.private_ids = sha, len(names), id_count
    report.chat_messages, report.chat_turns = sum(map(len, messages)), sum(map(len, turns))
    report.chat_short_messages = sum(index.short_messages for index in chat)
    report.short_tokens_skipped, report.placeholder_senders_skipped = short, placeholders
    if not report.ok:
        shutil.rmtree(staging)
        return report
    if out.exists():
        out.rmdir()  # check_inputs made sure it is empty
    staging.rename(out)
    return report


def _write_and_check(files: dict[str, tuple[bytes, int]], texts: dict[str, str], excluded: list[dict], out: Path,
                     names: frozenset[str], ids: re.Pattern[str], chat: tuple[ChatIndex, ...]) -> Report:
    kept = write_files(files, out)
    report = Report(ref="", files_copied=len(kept), excluded=excluded)
    matched = {exclusion_entry(e["path"]) for e in excluded}
    report.excludes_unmatched = [entry.path for entry in EXCLUDES if entry.path not in matched]
    texts = {**texts, NOTICE_NAME: notice_text()}
    (out / NOTICE_NAME).write_text(texts[NOTICE_NAME], encoding="utf-8")

    for path in (p for p in kept if p in HAND_LISTS):
        text, removed = strip_list(texts[path], names)
        if removed:
            (out / path).write_text(text, encoding="utf-8")
            texts[path] = text
            report.list_entries_removed[path] = removed

    excluded_refs = [entry.path for entry in EXCLUDES]
    for path, text in texts.items():
        name_text, authors = without_author(path, text)
        report.author_strings_allowed += authors
        text_hits = 0
        if path not in GENERATED_LISTS:
            text_hits = count_name_hits(name_text, names, allowed := inner_allowed(path))
            if allowed:
                report.inner_name_words_allowed += count_name_hits(name_text, names) - text_hits
        if hits := text_hits + count_name_hits(path, names):  # a path never uses the allow-list
            report.name_hits[path] = hits
        if id_hits := count_private_ids(text, ids) + count_private_ids(path, ids):
            report.private_id_hits[path] = id_hits
        if has_chat_text(text, chat) or has_chat_text(path, chat):
            report.chat_text_hits.append(path)
        if secrets := secret_hits(path, text):
            report.secret_hits[path] = dict(sorted(secrets.items()))
        report.emails_allowed += allowed_email_count(path, text)
        if n := text.count(PRIVATE_HF_ID):
            report.private_hf_ids[path] = n
        for ref_path in excluded_refs:
            if ref_path in text:
                report.references_to_excluded[ref_path] = report.references_to_excluded.get(ref_path, 0) + 1
    _redact_named_paths(report, names, ids, chat)
    return report


def _redact_named_paths(report: Report, names: frozenset[str], ids: re.Pattern[str],
                        chat: tuple[ChatIndex, ...]) -> None:
    """A repo path that itself holds a name token, a private id or copied chat text is reported as
    `<redacted path N>`."""
    aliases: dict[str, str] = {}

    def alias(path: str) -> str:
        # Read through the same escapes and readings as the scan. Redaction is cheap, so a name anywhere inside
        # a path word counts here, known word or not ("bodozafy").
        if (any(name in word for word in words(path) for name in names) or count_name_hits(path, names)
                or count_private_ids(path, ids) or has_chat_text(path, chat)):
            return aliases.setdefault(path, f"<redacted path {len(aliases) + 1}>")
        return path

    for attr in ("list_entries_removed", "name_hits", "private_id_hits", "secret_hits", "private_hf_ids"):
        setattr(report, attr, {alias(p): v for p, v in sorted(getattr(report, attr).items())})
    report.chat_text_hits = [alias(p) for p in report.chat_text_hits]
    report.excluded = [{**e, "path": alias(e["path"])} for e in report.excluded]


def write_report(report: Report, *paths: Path) -> None:
    text = json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n"
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
