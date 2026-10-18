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
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from fym1.data.anonymize import compare_key, fold, is_variation_selector
from fym1.data.known_words import FIRST_NAME_WORDS, MG_FIRST_NAMES
from fym1.data.messenger import NICKNAME_EVENTS, NICKNAME_WORD, PLACEHOLDER, Thread, fix_meta_text
from fym1.data.name_scan import (
    _LETTERS,
    _LETTERS_AND_DIGITS,
    _READ_ERRORS,
    ASCII_PUNCTUATION,
    CHAT_TEXT_NGRAM,
    DIGIT_LETTERS,
    SHORT_MESSAGE_DISTINCTIVE,
    ChatIndex,
    RefusedError,
    _unescape,
    chat_words,  # noqa: F401  (re-exported: the release tests read it from here)
    count_name_hits,
    export_conversations,
    export_files,
    export_senders,
    has_chat_text,
    is_distinctive,
    is_name_word,  # noqa: F401  (re-exported: the release tests read it from here)
    name_tokens,
    read_export,
    readings,  # noqa: F401  (re-exported: the release tests read it from here)
    words,
)


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
# The vocabulary trim's kept token ids: a private build record that never ships (security finding #13). The trim
# keeps them only in the private checkpoint repo; a tracked copy fails the release before anything is written.
PRIVATE_BUILD_FILES = frozenset({"keep_ids.json"})
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
    ("scripts/trim_cost.py", "random"): REASON_RANDOM,
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

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")
# Text a reader decodes before it can be read (review findings 14, 20, 22): a `data:` URI carries bytes (base64 or
# percent-encoded) the scan never sees. A text file holding one is refused, in any spelling a browser or renderer
# still reads as one: HTML entities ("data&colon;", "data&#58;"), percent escapes ("data%3A") and backslash escapes
# are decoded first (`_unescape`), and whitespace inside the scheme and media type is ignored, as a URL parser does
# with tab, CR and LF ("da\ta:", "data:image/png;\nbase64,"). A data URI is `data:` + an optional `type/subtype`
# + `;parameters` + ",", or `data:` + anything with ";base64" before the comma (RFC 2397).
_DATA_SCHEME = re.compile(r"(?i)(?<![a-z0-9+.-])d\s*a\s*t\s*a\s*:")  # a candidate's start; "metadata:" is not one
_DATA_URI_HEAD = re.compile(r"(?i)data:(?:(?:[\w!#$&^.+-]+/[\w!#$&^.+-]+)?(?:;[^,;]*)*|[^,]*;base64[^,]*),")
_SPACE = re.compile(r"\s+")
# Invisible characters (review finding 21): every Unicode format character (category Cf: zero-width spaces and
# joiners, bidi controls, soft hyphen, tag characters...) and every variation selector (U+FE00-FE0F,
# U+E0100-E01EF) can hide or reorder what a reader sees. Allowed only: a BOM at the very start, and ZWJ (U+200D) or
# VS16 (U+FE0F) directly next to an emoji code point ("\u2764\ufe0f", "\U0001f468\u200d\U0001f4bb").
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_TAG_CHARS = re.compile("[\U000E0000-\U000E007F]")  # unassigned ones too
BOM = "\ufeff"
EMOJI_JOINERS = frozenset({"\u200d", "\ufe0f"})
KEYCAP = "\u20e3"


@dataclass
class Report:
    ref: str
    files_copied: int = 0
    excluded: list[dict] = field(default_factory=list)
    excludes_unmatched: list[str] = field(default_factory=list)
    name_tokens: int = 0
    short_tokens_skipped: int = 0
    placeholder_senders_skipped: int = 0
    export_schema_keys: int = 0
    label_words_common: int = 0
    nickname_events: int = 0
    export_words_allowed: int = 0
    author_strings_allowed: int = 0
    inner_name_words_allowed: int = 0
    list_entries_removed: dict[str, int] = field(default_factory=dict)
    name_hits: dict[str, int] = field(default_factory=dict)
    private_ids: int = 0
    private_id_hits: dict[str, int] = field(default_factory=dict)
    chat_messages: int = 0
    chat_turns: int = 0
    chat_short_messages: int = 0
    chat_free_texts: int = 0
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
                "export_schema_keys": self.export_schema_keys,
                "label_words_left_out_as_common": self.label_words_common,
                "nickname_events": self.nickname_events,
                "export_words_allowed": self.export_words_allowed,
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
                "export_free_text_fields_indexed": self.chat_free_texts,
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


# What each key of an export file holds (review finding 14, round 2). Every key of every export file must be in its
# format's schema, or the export is refused: a name under a key nobody expected ("nickname", a mention's "username",
# a name used as a key) would never reach the name list. A schema is an object (`{key: schema}`), a list of one
# schema (`[schema]`, every item matches it) or one of these leaves:
NAME = "name"  # a person's name (text): joins the name list
LABEL = "label"  # a conversation title, which may name someone ("Hanta's 30th"): the chat-text check reads it, and
# its distinctive words (`label_words`) join the name list
NOTE = "note"  # free text about the export (notes, platform): the chat-text check reads it; its listed given names
# (`note_names`) and its distinctive words that are not EXPORT_WORDS (`described_words`) join the name list (review
# findings 24/26: a union, the newer rule never replaces the older one).
TEXT = "text"  # free text someone typed (a message): the chat-text check reads it
DATA = "data"  # ids, timestamps, counts, flags, links, emoji: a number, true/false, null or text; no object or list.
# A DATA text's distinctive letter words that are not EXPORT_WORDS (`described_words`) join the name list (finding 23).
Schema = str | dict[str, "Schema"] | list["Schema"]

_MEDIA: Schema = [{"uri": DATA, "creation_timestamp": DATA, "backup_uri": DATA, "thumbnail": {"uri": DATA}}]
# A "Download your information" thread page (`message_*.json`), as Meta writes it.
DYI_SCHEMA: Schema = {
    "participants": [{"name": NAME}],
    "messages": [{
        "sender_name": NAME, "timestamp_ms": DATA, "content": TEXT, "type": DATA, "is_unsent": DATA,
        "is_taken_down": DATA, "is_geoblocked_for_viewer": DATA, "is_unsent_image_by_messenger_kid_parent": DATA,
        "photos": _MEDIA, "videos": _MEDIA, "audio_files": _MEDIA, "gifs": _MEDIA, "files": _MEDIA,
        "sticker": {"uri": DATA}, "share": {"link": DATA, "share_text": TEXT, "original_content_owner_name": NAME},
        "reactions": [{"reaction": DATA, "actor": NAME, "timestamp": DATA}], "users": [{"name": NAME}],
        "call_duration": DATA, "missed": DATA, "ip": DATA,
        "bumped_message_metadata": {"bumped_message": TEXT, "is_bumped": DATA},
    }],
    "title": LABEL, "is_still_participant": DATA, "thread_type": DATA, "thread_path": DATA,
    "image": {"uri": DATA, "creation_timestamp": DATA}, "joinable_mode": {"mode": DATA, "link": DATA},
    "magic_words": [{"magic_word": TEXT, "creation_timestamp_ms": DATA, "animation_emoji": DATA}],
}
# A one-file export (`fym1.data.messenger.load_flat_threads`).
FLAT_SCHEMA: Schema = {
    "conversation": {
        "title": LABEL, "participants": [NAME], "owner_name": NAME, "timezone": DATA, "platform": NOTE,
        "notes": NOTE, "date_range": {"start": DATA, "end": DATA}, "message_count": DATA,
    },
    "messages": [{
        "id": DATA, "sender": NAME, "text": TEXT, "timestamp": DATA,
        "reactions": [{"reaction": DATA, "actor": NAME}], "users": [{"name": NAME}],
    }],
}


# Words of DATA and NOTE fields that describe the export, not a person (review findings 23/24). Every letter word of a
# DATA text (links, paths, ids) and every word of a NOTE (notes, platform) goes through `label_words` like a title, so a
# name inside a link ("facebook.com/zafy.test") or a note joins the name list. The words below are the export's own
# vocabulary, read on the real export and reviewed one by one: exact folded words, each with its reason, never a name.
EXPORT_WORDS: dict[str, str] = {
    "america": "a time-zone region (`timezone` \"America/...\", notes saying which zone timestamps are in)",
    "call": "the English word, notes saying what a call entry is; not in the common-word lists, no person is named",
    "facebook": "the platform the export comes from (`platform`, notes); a link's host",
    "timestamps": "notes saying how timestamps are written",
    "urls": "notes saying how links are written",
    "utc": "the time-zone name UTC (`timezone`)",
}

# Nickname events as the release check reads them (review round 3): set, cleared, removed or changed, EN and FR UI,
# about anyone, participant or not. The loaders keep a nickname only when its person is a participant (they map it
# to that person's fake name); here the actor, the target and the nickname all join the name list, whoever they are.
# The loaders' own reading (`fym1.data.messenger.NICKNAME_EVENTS`, `Thread.nicknames`) is kept alongside, never
# replaced (review finding 25): a text either parser reads is an event, and a text the two read differently refuses
# the export.
NICKNAME_EVENT_NAMES = (
    re.compile(
        r"^(?P<actor>.+?) (?:set|cleared|removed|changed) (?:the nickname for (?P<target>.+?)"
        r"|(?:his|her|their|your)(?: own)? nickname)(?: to (?P<nickname>.+?))?\.?$",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^(?P<actor>.+?) (?:a|avez) (?:défini|modifié|supprimé|effacé) (?:le (?:surnom|pseudo) (?:de |d['’])"
        r"(?P<target>.+?)|(?:son|votre) (?:propre )?(?:surnom|pseudo))(?: sur (?P<nickname>.+?))?\.?$",
        re.IGNORECASE | re.DOTALL,
    ),
)
_READER_WORDS = frozenset({"you", "vous"})  # "You set…", "set the nickname for you": the reader, named elsewhere


UNREAD, DISAGREE = "unread", "disagree"  # why `nickname_event` refuses a text


@dataclass(frozen=True)
class NicknameEvent:
    names: list[str]  # actor, target, nickname as either parser reads them; "you" left out
    refused: str | None = None  # UNREAD: event wording no parser reads; DISAGREE: the two parsers read it differently


def _first_match(patterns: Iterable[re.Pattern[str]], text: str) -> re.Match[str] | None:
    return next((m for pattern in patterns if (m := pattern.match(text))), None)


def _said(match: re.Match[str]) -> tuple[str | None, str | None]:
    """(target, nickname) of a nickname-event match, folded for comparison."""
    return tuple(fold(value.strip()) if value else None for value in (match["target"], match["nickname"]))


def nickname_event(text: str) -> NicknameEvent | None:
    """None if `text` is not a nickname event. Otherwise the names it holds, read by both the release patterns
    (NICKNAME_EVENT_NAMES, on Messenger's event wording) and the loaders' (`messenger.NICKNAME_EVENTS`); a text with a
    nickname word that either reads is an event. Refused: event wording neither reads (UNREAD), or one reads it and
    the other does not, or they read a different target or nickname (DISAGREE)."""
    text = unicodedata.normalize("NFC", text).strip()
    if not NICKNAME_WORD.search(text):
        return None
    ours = _first_match(NICKNAME_EVENT_NAMES, text) if PLACEHOLDER.match(text) else None
    theirs = _first_match(NICKNAME_EVENTS, text)
    if ours is None and theirs is None:
        return NicknameEvent([], UNREAD) if PLACEHOLDER.match(text) else None
    # The loaders' `own` and `reader` groups hold a pronoun ("her own nickname"), not a name: left out.
    held = [*(ours.groupdict().values() if ours else ()), *((theirs["target"], theirs["nickname"]) if theirs else ())]
    names = [name for name in held if name and name.strip().lower() not in _READER_WORDS]
    agree = ours is not None and theirs is not None and _said(ours) == _said(theirs)
    return NicknameEvent(names, None if agree else DISAGREE)


@dataclass
class _Fields:
    """What the schema walk read, by leaf kind (DATA: its texts only), plus the distinct schema keys seen and the
    unknown and duplicate keys counted."""
    values: dict[str, list[str]] = field(default_factory=lambda: {NAME: [], LABEL: [], NOTE: [], TEXT: [], DATA: []})
    keys: set[str] = field(default_factory=set)
    unknown_keys: int = 0
    duplicate_keys: int = 0

    def load(self, path: Path) -> object:
        """`path` parsed as JSON; a key written twice in one object is counted (json.loads would keep the last)."""
        def pairs_to_dict(pairs: list[tuple[str, object]]) -> dict:
            self.duplicate_keys += len(pairs) - len({key for key, _ in pairs})
            return dict(pairs)
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs_to_dict)


def _walk(node: object, schema: Schema, found: _Fields, fix: Callable[[str], str]) -> None:
    """Read `node` against `schema` into `found`; a key not in the schema is counted (and not read). A node of the
    wrong shape, a name that is not text or free text that is neither text nor null raises TypeError."""
    if isinstance(schema, dict):
        if not isinstance(node, dict):
            raise TypeError("an export field is not an object")
        for key, value in node.items():
            if key not in schema:
                found.unknown_keys += 1
                continue
            found.keys.add(key)
            _walk(value, schema[key], found, fix)
    elif isinstance(schema, list):
        if not isinstance(node, list):
            raise TypeError("an export field is not a list")
        for item in node:
            _walk(item, schema[0], found, fix)
    elif schema == DATA:
        if isinstance(node, (dict, list)):
            raise TypeError("an export data field holds an object or a list")
        if isinstance(node, str):
            found.values[DATA].append(fix(node))
    elif node is None and schema != NAME:
        return
    elif not isinstance(node, str):
        raise TypeError("an export name or text field is not text")
    else:
        found.values[schema].append(fix(node))


@dataclass(frozen=True)
class ExportScan:
    threads: list[Thread]  # `read_export`
    names: list[str]  # every NAME field, nickname-event name, loader nickname, participant and sender, and
    # distinctive LABEL, NOTE and DATA word
    free_texts: list[str]  # every LABEL, NOTE and TEXT field, for the chat-text check
    label_words_common: int  # LABEL words left out of the names as common (`label_words`)
    keys: int  # distinct schema keys the export uses
    nickname_events: int = 0  # nickname events read (`nickname_event`)
    export_words_allowed: int = 0  # distinctive NOTE and DATA words left out as EXPORT_WORDS


def scan_export(export: Path) -> ExportScan:
    """Every person name and free text the export holds, read from its raw files against the format's schema
    (DYI_SCHEMA for thread pages in any folder, FLAT_SCHEMA for one-file exports) rather than from the loaded
    conversations (which drop unsent, call and event messages), plus the names in every nickname event, the
    loaders' nicknames (`Thread.nicknames`), the loaded threads' participants and senders, the listed given names of every note (`note_names`) and the distinctive
    words of every title, note and data text (`described_words`). Refused, counted, never named: a key outside the
    schema, a key written twice in one object, a nickname event no pattern reads or that the release patterns and
    the loaders' read differently, a page of any other shape. Errors never carry the export's paths or text."""
    found_files, found = export_files(export), _Fields()
    threads = read_export(export, found_files)
    try:
        for path in found_files.pages:
            page = found.load(path)
            if not isinstance(page.get("participants"), list) or not isinstance(page.get("messages"), list):
                raise TypeError("a thread page needs participant and message lists")
            _walk(page, DYI_SCHEMA, found, fix_meta_text)
        for path in found_files.flat:
            _walk(found.load(path), FLAT_SCHEMA, found, str)
    except _READ_ERRORS as error:
        raise RefusedError(f"export could not be read ({type(error).__name__})") from None
    if found.unknown_keys:
        raise RefusedError(f"export holds {found.unknown_keys} keys outside its format's schema "
                           f"(fym1.release.DYI_SCHEMA / FLAT_SCHEMA)")
    if found.duplicate_keys:
        raise RefusedError(f"export holds {found.duplicate_keys} keys written twice in one object")
    events = [event for text in found.values[TEXT] if (event := nickname_event(text)) is not None]
    refused = Counter(event.refused for event in events if event.refused)
    if refused[UNREAD]:
        raise RefusedError(f"export holds {refused[UNREAD]} nickname events the release check cannot read "
                           f"(fym1.release.NICKNAME_EVENT_NAMES)")
    if refused[DISAGREE]:
        raise RefusedError(f"export holds {refused[DISAGREE]} nickname events the release check and the loaders "
                           f"read differently (fym1.release.NICKNAME_EVENT_NAMES / messenger.NICKNAME_EVENTS)")
    labels = [label_words(label) for label in found.values[LABEL]]
    described = [described_words(text) for text in found.values[NOTE]]
    described += [described_words(" ".join(_LETTERS.findall(text))) for text in found.values[DATA]]
    return ExportScan(
        threads=threads,
        names=[*found.values[NAME], *(name for event in events for name in event.names),
               *(nickname for thread in threads for nickname in thread.nicknames),
               *export_senders(threads),
               *(word for kept, _ in labels for word in kept), *(word for kept, _ in described for word in kept),
               *(name for note in found.values[NOTE] for name in note_names(note))],
        free_texts=[*found.values[LABEL], *found.values[NOTE], *found.values[TEXT]],
        label_words_common=sum(common for _, common in labels),
        keys=len(found.keys),
        nickname_events=len(events),
        export_words_allowed=sum(allowed for _, allowed in described),
    )


def note_names(note: str) -> list[str]:
    """The words of a NOTE that are listed given names (`first_names.txt`, `mg_first_names.txt`), case and accents
    folded, EXPORT_WORDS or not: they join the name list alongside `described_words` (review finding 26)."""
    return [word for word in _LETTERS.findall(fold(note)) if word in FIRST_NAME_WORDS or word in MG_FIRST_NAMES]


def described_words(text: str) -> tuple[list[str], int]:
    """The `label_words` of a NOTE or of a DATA text's letter words that are not EXPORT_WORDS, and how many
    distinctive words were left out as EXPORT_WORDS."""
    kept = label_words(text)[0]
    named = [word for word in kept if word not in EXPORT_WORDS]
    return named, len(kept) - len(named)


def label_words(label: str) -> tuple[list[str], int]:
    """The `is_distinctive` words of a LABEL (a conversation title), and how many words were left out as
    common: a group title may hold a person's name ("Hanta's 30th"), while its everyday words ("famille", "tsara")
    name nobody and would fail every file that uses them."""
    # Checked as typed (case and accents folded), before `compare_key` collapses letters: "famille" is a common
    # French word, "famile" is not.
    typed = _LETTERS_AND_DIGITS.findall(fold(label.translate(ASCII_PUNCTUATION)))
    kept = [word for word in typed if is_distinctive(word.translate(DIGIT_LETTERS))]
    return kept, len(typed) - len(kept)


# ---- chat text ------------------------------------------------------------------------------------------------


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


def inner_allowed(path: str) -> frozenset[str]:
    """Compare keys of the INNER_NAME_ALLOWED words for the file at `path` (its text only, never its path)."""
    return frozenset(compare_key(word) for file, word in INNER_NAME_ALLOWED if file == path)


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
    if private := sum(PurePosixPath(path).name.lower() in PRIVATE_BUILD_FILES for path in files):  # counted
        raise RefusedError(f"{private} private build files (trim keep ids); they never ship")
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
    control character but tab, LF and CR, no `data:` URI (`has_data_uri`) and no invisible format character
    (`hidden_format_chars`). Anything else is None:
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
    return None if has_data_uri(text) or hidden_format_chars(text) else text


def has_data_uri(text: str) -> bool:
    """`text`, as written or with its escapes decoded (`_unescape`), holds a `data:` URI (`_DATA_URI_HEAD`), read with
    the whitespace between the scheme's start and the next comma taken out."""
    for form in dict.fromkeys((text, _unescape(text))):
        for start in _DATA_SCHEME.finditer(form):
            comma = form.find(",", start.start())
            if comma < 0:
                break  # no comma left in this form: no data URI after here
            if _DATA_URI_HEAD.match(_SPACE.sub("", form[start.start():comma + 1])):
                return True
    return False


def hidden_format_chars(text: str) -> int:
    """How many invisible characters `text` holds: Unicode format characters (Cf), variation selectors and tag
    characters, except a BOM at the start and ZWJ or VS16 directly next to an emoji code point."""
    hidden = 0
    for match in _NON_ASCII.finditer(text):
        i, c = match.start(), match.group()
        if not (unicodedata.category(c) == "Cf" or is_variation_selector(c) or _TAG_CHARS.match(c)):
            continue
        if (c == BOM and i == 0) or (c in EMOJI_JOINERS and _next_to_emoji(text, i)):
            continue
        hidden += 1
    return hidden


def _is_emoji(c: str) -> bool:
    """An emoji code point: a symbol (So: pictographs, flags' regional indicators), a skin-tone modifier or the
    keycap mark that ends "1\ufe0f\u20e3"."""
    return unicodedata.category(c) == "So" or "\U0001f3fb" <= c <= "\U0001f3ff" or c == KEYCAP


def _next_to_emoji(text: str, i: int) -> bool:
    return (i > 0 and _is_emoji(text[i - 1])) or (i + 1 < len(text) and _is_emoji(text[i + 1]))


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
    scan = scan_export(export)
    names, short, placeholders = name_tokens(scan.names)
    if not names:
        raise RefusedError("export gave no sender name tokens")
    messages = export_conversations(scan.threads)
    # Every free-text field of the raw files (titles, notes, unsent and event messages too) is indexed on its own.
    chat = (ChatIndex([*messages, *([text] for text in scan.free_texts)], "export"), ChatIndex(turns, "chats file"))
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
    report.export_schema_keys, report.label_words_common = scan.keys, scan.label_words_common
    report.nickname_events, report.export_words_allowed = scan.nickname_events, scan.export_words_allowed
    report.chat_free_texts = len(scan.free_texts)
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
