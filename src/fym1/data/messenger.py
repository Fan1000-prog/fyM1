"""Read Messenger exports into plain text threads.

Each export format has its own loader (`Path -> list[Thread]`) listed in `LOADERS`.
To support a new format, write a loader and add it there; nothing downstream changes.
"""

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import ftfy


@dataclass(frozen=True)
class Message:
    sender: str
    text: str
    ts_ms: int


@dataclass(frozen=True)
class Thread:
    title: str
    participants: list[str]
    messages: list[Message]
    nicknames: dict[str, str] = field(default_factory=dict)  # nickname -> participant


def fix_meta_text(s: str) -> str:
    """Undo Meta's export encoding (UTF-8 bytes written out as Latin-1 characters), as NFC."""
    try:
        fixed = s.encode("latin-1").decode("utf-8")
    except UnicodeError:
        fixed = ftfy.fix_text(s)
    return unicodedata.normalize("NFC", fixed)


# "Download your information" folders with real conversations. message_requests and
# filtered_threads hold strangers and spam, so they are left out.
DYI_FOLDERS = ("inbox", "archived_threads", "e2ee_cutover")

# "X set the nickname for Y to Z." / "X set her own nickname to Z."
NICKNAME_SET = re.compile(
    r"^.+? set (?:the nickname for (?P<target>.+?)|(?:his|her|their|your) own nickname) to (?P<nickname>.+?)\.?$"
)

# Event types that are not something a person typed. Newer exports drop `type`.
DYI_TEXT_TYPES = {"Generic", "Share"}

# Text Messenger writes in place of media, reactions, unsends and group events (EN + FR UI).
PLACEHOLDER = re.compile(
    r"""^(?:
        .{0,80}\b(?:sent|a\ envoyé|avez\ envoyé)\ (?:a|an|une|un|\d+)\ (?:photos?|attachments?|stickers?|gifs?|
            videos?|voice\ messages?|links?|pièces?\ jointes?|autocollants?|vidéos?|messages?\ vocal|liens?)\b.*
      | .{0,80}\b(?:started\ a\ (?:video\ )?(?:chat|call)|missed\ (?:a|your)\ (?:video\ )?call|
            liked\ a\ message|a\ aimé\ un\ message)\b.*
      | (?:.{0,80}\ )?(?:reacted|a\ réagi|avez\ réagi)\ .{1,20}\ (?:to|à)\ .{0,40}message\.?
      | .{0,80}\b(?:unsent\ a\ message|a\ retiré\ un\ message|avez\ retiré\ un\ message)\.?
      | .{0,80}\b(?:changed\ the\ group\ photo|named\ the\ group|set\ the\ nickname|set\ (?:his|her|their|your)\ own\ nickname|set\ your\ nickname|
            changed\ the\ theme|a\ changé\ la\ photo\ du\ groupe|a\ nommé\ le\ groupe|a\ défini\ le\ surnom)\b.*
    )$""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def _dyi_text(raw: dict) -> str | None:
    """The typed text of one exported message, or None if it is not typed text."""
    if raw.get("is_unsent") or "call_duration" in raw:
        return None
    if raw.get("type", "Generic") not in DYI_TEXT_TYPES:
        return None
    text = fix_meta_text(raw.get("content", "")).strip()
    if not text or PLACEHOLDER.match(text):
        return None
    return text


def _nickname(raw: dict, participants: list[str]) -> tuple[str, str] | None:
    """(nickname, participant) if this message is a nickname-change event about a participant."""
    match = NICKNAME_SET.match(fix_meta_text(raw.get("content", "")).strip())
    if not match:
        return None
    sender = fix_meta_text(raw["sender_name"])
    person = match["target"] or sender
    # "you" is whoever downloaded the export: known in a 1:1 thread. In a group, pin the
    # nickname on the sender rather than drop it, so it still gets anonymised.
    if person.lower() == "you":
        others = [p for p in participants if p != sender]
        person = others[0] if len(others) == 1 else sender
    return (match["nickname"], person) if person in participants else None


def _load_dyi_thread(thread_dir: Path) -> Thread:
    pages = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(thread_dir.glob("message_*.json"))]
    first = pages[0]
    participants = [fix_meta_text(p["name"]) for p in first["participants"]]
    raws = [raw for page in pages for raw in page["messages"]]
    messages = [
        Message(sender=fix_meta_text(raw["sender_name"]), text=text, ts_ms=raw["timestamp_ms"])
        for raw in raws
        if (text := _dyi_text(raw)) is not None
    ]
    return Thread(
        title=fix_meta_text(first.get("title", "")),
        participants=participants,
        messages=sorted(messages, key=lambda m: m.ts_ms),
        nicknames=dict(n for raw in raws if (n := _nickname(raw, participants))),
    )


def load_dyi_threads(export_dir: Path) -> list[Thread]:
    """Threads from the standard "Download your information" JSON export."""
    thread_dirs = sorted(
        {
            page.parent
            for folder in DYI_FOLDERS
            for page in export_dir.glob(f"**/messages/{folder}/*/message_*.json")
        }
    )
    return [_load_dyi_thread(d) for d in thread_dirs]


LOADERS: tuple[Callable[[Path], list[Thread]], ...] = (load_dyi_threads,)


def load_threads(export_dir: Path) -> list[Thread]:
    """Every thread found in `export_dir`, from every known export format."""
    return [thread for load in LOADERS for thread in load(export_dir)]
