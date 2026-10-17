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
    # nickname events we dropped but could not read (unknown wording or non-participant)
    unharvested_nickname_events: int = 0


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

# Nickname-change events, EN and FR UI. `target` names the person; `own` means the
# sender named themselves; `reader` ("your"/"votre") means whoever downloaded the export.
NICKNAME_EVENTS = (
    re.compile(
        r"^.+? set (?:the nickname for (?P<target>.+?)|(?P<own>his|her|their) own nickname"
        r"|(?P<reader>your)(?: own)? nickname) to (?P<nickname>.+?)\.?$"
    ),
    re.compile(
        r"^.+? (?:a|avez) défini (?:le (?:surnom|pseudo) (?:de |d')(?P<target>.+?)"
        r"|(?P<own>son) (?:propre )?(?:surnom|pseudo)|(?P<reader>votre) (?:propre )?(?:surnom|pseudo))"
        r" sur (?P<nickname>.+?)\.?$"
    ),
)
NICKNAME_WORD = re.compile(r"\b(?:nickname|surnom|pseudo)\b", re.IGNORECASE)

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
      | .{0,80}\b(?:changed\ the\ group\ photo|named\ the\ group|changed\ the\ theme|
            a\ changé\ la\ photo\ du\ groupe|a\ nommé\ le\ groupe)\b.*
      | .{0,80}\b(?:set|cleared|removed|changed)\ (?:the|your|his|her|their)(?:\ own)?\ nickname\b.*
      | .{0,80}\b(?:a|avez)\ (?:défini|modifié|supprimé|effacé)\ (?:le|votre|son)\ (?:propre\ )?(?:surnom|pseudo)\b.*
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


def parse_nickname_event(raw: dict, participants: list[str]) -> tuple[str, str] | None:
    """(nickname, participant) if this message is a readable nickname event about a participant."""
    text = fix_meta_text(raw.get("content", "")).strip()
    match = next((m for pattern in NICKNAME_EVENTS if (m := pattern.match(text))), None)
    if not match:
        return None
    sender = fix_meta_text(raw["sender_name"])
    if match["own"]:
        person = sender
    elif match["reader"] or match["target"].lower() in ("you", "vous"):
        person = _reader(text, sender, participants)
    else:
        person = match["target"]
    return (match["nickname"], person) if person in participants else None


def _reader(text: str, sender: str, participants: list[str]) -> str:
    """Whoever downloaded the export: the sender of a "You …" event, else the other side of
    a 1:1 thread. In a group it is unknown; pin it on the sender so the nickname is still
    anonymised."""
    if text.startswith(("You ", "Vous ")):
        return sender
    others = [p for p in participants if p != sender]
    return others[0] if len(others) == 1 else sender


def _is_nickname_event(raw: dict) -> bool:
    text = fix_meta_text(raw.get("content", "")).strip()
    return bool(PLACEHOLDER.match(text) and NICKNAME_WORD.search(text))


def _load_dyi_thread(thread_dir: Path) -> Thread:
    pages = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(thread_dir.glob("message_*.json"))]
    first = pages[0]
    participants = [fix_meta_text(p["name"]) for p in first["participants"]]
    raws = [raw for page in pages for raw in page["messages"]]
    nicknames = {}
    unharvested = 0
    for raw in raws:
        if nickname := parse_nickname_event(raw, participants):
            nicknames[nickname[0]] = nickname[1]
        elif _is_nickname_event(raw):
            unharvested += 1
    messages = [
        Message(sender=fix_meta_text(raw["sender_name"]), text=text, ts_ms=raw["timestamp_ms"])
        for raw in raws
        if (text := _dyi_text(raw)) is not None
    ]
    return Thread(
        title=fix_meta_text(first.get("title", "")),
        participants=participants,
        messages=sorted(messages, key=lambda m: m.ts_ms),
        nicknames=nicknames,
        unharvested_nickname_events=unharvested,
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
