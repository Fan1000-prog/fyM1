"""Read Messenger exports into plain text threads.

Each export format has its own loader (`Path -> list[Thread]`) listed in `LOADERS`.
To support a new format, write a loader and add it there; nothing downstream changes.
"""

import json
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    """(nickname, participant) if this DYI message is a readable nickname event about a participant."""
    return _parse_nickname_text(_dyi_event_text(raw), fix_meta_text(raw["sender_name"]), participants)


def _parse_nickname_text(text: str, sender: str, participants: list[str]) -> tuple[str, str] | None:
    match = next((m for pattern in NICKNAME_EVENTS if (m := pattern.match(text))), None)
    if not match:
        return None
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


def _dyi_event_text(raw: dict) -> str:
    return fix_meta_text(raw.get("content", "")).strip()


def _is_nickname_text(text: str) -> bool:
    return bool(PLACEHOLDER.match(text) and NICKNAME_WORD.search(text))


def _harvest_nicknames(
    events: Iterable[tuple[str, str]], participants: list[str]
) -> tuple[dict[str, str], int]:
    """From (text, sender) pairs: nickname -> participant, and how many nickname events could not be read."""
    nicknames = {}
    unharvested = 0
    for text, sender in events:
        if nickname := _parse_nickname_text(text, sender, participants):
            nicknames[nickname[0]] = nickname[1]
        elif _is_nickname_text(text):
            unharvested += 1
    return nicknames, unharvested


def _load_dyi_thread(thread_dir: Path) -> Thread:
    pages = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(thread_dir.glob("message_*.json"))]
    first = pages[0]
    participants = [fix_meta_text(p["name"]) for p in first["participants"]]
    raws = [raw for page in pages for raw in page["messages"]]
    nicknames, unharvested = _harvest_nicknames(
        ((_dyi_event_text(raw), fix_meta_text(raw["sender_name"])) for raw in raws), participants
    )
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


FLAT_MESSAGE_KEYS = {"sender", "timestamp"}


def _is_flat_export(data: object) -> bool:
    """`{"conversation": {...}, "messages": [{"id", "sender", "text", "timestamp"}, ...]}`."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("conversation"), dict)
        and isinstance(data.get("messages"), list)
        and all(isinstance(m, dict) and FLAT_MESSAGE_KEYS <= m.keys() for m in data["messages"])
    )


def _flat_zone(conversation: dict, path: Path) -> ZoneInfo:
    """The zone naive timestamps are written in: `conversation.timezone`, else UTC."""
    name = conversation.get("timezone") or "UTC"
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as error:
        raise ValueError(f"{path}: unknown conversation timezone") from error


def _flat_ts_ms(value: object, zone: ZoneInfo, path: Path, index: int) -> int:
    """ISO 8601 timestamp -> epoch milliseconds; a timestamp without an offset is in `zone`."""
    try:
        moment = datetime.fromisoformat(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: message {index} has an unreadable timestamp (want ISO 8601)") from error
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=zone)
    return int(moment.timestamp() * 1000)


def _flat_event_text(raw: dict) -> str:
    """A flat-export message's text as NFC, or "" if it is not a string."""
    text = raw.get("text")
    return unicodedata.normalize("NFC", text).strip() if isinstance(text, str) else ""


def _flat_text(raw: dict) -> str | None:
    """The typed text of one flat-export message, or None if it is missing, empty, media or an event."""
    text = _flat_event_text(raw)
    if not text or PLACEHOLDER.match(text):
        return None
    return text


def _load_flat_thread(path: Path, data: dict) -> Thread:
    conversation = data["conversation"]
    zone = _flat_zone(conversation, path)
    messages = [
        Message(sender=raw["sender"], text=text, ts_ms=_flat_ts_ms(raw["timestamp"], zone, path, index))
        for index, raw in enumerate(data["messages"])
        if (text := _flat_text(raw)) is not None
    ]
    listed = conversation.get("participants")
    listed = [p for p in listed if isinstance(p, str)] if isinstance(listed, list) else []
    senders = [raw["sender"] for raw in data["messages"]]
    participants = list(dict.fromkeys([*listed, *senders]))
    nicknames, unharvested = _harvest_nicknames(
        ((_flat_event_text(raw), raw["sender"]) for raw in data["messages"]), participants
    )
    return Thread(
        title=str(conversation.get("title") or ""),
        participants=participants,
        messages=sorted(messages, key=lambda m: m.ts_ms),
        nicknames=nicknames,
        unharvested_nickname_events=unharvested,
    )


def load_flat_threads(export: Path) -> list[Thread]:
    """Threads from one-file exports: `export` itself if it is a file, else its top-level *.json.

    A top-level JSON file of any other shape raises, so a renamed or reshaped export is never
    silently skipped. ("Download your information" pages sit in subfolders and are not read here.)
    """
    paths = [export] if export.is_file() else sorted(export.glob("*.json"))
    threads = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not _is_flat_export(data):
            raise ValueError(
                f"{path}: unknown export shape. Expected {{'conversation': {{...}}, 'messages': "
                "[{'id', 'sender', 'text', 'timestamp'}, ...]}."
            )
        threads.append(_load_flat_thread(path, data))
    return threads


LOADERS: tuple[Callable[[Path], list[Thread]], ...] = (load_dyi_threads, load_flat_threads)


def load_threads(export: Path) -> list[Thread]:
    """Every thread found in `export` (a folder, or one export file), from every known format."""
    return [thread for load in LOADERS for thread in load(export)]
