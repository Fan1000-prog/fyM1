"""Threads -> anonymised, Malagasy-filtered training records."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from typing import NamedTuple

from fym1.data.anonymize import Anonymizer, assign_fake_names
from fym1.data.messenger import Thread
from fym1.data.segment import Conversation, to_conversations

MIN_MALAGASY_SHARE = 0.5


class BuildResult(NamedTuple):
    records: list[dict]
    common_word_replacements: int  # everyday words wiped because a contact is named that


def _malagasy_share(conversation: Conversation, is_malagasy: Callable[[str], bool]) -> float:
    return sum(is_malagasy(turn["content"]) for turn in conversation) / len(conversation)


def _record_id(conversation: Conversation) -> str:
    text = json.dumps(conversation, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _fake_names(threads: list[Thread]) -> dict[str, str]:
    """One fake per person across all threads; a nickname gets its person's fake first name."""
    people = {name for t in threads for name in t.participants} | {
        m.sender for t in threads for m in t.messages
    }
    nicknames = {nick: person for t in threads for nick, person in t.nicknames.items()}
    # nicknames go into the pool check too, so no fake name equals a real nickname
    names = assign_fake_names(people | set(nicknames))
    for nickname, person in nicknames.items():
        names[nickname] = names[person].split()[0]
    return names


def build_records(
    threads: list[Thread], owner: str, is_malagasy: Callable[[str], bool]
) -> BuildResult:
    """`{"id", "messages"}` per anonymised conversation that is at least half Malagasy.

    Identical conversations are kept once.
    """
    anonymize = Anonymizer(_fake_names(threads))
    records: dict[str, dict] = {}
    for thread in threads:
        # anonymise each message once; senders stay real so roles can still be assigned
        clean = replace(thread, messages=[replace(m, text=anonymize(m.text)) for m in thread.messages])
        for conversation in to_conversations(clean, owner):
            if _malagasy_share(conversation, is_malagasy) >= MIN_MALAGASY_SHARE:
                record_id = _record_id(conversation)
                records.setdefault(record_id, {"id": record_id, "messages": conversation})
    return BuildResult(list(records.values()), anonymize.common_word_replacements)
