"""Threads -> anonymised, Malagasy-filtered training records."""

import hashlib
import json
from collections.abc import Callable

from fym1.data.anonymize import anonymize, assign_fake_names
from fym1.data.messenger import Thread
from fym1.data.segment import Conversation, to_conversations

MIN_MALAGASY_SHARE = 0.5


def _malagasy_share(conversation: Conversation, is_malagasy: Callable[[str], bool]) -> float:
    return sum(is_malagasy(turn["content"]) for turn in conversation) / len(conversation)


def _record_id(conversation: Conversation) -> str:
    text = json.dumps(conversation, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def build_records(
    threads: list[Thread], owner: str, is_malagasy: Callable[[str], bool]
) -> list[dict]:
    """`{"id", "messages"}` per anonymised conversation that is at least half Malagasy.

    Fake names are assigned across all threads, so a person keeps one fake name
    everywhere. Identical conversations are kept once.
    """
    people = {name for t in threads for name in t.participants} | {
        m.sender for t in threads for m in t.messages
    }
    names = assign_fake_names(people)
    records: dict[str, dict] = {}
    for thread in threads:
        for conversation in to_conversations(thread, owner):
            clean = [{**turn, "content": anonymize(turn["content"], names)} for turn in conversation]
            if _malagasy_share(clean, is_malagasy) >= MIN_MALAGASY_SHARE:
                record_id = _record_id(clean)
                records.setdefault(record_id, {"id": record_id, "messages": clean})
    return list(records.values())
