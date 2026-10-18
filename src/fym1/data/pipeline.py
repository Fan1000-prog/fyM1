"""Threads -> anonymised, Malagasy-filtered training records."""

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import NamedTuple

from fym1.data.anonymize import Anonymizer, assign_fake_names
from fym1.data.leftovers import leftover_names
from fym1.data.messenger import Thread
from fym1.data.segment import Conversation, to_conversations

MIN_MALAGASY_SHARE = 0.5
# results/data_stats.json, read by paper/numbers.py. Counts only, never text.
DATA_STATS_KEYS = ("threads", "conversations", "approved", "identity", "formal", "synthetic", "train_examples")


class BuildResult(NamedTuple):
    records: list[dict]
    common_word_replacements: int  # everyday words wiped because a contact is named that
    leftover_names: Counter  # capitalised words that survived, for the owner to check


def _malagasy_share(conversation: Conversation, is_malagasy: Callable[[str], bool]) -> float:
    return sum(is_malagasy(turn["content"]) for turn in conversation) / len(conversation)


def _record_id(conversation: Conversation) -> str:
    text = json.dumps(conversation, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _fake_names(threads: list[Thread], blocklist: Iterable[str]) -> dict[str, str]:
    """One fake per person across all threads; a nickname gets its person's fake first name.

    Blocklisted words (names the owner spotted on the leftover sheet) are treated as names.
    """
    people = {name for t in threads for name in t.participants} | {
        m.sender for t in threads for m in t.messages
    }
    nicknames = {nick: person for t in threads for nick, person in t.nicknames.items()}
    # nicknames go into the pool check too, so no fake name equals a real nickname
    names = assign_fake_names(people | set(nicknames) | set(blocklist))
    for nickname, person in nicknames.items():
        if person in names:
            names[nickname] = names[person].split()[0]
    return names


def build_records(
    threads: list[Thread], owner: str | None, is_malagasy: Callable[[str], bool], blocklist: Iterable[str] = ()
) -> BuildResult:
    """`{"id", "messages"}` per anonymised conversation that is at least half Malagasy.

`owner=None` makes every sender the assistant in turn (see `to_conversations`).

    Identical conversations are kept once.
    """
    names = _fake_names(threads, blocklist)
    anonymize = Anonymizer(names)
    records: dict[str, dict] = {}
    for thread in threads:
        # anonymise each message once; senders stay real so roles can still be assigned
        clean = replace(thread, messages=[replace(m, text=anonymize(m.text)) for m in thread.messages])
        for conversation in to_conversations(clean, owner):
            if _malagasy_share(conversation, is_malagasy) >= MIN_MALAGASY_SHARE:
                record_id = _record_id(conversation)
                records.setdefault(record_id, {"id": record_id, "messages": conversation})
    texts = [turn["content"] for record in records.values() for turn in record["messages"]]
    return BuildResult(
        list(records.values()), anonymize.common_word_replacements, leftover_names(texts, names.values())
    )


def merge_data_stats(existing: dict, **counts: int) -> dict:
    """`existing` stats with `counts` written over them; keys not known yet stay None."""
    unknown = set(counts) - set(DATA_STATS_KEYS)
    if unknown:
        raise ValueError(f"Unknown data_stats keys {sorted(unknown)}; expected {DATA_STATS_KEYS}")
    merged = {**existing, **counts}
    return {key: merged.get(key) for key in DATA_STATS_KEYS}
