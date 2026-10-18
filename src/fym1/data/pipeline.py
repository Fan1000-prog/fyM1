"""Threads -> anonymised, Malagasy-filtered training records."""

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import NamedTuple

from fym1.data.anonymize import Anonymizer, assign_fake_names
from fym1.data.gate import FAKE_STATS_KEYS, pool_names_in, strict_gate
from fym1.data.kept_checks import KEPT_COUNT_KEYS
from fym1.data.leftovers import leftover_names
from fym1.data.messenger import Thread
from fym1.data.segment import Conversation, to_conversations

MIN_MALAGASY_SHARE = 0.5
# results/data_stats.json, read by paper/make_numbers.py. Counts only, never text.
# KEPT_COUNT_KEYS: messages per role and post-run leak checks on the kept chats (fym1.data.kept_checks).
# FAKE_STATS_KEYS: conversations dropped for too many fakes, fakes per kept conversation (fym1.data.gate).
DATA_STATS_KEYS = (
    "threads",
    "conversations",
    *KEPT_COUNT_KEYS,
    *FAKE_STATS_KEYS,
    "approved",
    "identity",
    "formal",
    "synthetic",
    "train_examples",
)


class BuildResult(NamedTuple):
    records: list[dict]  # conversations that passed the strict gate
    common_word_replacements: int  # everyday words wiped because a contact is named that
    leftover_names: Counter  # capitalised words that survived anonymising, before the gate
    gate: dict[str, int]  # strict-gate counts (fym1.data.gate); counts only, never text


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

    Identical conversations are kept once. Then the strict gate replaces unknown capitalised
    words with fake names and drops every conversation with a long number, a contact or too
    many replaced words (`fym1.data.gate.strict_gate`). Pool names the raw conversation already
    held are real people there, so the gate replaces them too.
    """
    names = _fake_names(threads, blocklist)
    anonymize = Anonymizer(names)
    records: dict[str, dict] = {}
    real_pool_names: dict[str, set[str]] = {}
    for thread in threads:
        # anonymise each message once; senders stay real so roles can still be assigned
        clean = replace(thread, messages=[replace(m, text=anonymize(m.text)) for m in thread.messages])
        # segmenting reads only senders and times, so raw and clean conversations line up
        raw_conversations = to_conversations(thread, owner)
        for raw, conversation in zip(raw_conversations, to_conversations(clean, owner), strict=True):
            if _malagasy_share(conversation, is_malagasy) >= MIN_MALAGASY_SHARE:
                record_id = _record_id(conversation)
                records.setdefault(record_id, {"id": record_id, "messages": conversation})
                real_pool_names.setdefault(record_id, set()).update(pool_names_in(t["content"] for t in raw))
    texts = [turn["content"] for record in records.values() for turn in record["messages"]]
    fakes_written = [*names.values(), *anonymize.mention_fakes]
    gated = strict_gate(list(records.values()), fakes_written, disallowed=real_pool_names)
    return BuildResult(
        gated.kept, anonymize.common_word_replacements, leftover_names(texts, names.values()), gated.counts
    )


def merge_data_stats(existing: dict, **counts: int) -> dict:
    """`existing` stats with `counts` written over them; keys not known yet stay None."""
    unknown = set(counts) - set(DATA_STATS_KEYS)
    if unknown:
        raise ValueError(f"Unknown data_stats keys {sorted(unknown)}; expected {DATA_STATS_KEYS}")
    merged = {**existing, **counts}
    return {key: merged.get(key) for key in DATA_STATS_KEYS}
