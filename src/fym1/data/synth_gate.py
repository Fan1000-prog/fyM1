"""Strict automatic sample gate for the filtered synthetic chats (spec §8: owner-review gates run by agents).

Every chat is checked by code; a chat with any of DROP_REASONS is dropped, never patched:

- `broken`: not `{"id", "messages"}`, a turn that is not user/assistant text, roles that do not alternate
  user -> assistant, an empty turn, a placeholder (`<TEL>`, `[NAME]`), a turn repeated verbatim, a turn count
  outside synth.MIN_TURNS..MAX_TURNS, an id that is not the hash of its messages, or an id seen before.
- `real_name`: a word that is a sender or participant name of the raw export (`fym1.release.count_name_hits`:
  any case, spaced-out letters, digits for letters, a name glued to a marker or ending).
- `unknown_name`: a capitalised word no known-word list clears and not a fake-pool name (lead decision 30).
- `long_number`, `contact`, `hidden_char`: the strict data gate's checks (`fym1.data.gate`).
- `leak_8gram`: any turn shares an 8-word run with a raw export message. Any such match also fails the gate.
- `drift`: an assistant turn with no Malagasy (full French/English). User turns may be French: counted, not dropped.
- `identity`: an assistant turn that claims to be, or names, another assistant (`fym1.eval.identity_check`).

A fixed-seed random sample also gets counts for the checks a person would judge: capitalised words outside the
common-word list (the leftover-names sheet) and words the anonymiser gate would replace (lower case included).
Reports hold counts and chat ids only, never text or names.
"""

import random
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from fym1 import release
from fym1.data.anonymize import normalise
from fym1.data.gate import contact_left, has_hidden_char, has_long_number, replaced_word_count, unknown_capitalised_words
from fym1.data.leftovers import leftover_names
from fym1.data.synth import FAKE_POOL, MAX_TURNS, MIN_TURNS, PLACEHOLDER, record_id
from fym1.eval.identity_check import self_claims, unexcused_mentions
from fym1.eval.metrics import NgramIndex

DROP_REASONS = ("broken", "real_name", "unknown_name", "long_number", "contact", "hidden_char", "leak_8gram",
                "drift", "identity")  # fmt: skip
FAIL_REASONS = ("leak_8gram",)  # any chat with one fails the whole gate (spec §8.6: fail on any 8-word match)
SAMPLE_SIZE = 100
SAMPLE_SEED = 20261018
ROLES = ("user", "assistant")
AUTO_DECISIONS = (
    "Every chat with any drop reason is dropped whole; nothing is patched or replaced.",
    "A real-name match counts any sender or participant name token of 3+ letters, also inside a longer word; "
    "ambiguous matches are dropped, not cleared.",
    "User turns with no Malagasy are kept (the spec's drift rule is about replies); counted in "
    "chats_with_user_turn_not_malagasy.",
    "Lower-case words no list knows are not dropped (lead decision 30: few-shot examples hold only fakes); "
    "real names in lower case are caught by real_name. Counted on the sample as chats_with_unlisted_word.",
    "Leftover-sheet words (capitalised, outside the common-word list) that a known-word list clears are kept; "
    "only uncleared ones would drop (unknown_name).",
    "Any 8-word match with the raw export fails the gate as well as dropping the chat.",
)


@dataclass(frozen=True)
class RawReference:
    """What the gate compares against in the raw export: name tokens and 8-word runs. Never written anywhere."""

    names: frozenset[str]
    leaks: NgramIndex


@dataclass(frozen=True)
class GateResult:
    kept_ids: list[str]
    report: dict


def raw_reference(senders: Iterable[str], texts: list[str]) -> RawReference:
    """Name tokens of every sender/participant and 8-word runs of every message text of the raw export."""
    names, _, _ = release.name_tokens(senders)
    return RawReference(names, release.chat_text_index(texts, "raw export"))


def _turns(record: object) -> list | None:
    messages = record.get("messages") if isinstance(record, dict) else None
    return messages if isinstance(messages, list) else None


def broken(record: object) -> bool:
    """The chat is not a well-formed user -> assistant conversation whose id is its own hash."""
    turns = _turns(record)
    if not turns or not MIN_TURNS <= len(turns) <= MAX_TURNS:
        return True
    if not all(isinstance(t, dict) and isinstance(t.get("content"), str) for t in turns):
        return True
    texts = [t["content"].strip() for t in turns]
    expected_roles = [ROLES[i % 2] for i in range(len(turns))]
    return (
        [t.get("role") for t in turns] != expected_roles
        or len(turns) % 2 == 1
        or not all(texts)
        or any(PLACEHOLDER.search(text) for text in texts)
        or len(set(texts)) < len(texts)
        or record.get("id") != record_id(turns)
    )


def _replies(turns: list[dict]) -> list[tuple[str, str]]:
    """(user turn, assistant reply) pairs."""
    return [(turns[i - 1]["content"], turns[i]["content"]) for i in range(1, len(turns), 2)]


def _claims_another_assistant(prompt: str, reply: str) -> bool:
    return bool(self_claims(reply) or unexcused_mentions(prompt, reply))


def chat_reasons(record: dict, raw: RawReference, is_malagasy: Callable[[str], bool]) -> list[str]:
    """Every drop reason of one chat, in DROP_REASONS order; empty when it passes."""
    if broken(record):
        return ["broken"]
    turns = record["messages"]
    texts = [t["content"] for t in turns]
    both = [*texts, *map(normalise, texts)]  # the chat is kept as written, so read it as written and normalised
    replies = _replies(turns)
    found = {
        "real_name": any(release.count_name_hits(text, raw.names) for text in texts),
        "unknown_name": unknown_capitalised_words(texts, fake_names=FAKE_POOL) > 0,
        "long_number": any(map(has_long_number, both)),
        "contact": any(map(contact_left, both)),
        "hidden_char": any(map(has_hidden_char, both)),
        "leak_8gram": any(release.has_chat_text(text, [raw.leaks]) for text in texts),
        "drift": not all(is_malagasy(reply) for _, reply in replies),
        "identity": any(_claims_another_assistant(prompt, reply) for prompt, reply in replies),
    }
    return [reason for reason in DROP_REASONS if found.get(reason)]


def _user_turn_not_malagasy(record: dict, is_malagasy: Callable[[str], bool]) -> bool:
    return any(not is_malagasy(t["content"]) for t in record["messages"] if t["role"] == "user")


def sample_counts(records: list[dict], dropped_ids: set[str], size: int, seed: int) -> dict:
    """Counts for the judgement checks on a fixed-seed sample of well-formed chats (ids, never text)."""
    sample = random.Random(seed).sample(records, min(size, len(records)))
    leftovers = [leftover_names((t["content"] for t in r["messages"]), FAKE_POOL) for r in sample]
    return {
        "size": len(sample),
        "seed": seed,
        "ids": sorted(r["id"] for r in sample),
        "dropped": sum(r["id"] in dropped_ids for r in sample),
        "leftover_words": sum(sum(c.values()) for c in leftovers),
        "chats_with_leftover_word": sum(bool(c) for c in leftovers),
        "leftover_words_unresolved": sum(
            unknown_capitalised_words((t["content"] for t in r["messages"]), FAKE_POOL) for r in sample),
        "chats_with_unlisted_word": sum(
            replaced_word_count((t["content"] for t in r["messages"]), FAKE_POOL) > 0 for r in sample),
    }  # fmt: skip


def run_gate(records: list[dict], raw: RawReference, is_malagasy: Callable[[str], bool],
             sample_size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> GateResult:  # fmt: skip
    """Kept chat ids (input order) and the counts-only report."""
    by_reason, dropped_ids, kept_ids, seen = Counter(dict.fromkeys(DROP_REASONS, 0)), {}, [], set()
    for index, record in enumerate(records):
        reasons = chat_reasons(record, raw, is_malagasy)
        record_key = record.get("id") if isinstance(record, dict) and isinstance(record.get("id"), str) else None
        if not reasons and record_key in seen:
            reasons = ["broken"]
        by_reason.update(reasons)
        if reasons:
            dropped_ids[record_key or f"index-{index}"] = reasons
        else:
            kept_ids.append(record_key)
            seen.add(record_key)
    well_formed = [r for r in records if not broken(r)]
    failed = any(by_reason[reason] for reason in FAIL_REASONS) or not kept_ids
    report = {
        "checked": len(records),
        "kept": len(kept_ids),
        "dropped": len(dropped_ids),
        "dropped_by_reason": dict(by_reason),
        "dropped_ids": dropped_ids,
        "chats_with_user_turn_not_malagasy": sum(_user_turn_not_malagasy(r, is_malagasy) for r in well_formed),
        "raw_name_tokens": len(raw.names),
        "sample": sample_counts(well_formed, set(dropped_ids), sample_size, seed),
        "auto_decisions": list(AUTO_DECISIONS),
        "status": "fail" if failed else "pass",
    }
    return GateResult(kept_ids, report)
