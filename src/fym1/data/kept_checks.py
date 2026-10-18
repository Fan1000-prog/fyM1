"""Counts over the chats the strict gate kept: messages per role and post-run leak checks. No text.

The leak checks are deliberately simpler and blunter than the gate (fym1.data.gate): they run on the
gate's output as a second, independent look. Text is NFKC-normalised first (anonymize.normalise), so
full-width or circled forms count like plain ones. The checks excuse nothing, so a clean build can still
count things the gate keeps on purpose: a price ("1 500 000 Ar" is a spaced digit run) or a bare domain
with no path ("example.com" is a link). A non-zero count means "look at these", not "the gate failed";
only a phone number, email address, @handle or link with a path among them is a leak.
"""

import re
from collections.abc import Iterable

from fym1.data.anonymize import normalise

ROLES = ("user", "assistant")
MIN_LEAK_DIGITS = 7
KEPT_COUNT_KEYS = (
    "kept_messages",
    "kept_messages_user",
    "kept_messages_assistant",
    "leak_digit_runs",
    "leak_spaced_digit_runs",
    "leak_emails",
    "leak_at_signs",
    "leak_links",
)

DIGIT_RUN = re.compile(rf"\d{{{MIN_LEAK_DIGITS},}}")
SPACED_DIGIT_RUN = re.compile(r"\d(?:[\W_]*\d)*")  # any non-letter gap joins digits, as in the gate
EMAIL = re.compile(r"[\w.+-]+\s*[@＠]\s*[\w-]+(?:\.[\w-]+)+")
AT_SIGN = re.compile("[@＠]")
LINK = re.compile(
    r"\bhttps?://|\bwww\.|\b[\w-]+\.(?:com|net|org|mg|fr|io|me|ly|gl|info|biz)\b",
    re.IGNORECASE,
)


def _texts(records: Iterable[dict]) -> list[str]:
    return [normalise(turn["content"]) for record in records for turn in record["messages"]]


def message_counts(records: Iterable[dict]) -> dict[str, int]:
    """Kept messages in total and per role. Any role other than user/assistant is an error."""
    per_role = dict.fromkeys(ROLES, 0)
    for record in records:
        for turn in record["messages"]:
            if turn["role"] not in per_role:
                raise ValueError(f"Unexpected role {turn['role']!r} in record {record.get('id')}")
            per_role[turn["role"]] += 1
    return {
        "kept_messages": sum(per_role.values()),
        **{f"kept_messages_{role}": count for role, count in per_role.items()},
    }


def leak_counts(records: Iterable[dict]) -> dict[str, int]:
    """Matches left in kept chats: 7+ digit runs (plain and split by separators), emails, @, links."""
    texts = _texts(records)
    return {
        "leak_digit_runs": sum(len(DIGIT_RUN.findall(t)) for t in texts),
        "leak_spaced_digit_runs": sum(
            sum(c.isdigit() for c in run) >= MIN_LEAK_DIGITS for t in texts for run in SPACED_DIGIT_RUN.findall(t)
        ),
        "leak_emails": sum(len(EMAIL.findall(t)) for t in texts),
        "leak_at_signs": sum(len(AT_SIGN.findall(t)) for t in texts),
        "leak_links": sum(len(LINK.findall(t)) for t in texts),
    }


def kept_counts(records: list[dict]) -> dict[str, int]:
    """`message_counts` and `leak_counts` together, as written to results/data_stats.json."""
    return {**message_counts(records), **leak_counts(records)}
