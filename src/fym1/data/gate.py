"""Strict automatic gate run after anonymising: drop any conversation that may still leak.

No person reads the anonymised text. Instead a whole conversation is dropped if any turn holds:

- `long_number`: 7+ digits in one run, counting across separators (space, no-break spaces,
  `. - / ,`, unicode dashes). A price is kept: clean 3-digit groups (or no separator) followed
  by a currency word (`1 500 000 Ar`, `2500000 ariary`, `1 250 000,50 €`). A full date
  (`15/03/2026`, 8 digits) counts as a long number too.
- `leftover_name`: a capitalised word that is not a common Malagasy word, a placeholder or a
  fake name (`fym1.data.leftovers`). A word capitalised only at the start of a sentence is
  ordinary if the corpus also has it in lower case ("Zazakely tonga." + "ilay zazakely").
- `contact`: an email, a link with a path or query, an `@handle`, or a private-use mark the
  anonymiser leaves only when a replacement failed.

Only counts leave this module; never text or words.
"""

import re
from collections.abc import Iterable
from typing import NamedTuple

from fym1.data.anonymize import EMAIL, FALLBACK_NAME, FIRST_NAMES, SURNAMES, compare_key
from fym1.data.leftovers import LETTERS, fake_name_keys, leftover_matches

GATE_REASONS = ("long_number", "leftover_name", "contact")

MIN_NUMBER_DIGITS = 7
_SEPARATORS = " \u00a0\u202f.,/\\-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
DIGIT_RUN = re.compile(rf"\d(?:[{_SEPARATORS}]*\d)*")
CURRENCY_AFTER = re.compile(r"\s*(?:€|(?:Ar|AR|Ariary|ariary|fmg|FMG|euros?|eur)(?!\w))")
CURRENCY_WORDS = ("Ar", "Ariary", "Fmg", "Euro", "Euros", "Eur")  # never a leftover name
# Clean price shapes: "1 500 000", "1.500.000", "2500000", each with optional cents ",50".
PRICE = re.compile(r"\d{1,3}(?:([ \u00a0\u202f.,])\d{3})?(?:\1\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?")

LINK_WITH_PATH = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:[\w-]+\.)+[a-z]{2,}[/?]\S*", re.IGNORECASE)
HANDLE = re.compile(r"(?<![\w.])@[\w.]*\w")
PRIVATE_USE = re.compile("[\ue000-\uf8ff]")
SENTENCE_END = ".!?…\n"


class GateResult(NamedTuple):
    kept: list[dict]
    counts: dict[str, int]  # counts only, safe to print and copy into results/


def has_long_number(text: str) -> bool:
    """`text` holds 7+ digits in one run that is not a price followed by a currency word."""
    for run in DIGIT_RUN.finditer(text):
        if sum(c.isdigit() for c in run.group()) < MIN_NUMBER_DIGITS:
            continue
        is_price = PRICE.fullmatch(run.group()) and CURRENCY_AFTER.match(text, run.end())
        if not is_price:
            return True
    return False


def contact_left(text: str) -> bool:
    """An email, link with a path or query, @handle, or failed-replacement mark is left in `text`."""
    return any(p.search(text) for p in (EMAIL, LINK_WITH_PATH, HANDLE, PRIVATE_USE))


def _sentence_initial(text: str, start: int) -> bool:
    before = text[:start].rstrip(" \t\u00a0\u202f\"'«(")
    return not before or before[-1] in SENTENCE_END


def _turns(records: Iterable[dict]) -> Iterable[str]:
    return (turn["content"] for record in records for turn in record["messages"])


def _name_noise(texts: list[str], fake_keys: set[str]) -> set[str]:
    """Keys of leftover words capitalised only at sentence starts and also seen in lower case."""
    lower = {compare_key(w) for text in texts for w in LETTERS.findall(text) if w[0].islower()}
    initial_only: dict[str, bool] = {}
    for text in texts:
        for match in leftover_matches(text, fake_keys):
            key = compare_key(match.group())
            initial_only[key] = initial_only.get(key, True) and _sentence_initial(text, match.start())
    return {key for key, only in initial_only.items() if only and key in lower}


def _reasons(texts: list[str], fake_keys: set[str], noise: set[str]) -> set[str]:
    reasons = set()
    for text in texts:
        if has_long_number(text):
            reasons.add("long_number")
        if contact_left(text):
            reasons.add("contact")
        if any(compare_key(m.group()) not in noise for m in leftover_matches(text, fake_keys)):
            reasons.add("leftover_name")
    return reasons


def strict_gate(records: list[dict], fake_names: Iterable[str]) -> GateResult:
    """Records with nothing unresolved, plus per-reason drop counts.

    `fake_names` are the participants' fakes; the whole fake-name pool is allowed too, since
    @mentions get fakes from it, and so are currency words ("Ar"). A conversation dropped for
    two reasons counts under both.
    """
    allowed = [*fake_names, *FIRST_NAMES, *SURNAMES, FALLBACK_NAME, *CURRENCY_WORDS]
    fake_keys = fake_name_keys(allowed)
    noise = _name_noise(list(_turns(records)), fake_keys)
    kept, dropped = [], dict.fromkeys(GATE_REASONS, 0)
    for record in records:
        reasons = _reasons([turn["content"] for turn in record["messages"]], fake_keys, noise)
        for reason in reasons:
            dropped[reason] += 1
        if not reasons:
            kept.append(record)
    counts = {
        "conversations_before_gate": len(records),
        "dropped": len(records) - len(kept),
        **{f"dropped_{reason}": n for reason, n in dropped.items()},
        "kept": len(kept),
    }
    return GateResult(kept, counts)
