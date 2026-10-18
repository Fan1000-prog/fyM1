"""Strict automatic gate run after anonymising: drop any conversation that may still leak.

No person reads the anonymised text. Instead a whole conversation is dropped if any turn holds:

- `long_number`: 7+ digits in one run, counting across separators (space, no-break spaces,
  `. - / ,`, unicode dashes). A price is kept: clean 3-digit groups (or no separator) followed
  by a currency word (`1 500 000 Ar`, `2500000 ariary`, `1 250 000,50 €`). A full date
  (`15/03/2026`, 8 digits) counts as a long number too.
- `contact`: an email, a link with a path or query, an `@handle`, or a private-use mark the
  anonymiser leaves only when a replacement failed.
- `too_many_unknown`: more than 30% of its words would be replaced as names (below).

Capitalised words (not our fake names, placeholders or currency words) are then sorted:

- cleared: in a known-word list (`fym1.data.known_words`), or capitalised only at sentence
  starts and also typed in lower case somewhere in the corpus ("Zazakely tonga." + "ilay zazakely").
- replaced: anything else becomes a fake first name from the anonymiser's pool, the same fake
  for the same word within one conversation, never a word already in that conversation.

A pool name the raw conversation already held ("Rabe no nandeha") is a real person there, not
our fake: the caller passes it in `disallowed` and it is always replaced, word or not.

Lower-case names are replaced too (the same way, never counted as "typed in lower case"):

- after a name particle (`i`, `ry`, `n'i`, `an'i`, `tamin'i`, `ho an'i`): any word that is a pool
  name or that no list knows ("ry hasina", "tamin'i rakoto"); "ry namana", "i dada" stay.
- `ra` + a word no list knows ("rakoto"), and any `Ra-`/`ra-` hyphenated word ("Ra-be").
- a first name that is not also an everyday word ("salut sandrine"; "je t'aime", "une rose" stay),
  and any first name hyphen-joined to another ("jean-pierre", "marie-claire").

Only counts leave this module; never text or words.
"""

import hashlib
import re
from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping
from typing import NamedTuple

from fym1.data.anonymize import EMAIL, FALLBACK_NAME, FIRST_NAMES, SURNAMES, compare_key, fold
from fym1.data.known_words import FIRST_NAME_WORDS, KNOWN_WORDS, WORDLIKE_FIRST_NAMES, Lexicon, known_by
from fym1.data.leftovers import LETTERS, MIN_LETTERS, fake_name_keys

GATE_REASONS = ("long_number", "contact", "too_many_unknown")
SEEN_LOWERCASE = "seen_lowercase"  # cleared: capitalised only at sentence starts, typed in lower case too
MAX_REPLACED_SHARE = 0.3

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

# A word token: letters only, or a hyphenated "Ra-be" taken whole.
TOKEN = re.compile(r"(?<![<\w])(?:[Rr]a-)?[^\W\d_]+(?![\w>])")
NAME_PARTICLE = re.compile(r"(?:(?<![\w'’])(?:i|ry)|\w['’]i)\s+$", re.IGNORECASE)
PARTICLE_WINDOW = 12  # characters looked back for a particle
RA_PREFIX = "ra"
MIN_RA_NAME = 4  # "rabe"
POOL_KEYS = frozenset(compare_key(name) for name in (*FIRST_NAMES, *SURNAMES))
ANY_FIRST_NAME = FIRST_NAME_WORDS | WORDLIKE_FIRST_NAMES
LOWERCASE_NAMES = FIRST_NAME_WORDS - WORDLIKE_FIRST_NAMES  # a name even in lower case
NAME_BEFORE_HYPHEN = re.compile(r"([^\W\d_]+)-$")
NAME_AFTER_HYPHEN = re.compile(r"-([^\W\d_]+)")
HYPHEN_WINDOW = 30  # characters looked back for "jean-" before "pierre"


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


def pool_names_in(texts: Iterable[str]) -> set[str]:
    """Keys of the fake-name pool's names written capitalised in `texts` (raw, before anonymising)."""
    capitalised = (w for text in texts for w in LETTERS.findall(text) if w[0].isupper())
    return {key for w in capitalised if (key := compare_key(w)) in POOL_KEYS}


def _is_capitalised(word: str, fake_keys: set[str]) -> bool:
    return word[0].isupper() and len(word) >= MIN_LETTERS and compare_key(word) not in fake_keys


def _capitalised(text: str, fake_keys: set[str]) -> Iterator[re.Match]:
    return (m for m in TOKEN.finditer(text) if _is_capitalised(m.group(), fake_keys))


def _in_compound_first_name(text: str, match: re.Match) -> bool:
    """`match` is a first name hyphen-joined to another first name ("jean-pierre")."""
    if fold(match.group()) not in ANY_FIRST_NAME:
        return False
    before = NAME_BEFORE_HYPHEN.search(text, max(0, match.start() - HYPHEN_WINDOW), match.start())
    after = NAME_AFTER_HYPHEN.match(text, match.end())
    return any(m and fold(m.group(1)) in ANY_FIRST_NAME for m in (before, after))


def _is_lowercase_name(text: str, match: re.Match, lexicon: Lexicon) -> bool:
    """A lower-case word that is a name by its place or shape (see the module docstring)."""
    word = match.group()
    if not word[0].islower() or len(word) < MIN_LETTERS:
        return False
    if "-" in word:  # "ra-koto"
        return True
    if fold(word) in LOWERCASE_NAMES or _in_compound_first_name(text, match):
        return True
    unknown = known_by(word, lexicon) is None
    if NAME_PARTICLE.search(text, max(0, match.start() - PARTICLE_WINDOW), match.start()):
        return unknown or compare_key(word) in POOL_KEYS
    return unknown and len(word) >= MIN_RA_NAME and fold(word).startswith(RA_PREFIX)


def _seen_lowercase(texts: list[str], fake_keys: set[str], lexicon: Lexicon) -> set[str]:
    """Keys of unknown words capitalised only at sentence starts and also seen in lower case."""
    lower = {
        compare_key(m.group())
        for text in texts
        for m in TOKEN.finditer(text)
        if m.group()[0].islower() and not _is_lowercase_name(text, m, lexicon)
    }
    initial_only: dict[str, bool] = {}
    for text in texts:
        for match in _capitalised(text, fake_keys):
            key = compare_key(match.group())
            if key in lower and known_by(match.group(), lexicon) is None:
                initial_only[key] = initial_only.get(key, True) and _sentence_initial(text, match.start())
    return {key for key, only in initial_only.items() if only}


def _fake_for(key: str, taken: set[str]) -> str:
    """A pool first name picked from `key`, so the same word gets the same fake; none of `taken`."""
    free = [name for name in FIRST_NAMES if compare_key(name) not in taken] or [FALLBACK_NAME]
    number = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")
    return free[number % len(free)]


class _Names(NamedTuple):
    texts: list[str]  # with unknown capitalised words replaced
    words: int
    replaced: int
    cleared: Counter  # list name -> capitalised words it cleared


def _clear_or_replace(
    texts: list[str], fake_keys: set[str], lexicon: Lexicon, seen_lower: set[str], disallowed: set[str]
) -> _Names:
    """One conversation's turns with every unknown or disallowed name swapped for a fake first name."""
    taken = {compare_key(w) for text in texts for w in LETTERS.findall(text)} | disallowed
    fakes: dict[str, str] = {}
    cleared, replaced = Counter(), 0

    def swap(text: str, match: re.Match) -> str:
        nonlocal replaced
        word = match.group()
        key = compare_key(word)
        if not _is_lowercase_name(text, match, lexicon):
            if not _is_capitalised(word, fake_keys):
                return word
            source = None if key in disallowed else known_by(word, lexicon)
            source = source or (SEEN_LOWERCASE if key in seen_lower and key not in disallowed else None)
            if source:
                cleared[source] += 1
                return word
        if key not in fakes:
            fakes[key] = _fake_for(key, taken)
            taken.add(compare_key(fakes[key]))
        replaced += 1
        return fakes[key]

    new_texts = [TOKEN.sub(lambda m, text=text: swap(text, m), text) for text in texts]
    words = sum(len(LETTERS.findall(text)) for text in texts)
    return _Names(new_texts, words, replaced, cleared)


def _with_texts(record: dict, texts: list[str]) -> dict:
    messages = [{**turn, "content": text} for turn, text in zip(record["messages"], texts, strict=True)]
    return {**record, "messages": messages}


def strict_gate(
    records: list[dict],
    fake_names: Iterable[str],
    lexicon: Lexicon = KNOWN_WORDS,
    disallowed: Mapping[str, Collection[str]] | None = None,
) -> GateResult:
    """Records with nothing unresolved and unknown names replaced, plus counts.

    `fake_names` are the participants' fakes; the whole fake-name pool is allowed too, since
    @mentions get fakes from it, and so are currency words ("Ar"). `disallowed` maps a record id
    to pool names its raw text already held: those are real there, so they are replaced (a
    participant's fake that equals one is replaced too, the same way). A conversation dropped for
    two reasons counts under both. `cleared_*` counts capitalised words in every conversation
    before the gate; `tokens_replaced` counts words replaced in kept conversations.
    """
    allowed = [*fake_names, *FIRST_NAMES, *SURNAMES, FALLBACK_NAME, *CURRENCY_WORDS]
    fake_keys = fake_name_keys(allowed)
    disallowed = disallowed or {}
    seen_lower = _seen_lowercase([turn["content"] for r in records for turn in r["messages"]], fake_keys, lexicon)
    kept, dropped, cleared, replaced = [], dict.fromkeys(GATE_REASONS, 0), Counter(), 0
    for record in records:
        texts = [turn["content"] for turn in record["messages"]]
        real_pool = {compare_key(name) for name in disallowed.get(record["id"], ())}
        names = _clear_or_replace(texts, fake_keys - real_pool, lexicon, seen_lower, real_pool)
        cleared += names.cleared
        reasons = _reasons(texts)
        if names.replaced > MAX_REPLACED_SHARE * names.words:
            reasons.add("too_many_unknown")
        for reason in reasons:
            dropped[reason] += 1
        if not reasons:
            kept.append(_with_texts(record, names.texts))
            replaced += names.replaced
    counts = {
        "conversations_before_gate": len(records),
        "dropped": len(records) - len(kept),
        **{f"dropped_{reason}": n for reason, n in dropped.items()},
        "kept": len(kept),
        **{f"cleared_{source}": cleared[source] for source in (*lexicon, SEEN_LOWERCASE)},
        "tokens_replaced": replaced,
    }
    return GateResult(kept, counts)


def _reasons(texts: list[str]) -> set[str]:
    reasons = set()
    for text in texts:
        if has_long_number(text):
            reasons.add("long_number")
        if contact_left(text):
            reasons.add("contact")
    return reasons
