"""Known words: a capitalised word in one of these lists is ordinary text, not a name.

Lists, checked in this order (the first that knows a word gets the credit in the gate counts):

- `mg_common`: the anonymiser's common Malagasy words that `mg_wiki` also has, and short
  texting words. The common list is ranked by article count whatever the case, so it holds
  first names ("jean", "marie"); keeping only what `mg_wiki` has drops them.
- `mg_wiki`: words Malagasy Wikipedia (CC BY-SA 4.0) types in lower case 3+ times.
- `known_names`: a short hand list of places, brands and religious words (`mg_known_names.txt`).
- `fr`, `en`: the 30,000 most frequent French and English words from `wordfreq` (data CC BY-SA 4.0).

`mg_wiki`, `fr` and `en` are built by scripts/build_known_words.py, which drops words Wikipedia
mostly capitalises mid-sentence ("Marie", "David") and every word in `first_names.txt`.

`first_names.txt` (French first names given 1,000+ times since 1900, INSEE, Licence Ouverte)
is checked before every list: a first name never clears, whatever list has it ("Sandrine",
"Kevin", "Tony"). First names that are everyday Malagasy words ("tia", "mila") are left out of it.

`wordlike_first_names.txt` holds first names that are also everyday words ("aime", "rose",
"claire", "tia"). The gate treats any other first name typed in lower case as a name ("salut sandrine").

`name_deny_list.txt` holds common Malagasy given names that are also words ("Hery", "Soa", "Vola");
`mg_first_names.txt` those that are not ("Mialy", "Njaka"), a name even in lower case.
`PERSON_NAMES` (first names, both Malagasy lists and the anonymiser's fake-name pool) never clear and are
taken out of `mg_wiki`, `fr` and `en` when they are built.

A word matches folded (lower case, no accents). Stretched letters are collapsed only for real
stretching, a letter typed 3+ times ("Bonjouuur" -> "bonjour"), and never when the collapsed word
is a person name: collapsing double letters would let "Anne" match "âne", and "Mariiie" match "marie".
"""

import re
from collections.abc import Iterable
from importlib.resources import files
from typing import NamedTuple

from fym1.data.anonymize import COMMON_WORDS, FALLBACK_NAME, FIRST_NAMES, SURNAMES, TEXTING_WORDS, compare_key, fold

STRETCHED = re.compile(r"(.)\1\1")  # a letter typed 3+ times in a row


class WordList(NamedTuple):
    folded: frozenset[str]
    keys: frozenset[str]  # compare keys, for stretched words only

    def has(self, word: str) -> bool:
        folded = fold(word)
        return folded in self.folded or (is_stretched(folded) and compare_key(folded) in self.keys)


Lexicon = dict[str, WordList]  # list name -> words, in checking order


def word_list(words: Iterable[str]) -> WordList:
    folded = frozenset(fold(w) for w in words)
    return WordList(folded, frozenset(compare_key(w) for w in folded))


def _words(name: str) -> list[str]:
    lines = files("fym1.data").joinpath(name).read_text("utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def is_stretched(folded: str) -> bool:
    """`folded` has a letter typed 3+ times and does not collapse to a person name."""
    return bool(STRETCHED.search(folded)) and compare_key(folded) not in PERSON_NAME_KEYS


FIRST_NAME_WORDS = frozenset(fold(w) for w in _words("first_names.txt"))
NAME_DENY_LIST = frozenset(fold(w) for w in _words("name_deny_list.txt"))
# Malagasy given names that are not everyday words: a name even in lower case ("salut mialy").
MG_FIRST_NAMES = frozenset(fold(w) for w in _words("mg_first_names.txt"))
POOL_NAMES = frozenset(fold(w) for w in (*FIRST_NAMES, *SURNAMES, FALLBACK_NAME))
# Never cleared as a word, whatever list has it.
PERSON_NAMES = FIRST_NAME_WORDS | NAME_DENY_LIST | MG_FIRST_NAMES | POOL_NAMES
PERSON_NAME_KEYS = frozenset(compare_key(w) for w in PERSON_NAMES)
# First names that are also everyday words ("aime", "rose", "tia"): a name only when capitalised.
WORDLIKE_FIRST_NAMES = frozenset(fold(w) for w in _words("wordlike_first_names.txt"))
_MG_WIKI = word_list(_words("mg_wiki_words.txt"))
KNOWN_WORDS: Lexicon = {
    "mg_common": word_list({fold(w) for w in COMMON_WORDS} & _MG_WIKI.folded | TEXTING_WORDS),
    "mg_wiki": _MG_WIKI,
    "known_names": word_list(_words("mg_known_names.txt")),
    "fr": word_list(_words("fr_words.txt")),
    "en": word_list(_words("en_words.txt")),
}


def known_by(word: str, lexicon: Lexicon) -> str | None:
    """Name of the first list in `lexicon` that has `word`, else None (always None for a person name)."""
    if fold(word) in PERSON_NAMES:
        return None
    return next((name for name, words in lexicon.items() if words.has(word)), None)
