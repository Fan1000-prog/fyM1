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

A word matches folded (lower case, no accents). Stretched letters are collapsed only when the
word has some ("Bonjouuur" -> "bonjour"): collapsing always would let "Marie" match "mariée".
"""

from collections.abc import Iterable
from importlib.resources import files
from typing import NamedTuple

from fym1.data.anonymize import COMMON_WORDS, TEXTING_WORDS, compare_key, fold


class WordList(NamedTuple):
    folded: frozenset[str]
    keys: frozenset[str]  # compare keys, for stretched words only

    def has(self, word: str) -> bool:
        folded, key = fold(word), compare_key(word)
        return folded in self.folded or (key != folded and key in self.keys)


Lexicon = dict[str, WordList]  # list name -> words, in checking order


def word_list(words: Iterable[str]) -> WordList:
    folded = frozenset(fold(w) for w in words)
    return WordList(folded, frozenset(compare_key(w) for w in folded))


def _words(name: str) -> list[str]:
    lines = files("fym1.data").joinpath(name).read_text("utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


FIRST_NAME_WORDS = frozenset(fold(w) for w in _words("first_names.txt"))
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
    """Name of the first list in `lexicon` that has `word`, else None (always None for a first name)."""
    if fold(word) in FIRST_NAME_WORDS:
        return None
    return next((name for name, words in lexicon.items() if words.has(word)), None)
