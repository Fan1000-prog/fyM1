"""Automatic metrics over model replies: language drift and training-data leaks."""

import re
import unicodedata
from collections.abc import Iterable

from fym1.eval.lang import is_malagasy

# Plain \w+ words (punctuation and emoji stripped). Unlike lang._WORD this splits on
# apostrophes and keeps digits, which is what exact n-gram matching wants.
_WORD = re.compile(r"\w+")


def _require_responses(responses: list) -> None:
    if not responses:
        raise ValueError("responses is empty; a rate over zero replies is undefined")


def drift_rate(responses: list[str]) -> float:
    """Share of replies that are not Malagasy (full French/English, or no words)."""
    _require_responses(responses)
    return sum(not is_malagasy(r) for r in responses) / len(responses)


def _ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """Distinct word n-grams in order of first appearance (lower-cased, NFC, no punctuation)."""
    words = _WORD.findall(unicodedata.normalize("NFC", text).lower())
    return list(dict.fromkeys(tuple(words[i : i + n]) for i in range(len(words) - n + 1)))


def _train_ngrams(train_texts: Iterable[str], n: int) -> set[tuple[str, ...]]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    train_ngrams: set[tuple[str, ...]] = set()
    for text in train_texts:
        train_ngrams.update(_ngrams(text, n))
    if not train_ngrams:
        # Release gate: a 0.0 from nothing to compare against would pass it unchecked.
        raise ValueError(f"training texts have no word {n}-grams; nothing to check leaks against")
    return train_ngrams


def leak_rate(responses: list[str], train_texts: Iterable[str], n: int = 8) -> float:
    """Share of replies that contain any word n-gram found in the training texts."""
    _require_responses(responses)
    train_ngrams = _train_ngrams(train_texts, n)
    return sum(not train_ngrams.isdisjoint(_ngrams(r, n)) for r in responses) / len(responses)


def leak_matches(responses: list[dict], train_texts: Iterable[str], n: int = 8) -> list[dict]:
    """Each distinct word n-gram that replies share with the training texts.

    `responses` rows are `{"id", "response"}`. Returns `{"ngram", "ids"}` rows in order of
    first appearance; `ids` lists every reply containing it, so each n-gram is reviewed once.
    Normalised as in `leak_rate`, so `leak_rate == 0` exactly when this list is empty.
    """
    _require_responses(responses)
    train_ngrams = _train_ngrams(train_texts, n)
    ids_by_ngram: dict[str, list[str]] = {}
    for row in responses:
        for gram in _ngrams(row["response"], n):
            if gram in train_ngrams:
                ids_by_ngram.setdefault(" ".join(gram), []).append(row["id"])
    return [{"ngram": ngram, "ids": ids} for ngram, ids in ids_by_ngram.items()]
