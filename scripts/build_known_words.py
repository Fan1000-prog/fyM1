"""Word lists the strict gate uses to tell an ordinary capitalised word from a name.

Writes, into src/fym1/data/:
- mg_wiki_words.txt: words typed in lower case at least --min-lower times in Malagasy Wikipedia
  (CC BY-SA 4.0). Proper names are nearly always capitalised there, so they mostly stay out.
- fr_words.txt, en_words.txt: the --top most frequent French and English words from the
  `wordfreq` package (code Apache-2.0, data CC BY-SA 4.0).

Name filter, on all three lists: a word Malagasy Wikipedia writes capitalised mid-sentence at
least NAME_MIN_CAPITALISED times and more than NAME_RATIO times as often as in lower case is
dropped. wordfreq is lower case only, so without this "marie", "jean" or "david" would clear a
person's name.

All words are folded (lower case, no accents), letters only, sorted.

    uv sync --group data && uv run python scripts/build_known_words.py
"""

import argparse
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from datasets import load_dataset
from wordfreq import top_n_list

from fym1.data.anonymize import fold

DATA = Path(__file__).resolve().parents[1] / "src/fym1/data"
LETTERS = re.compile(r"[^\W\d_]+")
MIN_LETTERS = 2
SENTENCE_END = ".!?…\n:"
NAME_MIN_CAPITALISED = 10
NAME_RATIO = 3


class CaseCounts(NamedTuple):
    lower: Counter  # folded word -> times it starts lower case
    capitalised: Counter  # folded word -> times it starts upper case, not at a sentence start


def count_cases(texts: Iterable[str]) -> CaseCounts:
    lower, capitalised = Counter(), Counter()
    for text in texts:
        for match in LETTERS.finditer(text):
            word = match.group()
            if len(word) < MIN_LETTERS:
                continue
            if word[0].islower():
                lower[fold(word)] += 1
                continue
            before = text[: match.start()].rstrip(" \t\"'«(")[-1:]
            if before and before not in SENTENCE_END:
                capitalised[fold(word)] += 1
    return CaseCounts(lower, capitalised)


def looks_like_a_name(word: str, cases: CaseCounts) -> bool:
    capitalised = cases.capitalised[word]
    return capitalised >= NAME_MIN_CAPITALISED and capitalised > NAME_RATIO * cases.lower[word]


def lower_case_words(cases: CaseCounts, min_count: int) -> list[str]:
    """Folded words that start lower case at least `min_count` times and do not look like names."""
    return sorted(w for w, n in cases.lower.items() if n >= min_count and not looks_like_a_name(w, cases))


def frequent_words(lang: str, top: int, cases: CaseCounts) -> list[str]:
    """The `top` most frequent `lang` words in wordfreq, folded, letters only, not names."""
    words = {fold(w) for w in top_n_list(lang, top)}
    return sorted(
        w for w in words if len(w) >= MIN_LETTERS and LETTERS.fullmatch(w) and not looks_like_a_name(w, cases)
    )


def write(words: list[str], name: str) -> None:
    path = DATA / name
    path.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"{len(words)} words -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="20231101.mg", help="wikimedia/wikipedia config")
    parser.add_argument("--min-lower", type=int, default=3)
    parser.add_argument("--top", type=int, default=30_000)
    args = parser.parse_args()

    articles = load_dataset("wikimedia/wikipedia", args.config, split="train")
    cases = count_cases(a["text"] for a in articles)
    write(lower_case_words(cases, args.min_lower), "mg_wiki_words.txt")
    for lang in ("fr", "en"):
        write(frequent_words(lang, args.top, cases), f"{lang}_words.txt")


if __name__ == "__main__":
    main()
