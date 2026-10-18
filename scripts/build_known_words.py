"""Word lists the strict gate uses to tell an ordinary capitalised word from a name.

Writes, into src/fym1/data/:
- mg_wiki_words.txt: words typed in lower case at least --min-lower times in Malagasy Wikipedia
  (CC BY-SA 4.0). Proper names are nearly always capitalised there, so they mostly stay out.
- fr_words.txt, en_words.txt: the --top most frequent French and English words from the
  `wordfreq` package (code Apache-2.0, data CC BY-SA 4.0).
- first_names.txt: French first names given at least --min-births times since 1900, from the
  INSEE "fichier des prénoms" (Licence Ouverte / Etalab 2.0), minus everyday Malagasy words
  (typed in lower case MG_WORD_MIN_LOWER+ times in Malagasy Wikipedia: "tia", "mila") and the
  hand list mg_known_names.txt. It holds the English names common in France too (Kevin, Jessica).

Name filter, on all three lists: a word Malagasy Wikipedia writes capitalised mid-sentence at
least NAME_MIN_CAPITALISED times and more than NAME_RATIO times as often as in lower case is
dropped, and so is every word in first_names.txt. wordfreq is lower case only, so without this
"marie", "sandrine" or "kevin" would clear a person's name.

All words are folded (lower case, no accents), letters only, sorted, under one `#` source line.

    uv sync --group data && uv run python scripts/build_known_words.py
"""

import argparse
import csv
import hashlib
import io
import re
import urllib.request
import zipfile
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
WIKIPEDIA_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"  # wikimedia/wikipedia, 2024-01-09
INSEE_URL = "https://www.insee.fr/fr/statistiques/fichier/7633685/nat2022_csv.zip"
INSEE_SHA256 = "c13b45e770c684b3c8531f28b6e3eae62194dc6dc86205fa17e4c6447b01b81d"
MG_WORD_MIN_LOWER = 150
SOURCES = {
    "mg_wiki_words.txt": f"Malagasy Wikipedia (wikimedia/wikipedia 20231101.mg @ {WIKIPEDIA_REVISION[:12]}), "
    "CC BY-SA 4.0",
    "fr_words.txt": "wordfreq top French words (code Apache-2.0, data CC BY-SA 4.0)",
    "en_words.txt": "wordfreq top English words (code Apache-2.0, data CC BY-SA 4.0)",
    "first_names.txt": f"INSEE fichier des prenoms 2022 ({INSEE_URL}), Licence Ouverte / Etalab 2.0",
}


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


def first_names(rows: Iterable[dict], cases: CaseCounts, min_births: int) -> list[str]:
    """Folded INSEE first names given `min_births`+ times, minus everyday Malagasy words."""
    births = Counter()
    for row in rows:
        if not row["preusuel"].startswith("_"):  # "_PRENOMS_RARES" is a bucket, not a name
            births[fold(row["preusuel"])] += int(row["nombre"])
    return sorted(
        name
        for name, n in births.items()
        if n >= min_births and LETTERS.fullmatch(name) and cases.lower[name] < MG_WORD_MIN_LOWER
    )


def insee_rows() -> list[dict]:
    """The INSEE first-name file, checked against its pinned hash."""
    with urllib.request.urlopen(INSEE_URL) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != INSEE_SHA256:
        raise SystemExit(f"{INSEE_URL} changed (sha256 mismatch); check it, then update INSEE_SHA256")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter=";"))


def lower_case_words(cases: CaseCounts, min_count: int, names: set[str]) -> list[str]:
    """Folded words that start lower case at least `min_count` times and are not names."""
    return sorted(
        w for w, n in cases.lower.items() if n >= min_count and w not in names and not looks_like_a_name(w, cases)
    )


def frequent_words(lang: str, top: int, cases: CaseCounts, names: set[str]) -> list[str]:
    """The `top` most frequent `lang` words in wordfreq, folded, letters only, not names."""
    words = {fold(w) for w in top_n_list(lang, top)}
    return sorted(
        w
        for w in words
        if len(w) >= MIN_LETTERS and LETTERS.fullmatch(w) and w not in names and not looks_like_a_name(w, cases)
    )


def write(words: list[str], name: str) -> None:
    path = DATA / name
    path.write_text(f"# {SOURCES[name]}\n" + "\n".join(words) + "\n", encoding="utf-8")
    print(f"{len(words)} words -> {path}")


def hand_listed() -> set[str]:
    lines = (DATA / "mg_known_names.txt").read_text("utf-8").splitlines()
    return {fold(line.strip()) for line in lines if line.strip() and not line.startswith("#")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="20231101.mg", help="wikimedia/wikipedia config")
    parser.add_argument("--min-lower", type=int, default=3)
    parser.add_argument("--top", type=int, default=30_000)
    parser.add_argument("--min-births", type=int, default=1_000)
    args = parser.parse_args()

    articles = load_dataset("wikimedia/wikipedia", args.config, split="train", revision=WIKIPEDIA_REVISION)
    cases = count_cases(a["text"] for a in articles)
    names = [n for n in first_names(insee_rows(), cases, args.min_births) if n not in hand_listed()]
    write(names, "first_names.txt")
    write(lower_case_words(cases, args.min_lower, set(names)), "mg_wiki_words.txt")
    for lang in ("fr", "en"):
        write(frequent_words(lang, args.top, cases, set(names)), f"{lang}_words.txt")


if __name__ == "__main__":
    main()
