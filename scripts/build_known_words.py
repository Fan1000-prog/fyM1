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
- wordlike_first_names.txt: INSEE first names that are also everyday words, so the gate leaves them
  in lower case ("je t'aime", "une rose"; "salut sandrine" is still a name). A name is a word if a
  language has it (wordfreq top --top French or English, or Malagasy Wikipedia lower case
  --min-lower+ times) and that language's Wikipedia types it in lower case --min-lower+ times and
  does not mostly capitalise it (the name filter below). French and English Wikipedia are read as
  the first --wiki-sample articles of the same pinned revision. "pierre" and "max" stay names.

- mg_corpus_words.txt: the --corpus-top words Malagasy web text (HPLT 2.0 cleaned, plt_Latn, CC0 1.0)
  types in lower case most often, at least --min-lower times. Wikipedia is formal; web text adds the
  everyday and texting spellings people chat in. Read row group by row group (streamed, nothing kept
  on disk). Web addresses, emails and @handles are cut before counting: names hide there in lower case.
  Web text types given names that are also words in lower case far more than Wikipedia ("riana",
  "nambinina"), so a stricter name rule applies on top of the one below: a word is kept only if web text
  capitalises it mid-sentence at most CORPUS_MAX_CAPITALISED (0.25) times as often as in lower case.
  Everyday words sit far below that ("trano" 0.04, "tsara" 0.03); those names above it (0.4-1.3).

Name filter, on all four lists: every person name is dropped (`name_list`: first_names.txt, the
anonymiser's fake-name pool, the Malagasy name deny list name_deny_list.txt and mg_first_names.txt), and so is a word
that Wikipedia writes capitalised mid-sentence at least NAME_MIN_CAPITALISED times and more than
NAME_RATIO times as often as in lower case in Malagasy Wikipedia. (French and English Wikipedia are not
used for this: titles capitalise everyday words there, and "Bonjour" or "Hello" would be dropped.)
wordfreq is lower case only, so without this "marie", "sandrine" or "kevin" would clear a person's name.

NAME_MIN_CAPITALISED is 3, the same floor as --min-lower for a word: a name seen capitalised 3 times
and never in lower case is a name. (It was 10, which let rarer names through.)

All words are folded (lower case, no accents), letters only, sorted, under one `#` source line.

    uv sync --group data && uv run python scripts/build_known_words.py
    uv run python scripts/build_known_words.py --corpus-only   # only mg_corpus_words.txt
"""

import argparse
import csv
import hashlib
import io
import re
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from wordfreq import top_n_list

from fym1.data.anonymize import FALLBACK_NAME, FIRST_NAMES, SURNAMES, fold
from fym1.data.public_sources import HPLT_DATASET, HPLT_FILES, HPLT_REVISION, WIKIPEDIA_REVISION

DATA = Path(__file__).resolve().parents[1] / "src/fym1/data"
LETTERS = re.compile(r"[^\W\d_]+")
MIN_LETTERS = 2
SENTENCE_END = ".!?…\n:"
NAME_MIN_CAPITALISED = 3  # as low as the lower-case floor for a word (--min-lower)
NAME_RATIO = 3
INSEE_URL = "https://www.insee.fr/fr/statistiques/fichier/7633685/nat2022_csv.zip"
INSEE_SHA256 = "c13b45e770c684b3c8531f28b6e3eae62194dc6dc86205fa17e4c6447b01b81d"
MG_WORD_MIN_LOWER = 150
CORPUS_MAX_CAPITALISED = 0.25  # mid-sentence capitalised / lower case, at most, for a web-text word
HPLT_REPO = f"datasets/{HPLT_DATASET}"
# A link, a bare domain, an email or an @handle, as one non-space run: "fb.com/rakoto.jean", "x@y.mg".
WEB_ADDRESS = re.compile(r"\S*(?:https?://|www\.|@|\.(?:com|net|org|mg|fr|io|me)\b)\S*", re.IGNORECASE)
SOURCES = {
    "mg_wiki_words.txt": f"Malagasy Wikipedia (wikimedia/wikipedia 20231101.mg @ {WIKIPEDIA_REVISION[:12]}), "
    "CC BY-SA 4.0",
    "fr_words.txt": "wordfreq top French words (code Apache-2.0, data CC BY-SA 4.0)",
    "en_words.txt": "wordfreq top English words (code Apache-2.0, data CC BY-SA 4.0)",
    "wordlike_first_names.txt": "INSEE first names (Licence Ouverte / Etalab 2.0) that wordfreq (CC BY-SA 4.0) "
    f"or Malagasy Wikipedia has as words and French/English/Malagasy Wikipedia @ {WIKIPEDIA_REVISION[:12]} "
    "(CC BY-SA 4.0) types mostly in lower case",
    "mg_corpus_words.txt": f"HPLT 2.0 cleaned, Malagasy plt_Latn (HPLT/HPLT2.0_cleaned @ {HPLT_REVISION[:12]}), "
    "CC0 1.0",
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


def insee_names(rows: Iterable[dict], min_births: int) -> list[str]:
    """Folded INSEE first names given `min_births`+ times, letters only."""
    births = Counter()
    for row in rows:
        if not row["preusuel"].startswith("_"):  # "_PRENOMS_RARES" is a bucket, not a name
            births[fold(row["preusuel"])] += int(row["nombre"])
    return sorted(name for name, n in births.items() if n >= min_births and LETTERS.fullmatch(name))


def first_names(rows: Iterable[dict], cases: CaseCounts, min_births: int) -> list[str]:
    """Folded INSEE first names given `min_births`+ times, minus everyday Malagasy words."""
    return [name for name in insee_names(rows, min_births) if cases.lower[name] < MG_WORD_MIN_LOWER]


def name_list(first: Iterable[str]) -> set[str]:
    """Every person name kept out of the word lists: `first` names, the fake-name pool, both Malagasy lists."""
    pool = {fold(name) for name in (*FIRST_NAMES, *SURNAMES, FALLBACK_NAME)}
    return set(first) | pool | read_words("name_deny_list.txt") | read_words("mg_first_names.txt")


def is_word_in(word: str, cases: CaseCounts, min_lower: int) -> bool:
    """`cases` (one language's Wikipedia) types `word` in lower case `min_lower`+ times, not mostly capitalised."""
    return cases.lower[word] >= min_lower and not looks_like_a_name(word, cases)


def wordlike_first_names(
    names: Iterable[str], vocabularies: dict[str, set[str]], cases: dict[str, CaseCounts], min_lower: int
) -> list[str]:
    """`names` that some language's vocabulary has and its Wikipedia `cases` show used as a word."""
    return sorted(
        name
        for name in names
        if any(name in words and is_word_in(name, cases[lang], min_lower) for lang, words in vocabularies.items())
    )


def wikipedia_cases(config: str, sample: int | None = None) -> CaseCounts:
    """Case counts over a pinned Wikipedia; only its first `sample` articles when given (streamed)."""
    articles = load_dataset(
        "wikimedia/wikipedia", config, split="train", revision=WIKIPEDIA_REVISION, streaming=sample is not None
    )
    return count_cases(a["text"] for a in (articles if sample is None else articles.take(sample)))


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


def without_web_addresses(text: str) -> str:
    return WEB_ADDRESS.sub(" ", text)


def corpus_texts() -> Iterator[str]:
    """Malagasy web text from the pinned HPLT files, one parquet row group at a time, web addresses cut."""
    fs = HfFileSystem()
    for name in HPLT_FILES:
        with fs.open(f"{HPLT_REPO}@{HPLT_REVISION}/{name}", block_size=16 << 20) as file:
            parquet = pq.ParquetFile(file)
            for group in range(parquet.metadata.num_row_groups):
                for text in parquet.read_row_group(group, columns=["text"]).column("text").to_pylist():
                    yield without_web_addresses(text)


def corpus_words(cases: CaseCounts, min_lower: int, names: set[str], top: int) -> list[str]:
    """The `top` most frequent `lower_case_words` that are rarely capitalised mid-sentence, sorted."""
    words = [
        w
        for w in lower_case_words(cases, min_lower, names)
        if cases.capitalised[w] <= CORPUS_MAX_CAPITALISED * cases.lower[w]
    ]
    return sorted(sorted(words, key=lambda w: (-cases.lower[w], w))[:top])


def write(words: list[str], name: str) -> None:
    path = DATA / name
    path.write_text(f"# {SOURCES[name]}\n" + "\n".join(words) + "\n", encoding="utf-8")
    print(f"{len(words)} words -> {path}")


def read_words(name: str) -> set[str]:
    lines = (DATA / name).read_text("utf-8").splitlines()
    return {fold(line.strip()) for line in lines if line.strip() and not line.startswith("#")}


def hand_listed() -> set[str]:
    return read_words("mg_known_names.txt")


def write_corpus_words(names: set[str], args: argparse.Namespace) -> None:
    cases = count_cases(corpus_texts())
    write(corpus_words(cases, args.min_lower, names, args.corpus_top), "mg_corpus_words.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="20231101.mg", help="wikimedia/wikipedia config")
    parser.add_argument("--min-lower", type=int, default=3)
    parser.add_argument("--top", type=int, default=30_000)
    parser.add_argument("--min-births", type=int, default=1_000)
    parser.add_argument("--wiki-sample", type=int, default=100_000, help="French/English Wikipedia articles read")
    parser.add_argument("--corpus-top", type=int, default=50_000, help="most frequent web-text words kept")
    parser.add_argument("--corpus-only", action="store_true", help="rebuild mg_corpus_words.txt only")
    args = parser.parse_args()

    if args.corpus_only:
        write_corpus_words(name_list(read_words("first_names.txt")), args)
        return

    cases = wikipedia_cases(args.config)
    all_cases = {"mg": cases, **{lang: wikipedia_cases(f"20231101.{lang}", args.wiki_sample) for lang in ("fr", "en")}}
    rows = insee_rows()
    first = [n for n in first_names(rows, cases, args.min_births) if n not in hand_listed()]
    write(first, "first_names.txt")
    names = name_list(first)
    write(lower_case_words(cases, args.min_lower, names), "mg_wiki_words.txt")
    for lang in ("fr", "en"):
        write(frequent_words(lang, args.top, cases, names), f"{lang}_words.txt")

    write_corpus_words(names, args)

    vocabularies = {lang: {fold(w) for w in top_n_list(lang, args.top)} for lang in ("fr", "en")}
    vocabularies["mg"] = set(cases.lower)
    wordlike = wordlike_first_names(insee_names(rows, args.min_births), vocabularies, all_cases, args.min_lower)
    write(wordlike, "wordlike_first_names.txt")


if __name__ == "__main__":
    main()
