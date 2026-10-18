"""Known-word lists and the script that builds them: synthetic text only."""

import re
from importlib.resources import files
from pathlib import Path

from fym1.data.anonymize import FIRST_NAMES, SURNAMES, fold
from fym1.data.known_words import KNOWN_WORDS, NAME_DENY_LIST, known_by, word_list


def test_stretched_letters_collapse_only_when_the_word_has_them():
    words = word_list({"mariée", "bonjour"})

    assert words.has("Mariée") and words.has("MARIEE") and words.has("Bonjouuur")
    assert not words.has("Marie")


def test_known_by_names_the_first_list_that_has_the_word():
    lexicon = {"a": word_list({"tana"}), "b": word_list({"tana", "hello"})}

    assert known_by("Tana", lexicon) == "a"
    assert known_by("Hello", lexicon) == "b"
    assert known_by("Bema", lexicon) is None


def test_every_list_is_loaded_in_checking_order():
    assert list(KNOWN_WORDS) == ["mg_common", "mg_wiki", "mg_corpus", "known_names", "fr", "en"]
    assert all(len(words.folded) > 0 for words in KNOWN_WORDS.values())


def test_build_keeps_lower_case_words_and_drops_names_capitalised_mid_sentence(build_known_words):
    # "Bema" is capitalised mid-sentence 10 times and never lower case: a name
    texts = ["ny trano sy ny saha. Trano tsara."] * 3 + ["hoy i Bema tamin'ny trano"] * 10

    cases = build_known_words.count_cases(texts)

    assert cases.lower["trano"] == 13 and cases.capitalised["trano"] == 0  # sentence start not counted
    assert build_known_words.looks_like_a_name("bema", cases)
    assert not build_known_words.looks_like_a_name("trano", cases)
    words = build_known_words.lower_case_words(cases, min_count=3, names={"saha"})
    assert "trano" in words and "saha" not in words and "bema" not in words


def test_first_names_never_clear_even_when_a_list_has_them():
    lexicon = {"fr": word_list({"sandrine", "trano"})}

    assert known_by("Sandrine", lexicon) is None
    assert known_by("Trano", lexicon) == "fr"


def test_build_takes_frequent_insee_first_names_but_not_malagasy_words(build_known_words):
    rows = [
        {"preusuel": "SANDRINE", "nombre": "900"},
        {"preusuel": "SANDRINE", "nombre": "200"},
        {"preusuel": "TIA", "nombre": "5000"},  # also an everyday Malagasy word
        {"preusuel": "RARENAME", "nombre": "10"},
        {"preusuel": "_PRENOMS_RARES", "nombre": "99999"},
    ]
    cases = build_known_words.count_cases(["tia aho"] * build_known_words.MG_WORD_MIN_LOWER)

    names = build_known_words.first_names(rows, cases, min_births=1000)

    assert names == ["sandrine"]


BUILD_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_known_words.py"


def _first_line(name):
    return files("fym1.data").joinpath(name).read_text("utf-8").splitlines()[0]


def test_every_word_list_file_starts_with_a_source_line():
    for name in ("mg_wiki_words.txt", "mg_corpus_words.txt", "fr_words.txt", "en_words.txt", "first_names.txt", "mg_known_names.txt"):
        assert _first_line(name).startswith("# "), name


def test_build_pins_the_wikipedia_revision():
    """Read from the script text, so this runs without the `data` group the script imports."""
    pinned = re.search(r'^WIKIPEDIA_REVISION = "([0-9a-f]+)"', BUILD_SCRIPT.read_text("utf-8"), re.MULTILINE)

    assert pinned and len(pinned.group(1)) == 40
    hplt = re.search(r'^HPLT_REVISION = "([0-9a-f]+)"', BUILD_SCRIPT.read_text("utf-8"), re.MULTILINE)
    assert hplt and len(hplt.group(1)) == 40


def test_build_keeps_first_names_a_wikipedia_uses_as_words_and_not_names(build_known_words):
    fr = build_known_words.count_cases(["une rose rouge, une Rose."] * 5 + ["hoy i Sandrine sy i Pierre"] * 10)
    mg = build_known_words.count_cases(["tia azy aho"] * 5)
    vocabularies = {"fr": {"rose", "sandrine", "pierre", "tia"}, "mg": set(mg.lower)}

    words = build_known_words.wordlike_first_names(
        ["rose", "sandrine", "pierre", "tia", "kevin"], vocabularies, {"fr": fr, "mg": mg}, min_lower=3
    )

    assert words == ["rose", "tia"]


def test_wordlike_first_names_file_starts_with_a_source_line():
    assert _first_line("wordlike_first_names.txt").startswith("# ")


def test_corpus_words_keep_the_most_frequent_lower_case_words_and_never_names(build_known_words):
    texts = (
        ["tsara be ilay izy"] * 9  # "be" has 2 letters: kept like any word
        + ["mbola tsy vita"] * 4
        + ["hoy i Bema, i Bema, i Bema"] * 3  # capitalised mid-sentence, never lower case: a name
        + ["bema bema"]  # lower case twice only: below --min-lower
        + ["salut sandrine"] * 5  # a first name, typed in lower case often
        + ["vary"] * 2  # too rare
    )
    cases = build_known_words.count_cases(texts)

    words = build_known_words.corpus_words(cases, min_lower=3, names={"sandrine"}, top=5)

    assert words == sorted(["tsara", "be", "ilay", "izy", "salut"])  # top 5 by count; ties by word
    assert "bema" not in build_known_words.corpus_words(cases, min_lower=2, names=set(), top=100)


def test_web_addresses_are_cut_before_counting_so_names_in_them_never_become_words(build_known_words):
    text = "jereo www.example.mg/fanja sy fb.com/rakoto.jean na fanja@mail.mg na @njaka https://x.io/koto ok"

    left = build_known_words.without_web_addresses(text).split()

    assert left == ["jereo", "sy", "na", "na", "ok"]


def test_corpus_words_drop_names_web_text_also_types_in_lower_case(build_known_words):
    # "riana" is a word and a given name: 10 times lower case, 4 capitalised mid-sentence (0.4)
    texts = ["ny riana sy ny trano"] * 10 + ["hoy i Riana, i Riana"] * 2 + ["tonga ny Trano"]
    cases = build_known_words.count_cases(texts)

    words = build_known_words.corpus_words(cases, min_lower=3, names=set(), top=100)

    assert not build_known_words.looks_like_a_name("riana", cases)  # the Wikipedia rule alone keeps it
    assert "riana" not in words and "trano" in words  # "trano": 1 capitalised in 10 (0.1)


WORD_LIST_FILES = (
    "mg_common_words.txt", "mg_wiki_words.txt", "mg_corpus_words.txt", "mg_known_names.txt", "fr_words.txt",
    "en_words.txt",
)  # fmt: skip


def test_a_pool_name_a_word_list_holds_is_on_the_name_deny_list():
    """A fake-pool name is a name everywhere the gate looks, so an everyday word in the pool is masked even in
    lower case unless the deny list keeps it ("soa ve"). The pool's word-names (Soa, Vola, Mamy) are on that
    list; any other pool name must be in no word list. Read from the shipped files: build_known_words.py
    drops pool names from its lists on a rebuild, which would hide the clash."""
    listed = set()
    for name in WORD_LIST_FILES:
        lines = files("fym1.data").joinpath(name).read_text("utf-8").splitlines()
        listed |= {fold(line.strip()) for line in lines if line.strip() and not line.startswith("#")}
    clashes = {fold(name) for name in (*FIRST_NAMES, *SURNAMES)} & listed
    assert clashes <= NAME_DENY_LIST, sorted(clashes - NAME_DENY_LIST)
