"""Known-word lists and the script that builds them: synthetic text only."""

from pathlib import Path

from fym1.data.known_words import KNOWN_WORDS, known_by, word_list


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
    assert list(KNOWN_WORDS) == ["mg_common", "mg_wiki", "known_names", "fr", "en"]
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


def test_every_word_list_file_starts_with_a_source_line(build_known_words):
    for name in ("mg_wiki_words.txt", "fr_words.txt", "en_words.txt", "first_names.txt", "mg_known_names.txt"):
        first = (Path(build_known_words.DATA) / name).read_text("utf-8").splitlines()[0]
        assert first.startswith("# "), name


def test_build_pins_the_wikipedia_revision(build_known_words):
    assert len(build_known_words.WIKIPEDIA_REVISION) == 40


def test_build_keeps_first_names_a_wikipedia_uses_as_words_and_not_names(build_known_words):
    fr = build_known_words.count_cases(["une rose rouge, une Rose."] * 5 + ["hoy i Sandrine sy i Pierre"] * 10)
    mg = build_known_words.count_cases(["tia azy aho"] * 5)
    vocabularies = {"fr": {"rose", "sandrine", "pierre", "tia"}, "mg": set(mg.lower)}

    words = build_known_words.wordlike_first_names(
        ["rose", "sandrine", "pierre", "tia", "kevin"], vocabularies, {"fr": fr, "mg": mg}, min_lower=3
    )

    assert words == ["rose", "tia"]


def test_wordlike_first_names_file_starts_with_a_source_line(build_known_words):
    first = (Path(build_known_words.DATA) / "wordlike_first_names.txt").read_text("utf-8").splitlines()[0]
    assert first.startswith("# ")
