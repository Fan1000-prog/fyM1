"""Known-word lists and the script that builds them: synthetic text only."""

import importlib.util
from pathlib import Path

from fym1.data.known_words import KNOWN_WORDS, known_by, word_list

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_known_words.py"
spec = importlib.util.spec_from_file_location("build_known_words", SCRIPT)
build_known_words = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_known_words)


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


def test_build_keeps_lower_case_words_and_drops_names_capitalised_mid_sentence():
    # "Bema" is capitalised mid-sentence 10 times and never lower case: a name
    texts = ["ny trano sy ny saha. Trano tsara."] * 3 + ["hoy i Bema tamin'ny trano"] * 10

    cases = build_known_words.count_cases(texts)

    assert cases.lower["trano"] == 13 and cases.capitalised["trano"] == 0  # sentence start not counted
    assert build_known_words.looks_like_a_name("bema", cases)
    assert not build_known_words.looks_like_a_name("trano", cases)
    words = build_known_words.lower_case_words(cases, min_count=3)
    assert "trano" in words and "saha" in words and "bema" not in words
