import pytest

from fym1.eval.baseline import (
    accuracy,
    belebele_prompt,
    chrf_pp,
    first_line,
    sample_sentences,
    tokens_per_word,
    translation_prompt,
)

ROW = {
    "flores_passage": "Ny alika dia mihinana.",
    "question": "Inona no ataon'ny alika?",
    "mc_answer1": "Matory",
    "mc_answer2": "Mihinana",
    "mc_answer3": "Milomano",
    "mc_answer4": "Mihira",
}


def test_belebele_prompt_lists_passage_question_and_numbered_options():
    p = belebele_prompt(ROW)
    for part in ["Ny alika dia mihinana.", "Inona no ataon'ny alika?", "1. Matory", "4. Mihira"]:
        assert part in p
    assert p.index("1. Matory") < p.index("2. Mihinana") < p.index("4. Mihira")


def test_accuracy_compares_int_preds_with_string_golds():
    assert accuracy([1, 2, 3, 4], ["1", "2", "4", "4"]) == 0.75


def test_accuracy_rejects_length_mismatch_and_empty():
    with pytest.raises(ValueError):
        accuracy([1], ["1", "2"])
    with pytest.raises(ValueError):
        accuracy([], [])


def test_translation_prompt_names_both_languages():
    p = translation_prompt("Hello.", "English", "Malagasy")
    assert "English" in p and "Malagasy" in p and p.endswith("Hello.")


def test_first_line_skips_blank_lines_and_strips():
    assert first_line("\n\n  Salama tompoko. \nExtra note") == "Salama tompoko."
    assert first_line("   ") == ""


def test_sample_sentences_splits_filters_by_length_and_caps_count():
    texts = ["Fohy. Ity fehezanteny ity dia lava kokoa noho ilay teo aloha.", "Iray roa telo efatra dimy."]
    got = sample_sentences(texts, n=2, min_words=5)
    assert got == ["Ity fehezanteny ity dia lava kokoa noho ilay teo aloha.", "Iray roa telo efatra dimy."]
    assert sample_sentences(texts, n=1, min_words=5) == got[:1]


def test_tokens_per_word_divides_total_tokens_by_total_words():
    count = lambda s: len(s)  # one token per character
    assert tokens_per_word(["ab cd", "efg"], count) == (5 + 3) / 3


def test_chrf_pp_perfect_and_disjoint():
    assert chrf_pp(["Salama tompoko"], ["Salama tompoko"]) == pytest.approx(100.0)
    assert chrf_pp(["xyz"], ["Salama tompoko"]) < 5
