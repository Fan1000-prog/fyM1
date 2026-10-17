import pytest

from fym1.eval.baseline import (
    accuracy,
    belebele_completion_prompt,
    belebele_prompt,
    chrf_pp,
    missing_sections,
    first_line,
    loss_trend,
    sample_sentences,
    selection_row,
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


def test_selection_row_maps_baseline_json_to_paper_schema():
    result = {
        "model": {"slug": "m", "hub_id": "org/m"},
        "belebele": {"acc": 0.5},
        "belebele_completion": {"acc": 0.6},
        "flores": {"eng-plt_chrf++": 40.0, "plt-eng_chrf++": 50.0},
        "tokens_per_word": {"value": 2.1},
        "qlora_smoke": {"ok": True, "peak_vram_gb": 12.0, "sec_per_step": 30.0},
    }
    assert selection_row(result, "Apache-2.0") == {
        "model": "org/m", "license": "Apache-2.0", "belebele_acc": 0.6, "belebele_protocol": "completion 0-shot",
        "belebele_chat_acc": 0.5,
        "chrf_plt_eng": 50.0, "chrf_eng_plt": 40.0, "flores": "measured", "tokens_per_word": 2.1,
        "sniff_natural": None, "peak_vram_gb": 12.0, "sec_per_step": 30.0, "qlora_fits": True,
        "loss_trend": None,
    }


def test_selection_row_leaves_missing_sections_none():
    row = selection_row({"model": {"slug": "m", "hub_id": "org/m"}, "skipped": "403"}, "x")
    assert row["belebele_acc"] is None and row["chrf_plt_eng"] is None and row["qlora_fits"] is None
    assert row["flores"] == "pending"


def test_selection_row_base_model_has_no_chat_column():
    result = {"model": {"slug": "m", "hub_id": "org/m"}, "belebele": {"acc": 0.24}}
    row = selection_row(result, "x", base_model=True)
    assert row["belebele_acc"] is None and row["belebele_chat_acc"] is None
    assert row["belebele_protocol"] == "completion 0-shot (pending)"


def test_belebele_completion_prompt_is_plain_text_ending_in_answer():
    p = belebele_completion_prompt(ROW)
    assert p.startswith("P: Ny alika dia mihinana.\nQ: Inona no ataon'ny alika?\n")
    assert "\nA: Matory\nB: Mihinana\nC: Milomano\nD: Mihira\n" in p
    assert p.endswith("Answer:")


def test_missing_sections_completion_for_all_chat_for_instruct_only():
    full = {"belebele_completion": {}, "flores": {}, "tokens_per_word": {}, "greetings": {}, "qlora_smoke": {}}
    assert missing_sections(full, base_model=True) == []
    assert missing_sections(full, base_model=False) == ["belebele"]
    assert missing_sections({"belebele": {}}, base_model=False) == [
        "belebele_completion", "flores", "tokens_per_word", "greetings", "qlora_smoke"]


def test_missing_sections_skipped_model_is_complete():
    assert missing_sections({"skipped": "403"}, base_model=False) == []


def test_loss_trend_from_curve_or_first_last():
    assert loss_trend({"loss_curve": [6.0, 3.0, 4.0, 5.0]}) == {
        "first": 6.0, "min": 3.0, "min_step": 2, "last": 5.0, "rising_at_end": True}
    assert loss_trend({"loss_first": 1.1, "loss_last": 0.7}) == {"first": 1.1, "last": 0.7}
    assert loss_trend({}) is None
