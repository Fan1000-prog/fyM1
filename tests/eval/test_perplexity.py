"""Held-out split and response-only perplexity. Synthetic text only."""

import math
import random

import pytest

from fym1.eval.perplexity import held_out_rows, perplexity, response_spans, token_mask

USER, BOT = "<u>", "<a>"


def test_held_out_rows_match_the_training_split():
    rows = [{"id": i} for i in range(100)]
    expected = list(rows)
    random.Random(42).shuffle(expected)  # what 01_train does inline
    assert held_out_rows(rows, 0.02, 42) == expected[:2]
    assert rows == [{"id": i} for i in range(100)]  # input not shuffled in place


def test_held_out_rows_keep_at_least_one_and_refuse_empty():
    assert len(held_out_rows([1, 2, 3], 0.02, 42)) == 1
    with pytest.raises(ValueError):
        held_out_rows([], 0.02, 42)


def test_response_spans_cover_every_reply_and_stop_at_the_next_user_turn():
    text = f"sys{USER}hi{BOT}salama{USER}ok{BOT}veloma"
    spans = response_spans(text, USER, BOT)
    assert [text[a:b] for a, b in spans] == ["salama", "veloma"]


def test_response_spans_empty_without_a_reply_and_refuse_empty_markers():
    assert response_spans(f"{USER}hi", USER, BOT) == []
    with pytest.raises(ValueError):
        response_spans("x", "", BOT)


def test_token_mask_scores_reply_tokens_only_and_never_padding():
    spans = [(10, 20)]
    offsets = [(0, 5), (10, 14), (14, 20), (20, 22), (0, 0)]
    assert token_mask(offsets, spans) == [False, True, True, False, False]


def test_perplexity_is_exp_mean_nll():
    assert perplexity(2 * math.log(4), 2) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        perplexity(1.0, 0)
