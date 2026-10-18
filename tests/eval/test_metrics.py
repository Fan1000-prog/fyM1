import pytest

from fym1.eval.metrics import drift_rate, leak_matches, leak_rate


def test_drift_rate_counts_non_malagasy_share():
    assert drift_rate(["Salama!", "Hello there"]) == 0.5


def test_drift_rate_code_mixed_is_not_drift():
    assert drift_rate(["Mety be zany, lancer-na fotsiny😂", "De ahoana?"]) == 0.0


def test_drift_rate_rejects_empty_list():
    with pytest.raises(ValueError):
        drift_rate([])


def test_leak_rate_finds_shared_8gram():
    assert leak_rate(["a b c d e f g h"], ["x a b c d e f g h y"]) == 1.0


def test_leak_rate_response_too_short_for_n():
    assert leak_rate(["a b c"], ["a b c d e f g h"]) == 0.0


def test_leak_rate_ignores_case_and_punctuation():
    assert leak_rate(["A, b! c d e f g h."], ["a b c d e f g h"]) == 1.0


def test_leak_rate_normalises_accents():
    decomposed = "be\u0301 a b c d e f g"
    assert leak_rate([decomposed], ["bé a b c d e f g"]) == 1.0


def test_leak_rate_share_of_responses():
    responses = ["a b c d e f g h", "z y x w v u t s"]
    assert leak_rate(responses, ["a b c d e f g h"]) == 0.5


def test_leak_rate_custom_n():
    assert leak_rate(["a b c"], ["a b c"], n=3) == 1.0


def test_leak_rate_rejects_empty_list():
    with pytest.raises(ValueError):
        leak_rate([], ["a b c d e f g h"])


@pytest.mark.parametrize("train_texts", [[], ["a b c"]])
def test_leak_rate_rejects_training_texts_without_ngrams(train_texts):
    # A silent 0.0 here would pass the release gate without checking anything.
    with pytest.raises(ValueError, match="no word 8-grams"):
        leak_rate(["a b c d e f g h"], train_texts)


def test_leak_rate_accepts_a_generator_of_training_texts():
    assert leak_rate(["a b c d e f g h"], (t for t in ["a b c d e f g h"])) == 1.0


def test_leak_rate_rejects_bad_n():
    with pytest.raises(ValueError):
        leak_rate(["a"], ["a"], n=0)


def test_leak_matches_lists_each_distinct_ngram_with_its_answer_ids():
    responses = [
        {"id": "p1", "response": "A b c d e f g h i!"},
        {"id": "p2", "response": "z y x w v u t s"},
        {"id": "p3", "response": "a b c d e f g h"},
    ]
    assert leak_matches(responses, ["a b c d e f g h i j"]) == [
        {"ngram": "a b c d e f g h", "ids": ["p1", "p3"]},
        {"ngram": "b c d e f g h i", "ids": ["p1"]},
    ]


def test_leak_matches_lists_an_answer_once_per_ngram():
    responses = [{"id": "p1", "response": "a b c d e f g h a b c d e f g h"}]
    assert leak_matches(responses, ["a b c d e f g h"]) == [
        {"ngram": "a b c d e f g h", "ids": ["p1"]}
    ]


def test_leak_matches_agrees_with_leak_rate_on_no_leak():
    responses = [{"id": "p1", "response": "z y x w v u t s"}]
    assert leak_matches(responses, ["a b c d e f g h"]) == []


def test_leak_matches_rejects_training_texts_without_ngrams():
    with pytest.raises(ValueError, match="no word 8-grams"):
        leak_matches([{"id": "p1", "response": "a b c d e f g h"}], ["a b c"])


def test_leak_matches_rejects_empty_responses():
    with pytest.raises(ValueError):
        leak_matches([], ["a b c d e f g h"])


def test_ngram_index_matches_leak_rate_normalisation():
    from fym1.eval.metrics import NgramIndex

    index = NgramIndex(["Ny alika mainty dia nandry teo ambanin'ny latabatra lehibe omaly"], n=8)
    assert index.shares_with("hoy izy: NY ALIKA mainty, dia nandry teo ambanin'ny latabatra!")
    assert not index.shares_with("ny alika mainty dia nandry teo")
    with pytest.raises(ValueError):
        NgramIndex(["too short"], n=8)
