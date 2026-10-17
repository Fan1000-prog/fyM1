import pytest

from fym1.eval.metrics import drift_rate, leak_rate


def test_drift_rate_counts_non_malagasy_share():
    assert drift_rate(["Salama!", "Hello there"]) == 0.5


def test_drift_rate_code_mixed_is_not_drift():
    assert drift_rate(["Mety be zany, lancer-na fotsiny😂", "De ahoana?"]) == 0.0


def test_drift_rate_rejects_empty_list():
    with pytest.raises(ValueError):
        drift_rate([])


def test_leak_rate_finds_shared_8gram():
    assert leak_rate(["a b c d e f g h"], ["x a b c d e f g h y"]) == 1.0


def test_leak_rate_too_short_for_n():
    assert leak_rate(["a b c"], ["a b c"]) == 0.0


def test_leak_rate_ignores_case_and_punctuation():
    assert leak_rate(["A, b! c d e f g h."], ["a b c d e f g h"]) == 1.0


def test_leak_rate_share_of_responses():
    responses = ["a b c d e f g h", "z y x w v u t s"]
    assert leak_rate(responses, ["a b c d e f g h"]) == 0.5


def test_leak_rate_custom_n():
    assert leak_rate(["a b c"], ["a b c"], n=3) == 1.0


def test_leak_rate_rejects_empty_list():
    with pytest.raises(ValueError):
        leak_rate([], ["a b c d e f g h"])


def test_leak_rate_rejects_bad_n():
    with pytest.raises(ValueError):
        leak_rate(["a"], ["a"], n=0)
