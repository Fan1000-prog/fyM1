import pytest

from fym1.eval.arena import tally, wilson_interval

A_IS = {"p1": "fym1", "p2": "base"}


def vote(pair_id, choice, session_id="s", wrong=()):
    return {"pair_id": pair_id, "choice": choice, "wrong": list(wrong), "session_id": session_id}


def test_wilson_interval_known_value():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-4)
    assert high == pytest.approx(0.9433, abs=1e-4)


def test_wilson_interval_stays_inside_zero_one():
    assert wilson_interval(0, 5)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson_interval(5, 5)[1] == pytest.approx(1.0, abs=1e-12)


def test_wilson_interval_needs_trials():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_tally_maps_sides_to_models():
    votes = [
        vote("p1", "A", "s1"),  # A is fym1 -> fym1 wins
        vote("p2", "B", "s1"),  # B is fym1 -> fym1 wins
        vote("p1", "B", "s2"),  # B is base -> base wins
        vote("p2", "same", "s2"),
    ]
    result = tally(votes, A_IS)
    assert result["fym1_wins"] == 2
    assert result["base_wins"] == 1
    assert result["same"] == 1
    assert result["n_votes"] == 4
    assert result["n_sessions"] == 2
    assert result["fym1_win_rate_excl_ties"] == pytest.approx(2 / 3)
    assert result["ci95"] == pytest.approx(list(wilson_interval(2, 3)))


def test_tally_counts_wrong_flags_per_model():
    result = tally([vote("p1", "same", wrong=["A", "B"]), vote("p2", "A", "s2", wrong=["A"])], A_IS)
    assert result["wrong_flags"] == {"base": 2, "fym1": 1}


def test_tally_keeps_first_vote_per_session_and_pair():
    result = tally([vote("p1", "A"), vote("p1", "B")], A_IS)
    assert (result["n_votes"], result["fym1_wins"], result["base_wins"]) == (1, 1, 0)


def test_tally_without_decisive_votes_has_no_rate():
    result = tally([vote("p1", "same")], A_IS)
    assert result["fym1_win_rate_excl_ties"] is None
    assert result["ci95"] is None


def test_tally_rejects_votes_for_unknown_pairs():
    with pytest.raises(ValueError, match="'p9'"):
        tally([vote("p9", "A")], A_IS)
