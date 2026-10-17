import random
from datetime import datetime, timedelta, timezone

import pytest

from fym1.eval.arena import build_pairs, tally, wilson_interval

A_IS = {"p1": "fym1", "p2": "base", "p3": "fym1"}
START = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def vote(pair_id, choice, session_id="s", wrong=(), at=0):
    """A vote cast `at` seconds after START."""
    ts = (START + timedelta(seconds=at)).isoformat()
    return {"pair_id": pair_id, "choice": choice, "wrong": list(wrong), "session_id": session_id, "ts": ts}


def answers(*ids, prompt="q", response="r"):
    return [{"id": i, "prompt": f"{prompt} {i}", "response": f"{response} {i}"} for i in ids]


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
        vote("p1", "A", "s1", at=0),  # A is fym1 -> fym1 wins
        vote("p2", "B", "s1", at=30),  # B is fym1 -> fym1 wins
        vote("p1", "B", "s2", at=0),  # B is base -> base wins
        vote("p2", "same", "s2", at=30),
    ]
    result = tally(votes, A_IS)
    assert result["fym1_wins"] == 2
    assert result["base_wins"] == 1
    assert result["same"] == 1
    assert result["votes"] == 4
    assert result["raters"] == 2
    assert result["fym1_win_rate"] == pytest.approx(2 / 3)
    assert (result["ci_low"], result["ci_high"]) == pytest.approx(wilson_interval(2, 3))
    assert result["excluded_sessions"] == 0


def test_tally_counts_wrong_flags_per_model():
    result = tally([vote("p1", "same", "s1", wrong=["A", "B"]), vote("p2", "A", "s2", wrong=["A"])], A_IS)
    assert result["wrong_flags"] == {"base": 2, "fym1": 1}


def test_tally_keeps_first_vote_per_session_and_pair():
    result = tally([vote("p1", "A", at=0), vote("p1", "B", at=30)], A_IS)
    assert (result["votes"], result["fym1_wins"], result["base_wins"]) == (1, 1, 0)


def test_tally_without_decisive_votes_has_no_rate():
    result = tally([vote("p1", "same")], A_IS)
    assert (result["fym1_win_rate"], result["ci_low"], result["ci_high"]) == (None, None, None)


def test_tally_rejects_votes_for_unknown_pairs():
    with pytest.raises(ValueError, match="'p9'"):
        tally([vote("p9", "A")], A_IS)


def test_tally_excludes_sessions_voting_faster_than_a_person_reads():
    human = [vote("p1", "B", "human", at=0), vote("p2", "A", "human", at=40)]  # both base wins
    bot = [vote(p, "A", "bot", at=i) for i, p in enumerate(["p1", "p2", "p3"])]  # 1 s apart
    result = tally(human + bot, A_IS)
    assert (result["votes"], result["raters"], result["fym1_wins"], result["base_wins"]) == (2, 1, 0, 2)
    assert result["excluded_sessions"] == 1
    assert result["excluded_votes"] == 3
    assert result["flagged_sessions"] == [{"votes": 3, "median_seconds_between_votes": 1.0}]
    assert result["votes_per_session"] == [3, 2]


def test_tally_keeps_a_session_with_one_vote():
    assert tally([vote("p1", "A", "once")], A_IS)["excluded_sessions"] == 0


def test_build_pairs_shuffles_sides_and_keeps_a_is():
    base, fym1 = answers("p1", "p2", "p3", response="base"), answers("p1", "p2", "p3", response="fym1")
    pairs, skipped = build_pairs(base, fym1, random.Random(42))
    assert skipped == []
    assert [p["pair_id"] for p in pairs] == ["p1", "p2", "p3"]
    for pair in pairs:
        other = "base" if pair["a_is"] == "fym1" else "fym1"
        assert pair["a"] == f"{pair['a_is']} {pair['pair_id']}"
        assert pair["b"] == f"{other} {pair['pair_id']}"
        assert pair["prompt"] == f"q {pair['pair_id']}"


def test_build_pairs_is_reproducible_with_a_seed():
    rows = answers(*[f"p{i}" for i in range(20)])
    first, _ = build_pairs(rows, rows, random.Random(42))
    again, _ = build_pairs(rows, rows, random.Random(42))
    assert first == again
    assert {p["a_is"] for p in first} == {"base", "fym1"}


def test_build_pairs_skips_ids_missing_from_either_side():
    pairs, skipped = build_pairs(answers("p1", "p2"), answers("p2", "p3"), random.Random(0))
    assert [p["pair_id"] for p in pairs] == ["p2"]
    assert skipped == ["p1", "p3"]


def test_build_pairs_rejects_mismatched_prompts():
    with pytest.raises(ValueError, match="'p1'"):
        build_pairs(answers("p1"), answers("p1", prompt="other"), random.Random(0))
