import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest
from votes import load_pairs, make_vote, pick_pair

from fym1.eval.arena import tally

SAMPLE_PAIRS = Path(__file__).resolve().parents[2] / "spaces" / "arena" / "pairs.sample.jsonl"
PAIR = {"pair_id": "p1", "prompt": "q", "a": "x", "b": "y", "a_is": "fym1"}


def test_no_choice_writes_no_vote():
    assert make_vote("p1", None, "s", set()) is None


def test_unknown_choice_writes_no_vote():
    assert make_vote("p1", "C", "s", set()) is None


def test_repeat_vote_writes_no_vote():
    assert make_vote("p1", "A", "s", {("s", "p1")}) is None


def test_same_pair_other_session_still_votes():
    assert make_vote("p1", "A", "s2", {("s", "p1")}) is not None


def test_vote_record():
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    vote = make_vote("p1", "B", "s", set(), wrong=["A"], now=now)
    assert vote == {
        "pair_id": "p1",
        "choice": "B",
        "wrong": ["A"],
        "session_id": "s",
        "ts": "2026-10-17T12:00:00+00:00",
    }


def test_choice_b():
    assert make_vote("p1", "B", "s", set())["choice"] == "B"


def test_wrong_flag_must_name_a_side():
    assert make_vote("p1", "A", "s", set(), wrong=["C"]) is None


def test_pick_pair_skips_pairs_this_session_voted_on():
    pairs = [PAIR, {**PAIR, "pair_id": "p2"}]
    picked = pick_pair(pairs, "s", {("s", "p1")}, random.Random(0))
    assert picked["pair_id"] == "p2"


def test_pick_pair_returns_none_when_all_done():
    assert pick_pair([PAIR], "s", {("s", "p1")}, random.Random(0)) is None


def write_pairs(tmp_path, *pairs):
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in pairs) + "\n", encoding="utf-8")
    return path


def test_load_pairs(tmp_path):
    assert load_pairs(write_pairs(tmp_path, PAIR)) == [PAIR]


def test_load_pairs_rejects_missing_key(tmp_path):
    broken = {k: v for k, v in PAIR.items() if k != "b"}
    with pytest.raises(ValueError, match=r"pairs.jsonl:1: missing keys \['b'\]"):
        load_pairs(write_pairs(tmp_path, broken))


def test_load_pairs_rejects_unknown_model(tmp_path):
    with pytest.raises(ValueError, match="a_is"):
        load_pairs(write_pairs(tmp_path, {**PAIR, "a_is": "gpt"}))


def test_load_pairs_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate pair_id 'p1'"):
        load_pairs(write_pairs(tmp_path, PAIR, PAIR))


def test_shipped_sample_pairs_load_and_are_marked_synthetic():
    pairs = load_pairs(SAMPLE_PAIRS)
    assert len(pairs) == 3
    assert all(p["pair_id"].startswith("SYNTHETIC-") for p in pairs)


def test_votes_from_the_space_tally_offline():
    # Contract between the Space (writer) and scripts/tally_votes.py (reader).
    vote = make_vote("p1", "A", "s", set(), wrong=["B"])
    result = tally([json.loads(json.dumps(vote))], {"p1": "fym1"})
    assert result["fym1_wins"] == 1
    assert result["wrong_flags"] == {"base": 1, "fym1": 0}
