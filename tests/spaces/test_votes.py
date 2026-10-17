import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest
from votes import VoteBook, is_valid_session_id, load_pairs, make_vote, new_session_id, pick_pair, sign

from fym1.eval.arena import build_pairs, tally

SAMPLE_PAIRS = Path(__file__).resolve().parents[2] / "spaces" / "arena" / "pairs.sample.jsonl"
KEY = b"test-key"
PAIR = {"pair_id": "p1", "prompt": "q", "a": "x", "b": "y"}


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


def test_load_pairs_refuses_a_file_that_names_the_models(tmp_path):
    # pairs.jsonl is public in the Space's Files tab; a_is there would unblind raters
    with pytest.raises(ValueError, match="a_is"):
        load_pairs(write_pairs(tmp_path, {**PAIR, "a_is": "fym1"}))


def test_load_pairs_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate pair_id 'p1'"):
        load_pairs(write_pairs(tmp_path, PAIR, PAIR))


def test_shipped_sample_pairs_load_and_are_marked_synthetic():
    pairs = load_pairs(SAMPLE_PAIRS)
    assert len(pairs) == 3
    assert all(p["pair_id"].startswith("SYNTHETIC-") for p in pairs)


def test_built_pairs_load_in_the_space(tmp_path):
    # Contract between scripts/build_pairs.py (writer) and the Space (reader).
    rows = [{"id": "p1", "prompt": "q", "response": "r"}]
    pairs, _, _ = build_pairs(rows, rows, random.Random(42))
    assert load_pairs(write_pairs(tmp_path, *pairs)) == pairs


def test_votes_from_the_space_tally_offline():
    # Contract between the Space (writer) and scripts/tally_votes.py (reader).
    vote = make_vote("p1", "A", "s", set(), wrong=["B"])
    result = tally([json.loads(json.dumps(vote))], {"p1": "fym1"})
    assert result["fym1_wins"] == 1
    assert result["wrong_flags"] == {"base": 1, "fym1": 0}


def test_minted_session_id_is_valid():
    assert is_valid_session_id(new_session_id(KEY), KEY)


@pytest.mark.parametrize(
    "session_id",
    [None, 7, {"x": 1}, "", "abc", "abc.def", "." + "0" * 64, "a" * 200],
)
def test_junk_session_ids_are_rejected(session_id):
    assert not is_valid_session_id(session_id, KEY)


def test_session_id_signed_with_another_key_is_rejected():
    assert not is_valid_session_id(new_session_id(b"other-key"), KEY)


def test_overlong_session_id_is_rejected_even_if_signed():
    nonce = "n" * 100
    assert not is_valid_session_id(f"{nonce}.{sign(nonce, KEY)}", KEY)


def test_vote_book_saves_once_per_pair():
    book = VoteBook(KEY, max_votes_per_session=10)
    session_id = new_session_id(KEY)
    assert book.record("p1", "A", session_id) is not None
    assert book.record("p1", "B", session_id) is None
    assert book.votes_cast[session_id] == 1


def test_vote_book_rejects_forged_session():
    assert VoteBook(KEY, max_votes_per_session=10).record("p1", "A", "made-up.id") is None


def test_vote_book_caps_votes_per_session():
    book = VoteBook(KEY, max_votes_per_session=2)
    session_id = new_session_id(KEY)
    saved = [book.record(f"p{i}", "A", session_id) for i in range(3)]
    assert [vote is not None for vote in saved] == [True, True, False]


def test_vote_book_ignores_empty_choice():
    book = VoteBook(KEY, max_votes_per_session=10)
    session_id = new_session_id(KEY)
    assert book.record("p1", None, session_id) is None
    assert book.votes_cast[session_id] == 0
