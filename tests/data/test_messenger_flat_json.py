"""The flat one-file export: {"conversation": {...}, "messages": [{"id", "sender", "text", "timestamp"}]}."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fym1.data.messenger import load_threads
from fym1.data.pipeline import build_records
from fym1.data.segment import to_conversations


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _write(path: Path, messages: list[dict], **conversation) -> Path:
    conversation = {"title": "Synthetic family", "participants": ["Alpha", "Bravo", "Charlie"], **conversation}
    path.write_text(json.dumps({"conversation": conversation, "messages": messages}), encoding="utf-8")
    return path


def _msg(i: int, sender: str, text, ts: str) -> dict:
    return {"id": i, "sender": sender, "text": text, "timestamp": ts}


def test_flat_file_becomes_one_thread_sorted_by_time(tmp_path):
    _write(
        tmp_path / "group.json",
        [
            _msg(2, "Bravo", "tsara be", "2026-07-01T10:05:00"),
            _msg(1, "Alpha", "salama daholo", "2026-07-01T10:00:00"),
        ],
    )

    [thread] = load_threads(tmp_path)

    assert thread.title == "Synthetic family"
    assert thread.participants == ["Alpha", "Bravo", "Charlie"]
    assert [(m.sender, m.text, m.ts_ms) for m in thread.messages] == [
        ("Alpha", "salama daholo", _ms("2026-07-01T10:00:00")),
        ("Bravo", "tsara be", _ms("2026-07-01T10:05:00")),
    ]


def test_export_can_point_at_the_file_itself(tmp_path):
    path = _write(tmp_path / "group.json", [_msg(1, "Alpha", "manao ahoana", "2026-07-01T10:00:00")])

    [thread] = load_threads(path)

    assert [m.text for m in thread.messages] == ["manao ahoana"]


def test_missing_empty_and_non_text_messages_are_dropped(tmp_path):
    messages = [
        _msg(1, "Alpha", "eny", "2026-07-01T10:00:00"),
        _msg(2, "Bravo", "", "2026-07-01T10:01:00"),
        _msg(3, "Bravo", "   ", "2026-07-01T10:02:00"),
        _msg(4, "Bravo", None, "2026-07-01T10:03:00"),
        _msg(5, "Bravo", {"photo": "x.jpg"}, "2026-07-01T10:04:00"),
        {"id": 6, "sender": "Bravo", "timestamp": "2026-07-01T10:05:00"},
        _msg(7, "Bravo", "Bravo sent a photo.", "2026-07-01T10:06:00"),
        _msg(8, "Bravo", "  misaotra  ", "2026-07-01T10:07:00"),
    ]
    _write(tmp_path / "group.json", messages)

    [thread] = load_threads(tmp_path)

    assert [m.text for m in thread.messages] == ["eny", "misaotra"]


def test_text_is_nfc_and_not_treated_as_meta_mojibake(tmp_path):
    _write(tmp_path / "group.json", [_msg(1, "Alpha", "café é", "2026-07-01T10:00:00")])

    [thread] = load_threads(tmp_path)

    assert thread.messages[0].text == "café é"


def test_naive_timestamps_use_the_conversation_timezone(tmp_path):
    _write(
        tmp_path / "group.json",
        [_msg(1, "Alpha", "eny", "2026-07-01T13:00:00")],
        timezone="Indian/Antananarivo",
    )

    [thread] = load_threads(tmp_path)

    assert thread.messages[0].ts_ms == _ms("2026-07-01T10:00:00")  # UTC+3


def test_timestamps_with_an_offset_keep_it(tmp_path):
    _write(
        tmp_path / "group.json",
        [_msg(1, "Alpha", "eny", "2026-07-01T13:00:00+03:00"), _msg(2, "Bravo", "tsia", "2026-07-01T11:00:00Z")],
        timezone="Indian/Antananarivo",
    )

    [thread] = load_threads(tmp_path)

    assert [m.ts_ms for m in thread.messages] == [_ms("2026-07-01T10:00:00"), _ms("2026-07-01T11:00:00")]


def test_participants_fall_back_to_senders_in_first_seen_order(tmp_path):
    path = tmp_path / "group.json"
    messages = [
        _msg(1, "Bravo", "a", "2026-07-01T10:00:00"),
        _msg(2, "Alpha", "b", "2026-07-01T10:01:00"),
        _msg(3, "Bravo", "c", "2026-07-01T10:02:00"),
    ]
    path.write_text(json.dumps({"conversation": {"title": "t"}, "messages": messages}), encoding="utf-8")

    [thread] = load_threads(tmp_path)

    assert thread.participants == ["Bravo", "Alpha"]


def test_senders_missing_from_participants_are_added(tmp_path):
    _write(tmp_path / "group.json", [_msg(1, "Delta", "a", "2026-07-01T10:00:00")], participants=["Alpha"])

    [thread] = load_threads(tmp_path)

    assert thread.participants == ["Alpha", "Delta"]


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": []},  # no conversation
        {"conversation": {}, "items": []},  # no messages
        [1, 2, 3],  # not an object
        {"conversation": {}, "messages": [{"from": "Alpha", "body": "x", "when": 1}]},  # unknown message keys
    ],
)
def test_unknown_top_level_json_fails_loud(tmp_path, payload):
    (tmp_path / "mystery.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mystery.json"):
        load_threads(tmp_path)


@pytest.mark.parametrize("timestamp", [1751364000, "01/07/2026 10:00", None])
def test_unreadable_timestamp_fails_loud(tmp_path, timestamp):
    _write(tmp_path / "group.json", [_msg(1, "Alpha", "eny", timestamp)])

    with pytest.raises(ValueError, match="timestamp"):
        load_threads(tmp_path)


def test_unknown_timezone_fails_loud(tmp_path):
    _write(tmp_path / "group.json", [_msg(1, "Alpha", "eny", "2026-07-01T10:00:00")], timezone="Mars/Olympus")

    with pytest.raises(ValueError, match="timezone"):
        load_threads(tmp_path)


def test_dyi_export_folder_is_not_mistaken_for_flat_files(tmp_path):
    fixtures = Path(__file__).parent / "fixtures" / "dyi_export"

    threads = load_threads(fixtures)

    assert threads and all(t.messages for t in threads)


def test_long_group_chat_is_split_by_time_gaps(tmp_path):
    def burst(start_hour: int, first_id: int) -> list[dict]:
        return [
            _msg(first_id, "Alpha", "inona no vaovao", f"2026-07-01T{start_hour:02d}:00:00"),
            _msg(first_id + 1, "Owner", "tsy misy", f"2026-07-01T{start_hour:02d}:05:00"),
            _msg(first_id + 2, "Bravo", "aiza ianareo", f"2026-07-01T{start_hour:02d}:10:00"),
            _msg(first_id + 3, "Owner", "ao an-trano", f"2026-07-01T{start_hour:02d}:20:00"),
        ]

    messages = burst(8, 1) + burst(12, 10) + burst(18, 20)  # three bursts, 3h+ apart
    _write(tmp_path / "group.json", messages, participants=["Alpha", "Bravo", "Owner"])

    [thread] = load_threads(tmp_path)
    conversations = to_conversations(thread, owner="Owner")

    assert len(conversations) == 3
    assert all(c[0]["role"] == "user" and c[-1]["role"] == "assistant" for c in conversations)


NICKNAME_EVENTS_FLAT = [
    _msg(1, "Alpha", "Alpha set the nickname for Bravo to Zazakely.", "2026-07-01T08:00:00"),
    _msg(2, "Charlie", "Charlie a défini son surnom sur Kotokely.", "2026-07-01T08:01:00"),
    _msg(3, "Alpha", "Alpha a modifié le surnom de Bravo.", "2026-07-01T08:02:00"),  # no new nickname: unreadable
    _msg(4, "Alpha", "Alpha set the nickname for Delta to Bibilava.", "2026-07-01T08:03:00"),  # not a participant
    _msg(5, "Alpha", "Zazakely, aiza ianao?", "2026-07-01T08:04:00"),
    _msg(6, "Bravo", "ao an-trano aho, Kotokely koa", "2026-07-01T08:05:00"),
]


def test_flat_nickname_events_are_harvested_and_counted(tmp_path):
    _write(tmp_path / "group.json", NICKNAME_EVENTS_FLAT)

    [thread] = load_threads(tmp_path)

    assert thread.nicknames == {"Zazakely": "Bravo", "Kotokely": "Charlie"}
    assert thread.unharvested_nickname_events == 2


def test_flat_nickname_events_never_become_turns(tmp_path):
    _write(tmp_path / "group.json", NICKNAME_EVENTS_FLAT)

    [thread] = load_threads(tmp_path)

    assert [m.text for m in thread.messages] == ["Zazakely, aiza ianao?", "ao an-trano aho, Kotokely koa"]


def test_flat_nicknames_are_replaced_in_built_records(tmp_path):
    _write(tmp_path / "group.json", NICKNAME_EVENTS_FLAT)

    result = build_records(load_threads(tmp_path), owner="Bravo", is_malagasy=lambda _: True)

    text = json.dumps([r["messages"] for r in result.records], ensure_ascii=False)
    assert result.records
    assert "Zazakely" not in text and "Kotokely" not in text
    assert "nickname" not in text and "surnom" not in text


@pytest.mark.parametrize("sender", [None, 7, ["Alpha"]])
def test_sender_that_is_not_a_string_fails_loud(tmp_path, sender):
    messages = [_msg(1, "Alpha", "eny", "2026-07-01T10:00:00"), _msg(2, sender, "tsia", "2026-07-01T10:01:00")]
    _write(tmp_path / "group.json", messages)

    with pytest.raises(ValueError, match=r"group\.json: message 1 has no sender name"):
        load_threads(tmp_path)
