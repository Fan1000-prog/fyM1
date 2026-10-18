import json
from pathlib import Path

import pytest

from fym1.data.messenger import PLACEHOLDER, Message, fix_meta_text, parse_nickname_event, load_threads

FIXTURES = Path(__file__).parent / "fixtures"


def test_fix_meta_text_undoes_latin1_mojibake():
    assert fix_meta_text("Ã©") == "é"
    assert fix_meta_text("fotsinyð\u009f\u0098\u0082") == "fotsiny😂"


def test_fix_meta_text_normalises_to_nfc():
    assert fix_meta_text("He\u0301le\u0300ne") == "Hélène"


def test_fix_meta_text_leaves_clean_text_alone():
    assert fix_meta_text("Salama é") == "Salama é"
    assert fix_meta_text("De aona?") == "De aona?"


def test_load_threads_skips_media_and_unsent():
    [thread] = load_threads(FIXTURES / "dyi_export")

    assert [m.text for m in thread.messages] == [
        "Salama ry Tojo! De aona?",
        "Tsara be, ary ianao?",
        "Mety be zany, lancer-na fotsiny😂",
        "Ok fa aza adino ny rdv à 18h é",
    ]


def test_load_threads_merges_pages_in_time_order():
    [thread] = load_threads(FIXTURES / "dyi_export")

    assert thread.title == "Hanta Rakoto"
    assert thread.participants == ["Hanta Rakoto", "Tojo Randria"]
    timestamps = [m.ts_ms for m in thread.messages]
    assert timestamps == sorted(timestamps)
    assert thread.messages[0] == Message(
        sender="Hanta Rakoto", text="Salama ry Tojo! De aona?", ts_ms=1_760_000_000_000
    )


def test_load_threads_ignores_message_requests():
    threads = load_threads(FIXTURES / "dyi_export")

    assert all("Olona Tsy Fantatra" not in t.participants for t in threads)


def test_load_threads_empty_dir_returns_nothing(tmp_path):
    assert load_threads(tmp_path) == []


def test_load_threads_harvests_nicknames():
    [thread] = load_threads(FIXTURES / "dyi_export")

    assert thread.nicknames == {"Bebe": "Tojo Randria"}


@pytest.mark.parametrize(
    "text",
    [
        "You sent a photo.",
        "Hanta sent 3 photos.",
        "Hanta a envoyé 2 photos.",
        "Hanta sent a link.",
        "Hanta started a video chat.",
        "You missed a call from Hanta.",
        "Hanta liked a message",
        "You reacted 😂 to Hanta's message",
        "Hanta set the nickname for Tojo to Bebe.",
    ],
)
def test_placeholder_matches_messenger_event_text(text):
    assert PLACEHOLDER.match(text)


def test_placeholder_keeps_typed_text():
    assert not PLACEHOLDER.match("Mety be zany, lancer-na fotsiny😂")


@pytest.mark.parametrize(
    "event, nickname, person",
    [
        ("Hanta set the nickname for Soa to Niry.", "Niry", "Soa"),
        ("Soa set her own nickname to Niry.", "Niry", "Soa"),
        ("You set your nickname to Niry.", "Niry", "Soa"),
        ("Hanta a défini le surnom de Soa sur Niry.", "Niry", "Soa"),
        ("Vous avez défini le pseudo de Soa sur Niry.", "Niry", "Soa"),
        ("Vous avez défini votre surnom sur Niry.", "Niry", "Soa"),
        ("Soa a défini son surnom sur Niry.", "Niry", "Soa"),
    ],
)
def test_nickname_events_en_and_fr(event, nickname, person):
    raw = {"sender_name": "Soa" if event.startswith(("Soa", "You", "Vous")) else "Hanta", "content": event}

    assert parse_nickname_event(raw, ["Hanta", "Soa"]) == (nickname, person)
    assert PLACEHOLDER.match(event)


def test_unharvested_nickname_events_are_counted(tmp_path):
    thread_dir = tmp_path / "messages" / "inbox" / "x_1"
    thread_dir.mkdir(parents=True)
    page = {
        "participants": [{"name": "Hanta"}, {"name": "Soa"}],
        "messages": [
            {"sender_name": "Hanta", "timestamp_ms": 1, "content": "Hanta set the nickname for Koto to Bebe."},
            {"sender_name": "Hanta", "timestamp_ms": 2, "content": "Hanta a modifié le surnom de Soa."},
            {"sender_name": "Soa", "timestamp_ms": 3, "content": "Soa set her own nickname to Niry."},
            {"sender_name": "Soa", "timestamp_ms": 4, "content": "Salama"},
        ],
    }
    (thread_dir / "message_1.json").write_text(json.dumps(page))

    [thread] = load_threads(tmp_path)

    assert thread.nicknames == {"Niry": "Soa"}
    assert thread.unharvested_nickname_events == 2


@pytest.mark.parametrize(
    "event, sender, person",
    [
        ("Hanta set your nickname to Fanfan.", "Hanta", "Soa"),  # "your" = the reader
        ("Hanta a défini votre surnom sur Fanfan.", "Hanta", "Soa"),
        ("You set your nickname to Fanfan.", "Soa", "Soa"),
        ("Vous avez défini votre surnom sur Fanfan.", "Soa", "Soa"),
        ("Hanta a défini son propre pseudo sur Fanfan.", "Hanta", "Hanta"),
        ("Hanta set the nickname for you to Fanfan.", "Hanta", "Soa"),
        ("Soa a défini le pseudo de Hanta sur Fanfan.", "Soa", "Hanta"),
    ],
)
def test_nickname_events_attribute_your_to_the_reader(event, sender, person):
    raw = {"sender_name": sender, "content": event}

    assert parse_nickname_event(raw, ["Hanta", "Soa"]) == ("Fanfan", person)
