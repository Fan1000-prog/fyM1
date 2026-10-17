from pathlib import Path

import pytest

from fym1.data.messenger import PLACEHOLDER, Message, fix_meta_text, load_threads

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

    assert thread.title == "Nirina Rakoto"
    assert thread.participants == ["Nirina Rakoto", "Tojo Randria"]
    timestamps = [m.ts_ms for m in thread.messages]
    assert timestamps == sorted(timestamps)
    assert thread.messages[0] == Message(
        sender="Nirina Rakoto", text="Salama ry Tojo! De aona?", ts_ms=1_760_000_000_000
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
        "Nirina sent 3 photos.",
        "Nirina a envoyé 2 photos.",
        "Nirina sent a link.",
        "Nirina started a video chat.",
        "You missed a call from Nirina.",
        "Nirina liked a message",
        "You reacted 😂 to Nirina's message",
        "Nirina set the nickname for Tojo to Bebe.",
    ],
)
def test_placeholder_matches_messenger_event_text(text):
    assert PLACEHOLDER.match(text)


def test_placeholder_keeps_typed_text():
    assert not PLACEHOLDER.match("Mety be zany, lancer-na fotsiny😂")
