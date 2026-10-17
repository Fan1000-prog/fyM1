from pathlib import Path

from fym1.data.messenger import Message, Thread, fix_meta_text, load_threads

FIXTURES = Path(__file__).parent / "fixtures"


def test_fix_meta_text_undoes_latin1_mojibake():
    assert fix_meta_text("Ã©") == "é"
    assert fix_meta_text("fotsinyð\u009f\u0098\u0082") == "fotsiny😂"


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

    assert thread == Thread(
        title="Nirina Rakoto",
        participants=["Nirina Rakoto", "Tojo Randria"],
        messages=thread.messages,
    )
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
