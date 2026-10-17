from fym1.data.messenger import Message, Thread
from fym1.data.pipeline import build_records

MIN = 60_000


def chat(lines):
    return Thread(
        title="Nirina Rakoto",
        participants=["Nirina Rakoto", "Tojo Randria"],
        messages=[Message(sender=s, text=t, ts_ms=i * MIN) for i, (s, t) in enumerate(lines)],
    )


def says_malagasy(text):
    return "fr:" not in text


def all_text(records):
    return " ".join(turn["content"] for r in records for turn in r["messages"])


def test_build_records_anonymises_and_ids():
    thread = chat([("Nirina Rakoto", "Salama ry Tojo"), ("Tojo Randria", "Eny ry nirin, 034 12 345 67")])

    records = build_records([thread], owner="Tojo Randria", is_malagasy=says_malagasy)

    assert len(records) == 1
    text = all_text(records)
    assert "Tojo" not in text and "nirin" not in text.lower() and "<TEL>" in text
    assert len(records[0]["id"]) == 12
    assert records[0]["messages"][0]["role"] == "user"


def test_build_records_ids_are_stable():
    thread = chat([("Nirina Rakoto", "Salama"), ("Tojo Randria", "Eny")])

    first = build_records([thread], owner="Tojo Randria", is_malagasy=says_malagasy)
    again = build_records([thread], owner="Tojo Randria", is_malagasy=says_malagasy)

    assert [r["id"] for r in first] == [r["id"] for r in again]


def test_build_records_keeps_half_malagasy_drops_less():
    half = chat([("Nirina Rakoto", "fr: Bonjour"), ("Tojo Randria", "Salama")])
    mostly_french = chat([("Nirina Rakoto", "fr: Bonjour"), ("Tojo Randria", "fr: Ça va")])

    assert len(build_records([half], owner="Tojo Randria", is_malagasy=says_malagasy)) == 1
    assert build_records([mostly_french], owner="Tojo Randria", is_malagasy=says_malagasy) == []


def test_build_records_drops_duplicates():
    thread = chat([("Nirina Rakoto", "Salama"), ("Tojo Randria", "Eny")])

    records = build_records([thread, thread], owner="Tojo Randria", is_malagasy=says_malagasy)

    assert len(records) == 1
