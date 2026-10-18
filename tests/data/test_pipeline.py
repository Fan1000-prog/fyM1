import pytest

from fym1.data.gate import FAKE_STATS_KEYS
from fym1.data.kept_checks import KEPT_COUNT_KEYS
from fym1.data.messenger import Message, Thread
from fym1.data.pipeline import build_records, merge_data_stats

MIN = 60_000


def chat(lines, nicknames=None):
    return Thread(
        title="Nirina Rakoto",
        participants=["Nirina Rakoto", "Tojo Randria"],
        messages=[Message(sender=s, text=t, ts_ms=i * MIN) for i, (s, t) in enumerate(lines)],
        nicknames=nicknames or {},
    )


def says_malagasy(text):
    return "fr:" not in text


def all_text(records):
    return " ".join(turn["content"] for r in records for turn in r["messages"])


def test_build_records_anonymises_and_ids():
    thread = chat([("Nirina Rakoto", "salama ry Tojo"), ("Tojo Randria", "Eny ry nirin, 034 12 345 67")])

    records = build_records([thread], owner="Tojo Randria", is_malagasy=says_malagasy).records

    assert len(records) == 1
    text = all_text(records)
    assert "Tojo" not in text and "nirin" not in text.lower() and "<TEL>" in text
    assert len(records[0]["id"]) == 12
    assert records[0]["messages"][0]["role"] == "user"


def test_build_records_ids_are_stable():
    thread = chat([("Nirina Rakoto", "salama"), ("Tojo Randria", "Eny")])

    first = build_records([thread], owner="Tojo Randria", is_malagasy=says_malagasy).records
    again = build_records([thread], owner="Tojo Randria", is_malagasy=says_malagasy).records

    assert [r["id"] for r in first] == [r["id"] for r in again]


def test_build_records_keeps_half_malagasy_drops_less():
    half = chat([("Nirina Rakoto", "fr: bonjour"), ("Tojo Randria", "salama")])
    mostly_french = chat([("Nirina Rakoto", "fr: bonjour"), ("Tojo Randria", "fr: Ça va")])

    assert len(build_records([half], owner="Tojo Randria", is_malagasy=says_malagasy).records) == 1
    assert build_records([mostly_french], owner="Tojo Randria", is_malagasy=says_malagasy).records == []


def test_build_records_drops_duplicates():
    thread = chat([("Nirina Rakoto", "salama"), ("Tojo Randria", "Eny")])

    records = build_records([thread, thread], owner="Tojo Randria", is_malagasy=says_malagasy).records

    assert len(records) == 1


def test_build_records_replaces_nicknames():
    thread = chat([("Nirina Rakoto", "Salama Bebe"), ("Tojo Randria", "Eny")], nicknames={"Bebe": "Tojo Randria"})

    records = build_records([thread], owner="Tojo Randria", is_malagasy=says_malagasy).records

    assert "Bebe" not in all_text(records)


def test_build_records_counts_common_words_replaced():
    vola = Thread(
        title="Vola Rabe",
        participants=["Vola Rabe", "Tojo Randria"],
        messages=[Message("Vola Rabe", "Misy vola ve?", 0), Message("Tojo Randria", "Tsy misy", MIN)],
    )

    result = build_records([vola], owner="Tojo Randria", is_malagasy=says_malagasy)

    assert result.common_word_replacements == 1


def test_merge_data_stats_fills_build_counts_and_keeps_later_ones():
    later = {"approved": 90, "identity": 16}

    stats = merge_data_stats(later, threads=3, conversations=120)

    assert stats == {
        "threads": 3,
        "conversations": 120,
        **dict.fromkeys(KEPT_COUNT_KEYS),
        **dict.fromkeys(FAKE_STATS_KEYS),
        "approved": 90,
        "identity": 16,
        "formal": None,
        "synthetic": None,
        "train_examples": None,
    }


def test_merge_data_stats_rejects_unknown_keys():
    with pytest.raises(ValueError, match="texts"):
        merge_data_stats({}, texts=1)


@pytest.mark.parametrize("deleted", ["Facebook User", "Utilisateur Facebook", "facebook user"])
def test_deleted_account_display_names_are_not_treated_as_names(deleted):
    thread = Thread(
        title="t",
        participants=[deleted, "Tojo Randria"],
        messages=[Message(deleted, "iza ilay user?", 0), Message("Tojo Randria", "facebook ve", MIN)],
    )

    records = build_records([thread], owner="Tojo Randria", is_malagasy=says_malagasy).records

    assert all_text(records) == "iza ilay user? facebook ve"


def test_build_records_applies_blocklist_everywhere():
    thread = chat([("Nirina Rakoto", "Hoy i Koto sy Kotoaaa"), ("Tojo Randria", "Eny, Koto ve?")])

    records = build_records([thread], owner="Tojo Randria", is_malagasy=says_malagasy, blocklist=["Koto"]).records

    assert "koto" not in all_text(records).lower()


def test_build_records_reports_leftover_names():
    thread = chat([("Nirina Rakoto", "Hoy i Bema"), ("Tojo Randria", "Eny, Bema ve?")])

    result = build_records([thread], owner="Tojo Randria", is_malagasy=says_malagasy)

    assert result.leftover_names["Bema"] == 2


def test_build_records_every_sender_drops_exact_duplicates():
    group = Thread(
        title="t",
        participants=["Nirina Rakoto", "Tojo Randria", "Hery Rabe"],
        messages=[
            Message(sender=s, text=t, ts_ms=m * MIN)
            for s, m, t in [
                ("Nirina Rakoto", 0, "salama"),
                ("Tojo Randria", 1, "Eny"),
                ("Nirina Rakoto", 600, "salama"),  # same exchange again, hours later
                ("Tojo Randria", 601, "Eny"),
            ]
        ],
    )

    records = build_records([group], owner=None, is_malagasy=says_malagasy).records

    assert len(records) == 1  # Tojo as assistant, twice -> once; Nirina and Hery yield nothing


def test_build_records_gate_keeps_prices_drops_numbers_and_replaces_leftover_names():
    threads = [
        chat([("Nirina Rakoto", "inona ny vidiny?"), ("Tojo Randria", "1 500 000 Ar")]),
        chat([("Nirina Rakoto", "inona ny ref?"), ("Tojo Randria", "ref 123-4567")]),
        chat([("Nirina Rakoto", "hoy i Bema"), ("Tojo Randria", "eny")]),
    ]

    result = build_records(threads, owner="Tojo Randria", is_malagasy=says_malagasy)

    assert "inona ny vidiny? 1 500 000 Ar" in all_text(result.records)
    assert "Bema" not in all_text(result.records) and "hoy i " in all_text(result.records)
    assert {k: v for k, v in result.gate.items() if not k.startswith("cleared_")} == {
        "conversations_before_gate": 3,
        "dropped": 1,
        "dropped_long_number": 1,
        "dropped_contact": 0,
        "dropped_too_many_unknown": 0,
        "dropped_too_many_fakes": 0,
        "dropped_hidden_char": 0,
        "kept": 2,
        "tokens_replaced": 1,
        "words_kept": 8,
        "fakes_per_kept_median": 0,
        "fakes_per_kept_max": 1,
    }
    assert result.leftover_names["Bema"] == 1  # the optional sheet still sees pre-gate words


def test_build_records_every_sender_hides_every_name_from_every_side():
    people = ["Nirina Rakoto", "Tojo Randria", "Hery Rabe"]
    group = Thread(
        title="t",
        participants=people,
        messages=[
            Message(sender=s, text=t, ts_ms=i * MIN)
            for i, (s, t) in enumerate([
                ("Nirina Rakoto", "salama ry Tojo sy Hery"),
                ("Tojo Randria", "eny ry Nirina, aiza i hery?"),
                ("Hery Rabe", "ato aho ry NIRINA sy tojo, Rakoto ve?"),
                ("Nirina Rakoto", "Randria sy Rabe tonga"),
            ])
        ],
    )

    records = build_records([group], owner=None, is_malagasy=says_malagasy).records

    assistants = {turn["content"] for r in records for turn in r["messages"] if turn["role"] == "assistant"}
    assert len(assistants) == 3  # every sender answered at least once, so every side was built
    text = all_text(records).lower()
    for part in ("nirina", "rakoto", "tojo", "randria", "hery", "rabe"):
        assert part not in text


def test_build_records_replaces_a_real_name_that_is_also_a_pool_name():
    thread = chat([("Nirina Rakoto", "Rabe no nandeha omaly"), ("Tojo Randria", "eny tsara ihany")])

    result = build_records([thread], owner="Tojo Randria", is_malagasy=says_malagasy)

    assert "Rabe" not in all_text(result.records)
    assert "no nandeha omaly" in all_text(result.records)
