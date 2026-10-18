"""The automatic synth sample gate. Every chat, sender and message here is made up for the test."""

import pytest

from fym1.data.synth import record_id
from fym1.data.synth_gate import (
    DROP_REASONS,
    RawReference,
    broken,
    chat_reasons,
    raw_reference,
    run_gate,
)

SENDERS = ["Zebulon Quixbert", "Facebook User"]
RAW_LINE = "Aza adino ny mitondra ilay boky manga rahampitso maraina fa ilaina any am-pianarana"
RAW_TEXTS = [RAW_LINE, "ok"]


def says_malagasy(text):
    """Stand-in for is_malagasy: French lines in these fixtures start with "fr:"."""
    return not text.startswith("fr:")


def chat(*turns, roles=None):
    roles = roles or ["user", "assistant"] * (len(turns) // 2) + ["user"] * (len(turns) % 2)
    messages = [{"role": role, "content": text} for role, text in zip(roles, turns, strict=True)]
    return {"id": record_id(messages), "messages": messages}


def clean_chat(tag="tsara"):
    return chat("Manao ahoana ianao androany?", f"Salama aho, {tag} fa vizaka kely")


@pytest.fixture
def raw():
    return raw_reference(SENDERS, RAW_TEXTS)


def reasons(record, raw):
    return chat_reasons(record, raw, says_malagasy)


def test_raw_reference_holds_name_tokens_and_leak_runs(raw):
    assert isinstance(raw, RawReference)
    assert raw.names == frozenset({"zebulon", "quixbert"})  # the placeholder sender is skipped
    assert raw.leaks.shares_with(f"hoy izy: {RAW_LINE}")


def test_clean_chat_has_no_reasons(raw):
    assert reasons(clean_chat(), raw) == []


@pytest.mark.parametrize("record", [
    {"id": "synth-x", "messages": []},
    {"messages": [{"role": "user", "content": "Salama"}, {"role": "assistant", "content": "Salama"}]},
    chat("Salama", "   "),
    chat("Salama", "Salama tompoko", roles=["assistant", "user"]),
    chat("Salama", "Salama tompoko", roles=["user", "user"]),
    chat("Salama", "Salama tompoko", "Veloma"),  # ends on a user turn
    chat("Salama", "Salama tompoko", roles=["system", "assistant"]),
    chat("Salama", "Salama <TEL> tompoko"),
    chat("Inona no vaovao?", "Tsy misy", "Inona no vaovao?", "Tsy misy"),  # a turn repeated verbatim
])
def test_broken_shapes(record):
    assert broken(record)


def test_id_must_match_the_content():
    record = clean_chat()
    assert not broken(record)
    assert broken({**record, "id": "synth-000000000000"})


def test_turn_content_must_be_text():
    record = clean_chat()
    record["messages"][1]["content"] = 42
    assert broken(record)


@pytest.mark.parametrize("text", ["Hoy i Zebulon fa ho avy izy", "salama zebulon", "Z e b u l o n no nandeha",
                                  "Hitako i Quixb3rt omaly"])
def test_real_sender_name_drops(raw, text):
    assert "real_name" in reasons(chat("Iza no tonga?", text), raw)


def test_unknown_capitalised_word_drops(raw):
    assert "unknown_name" in reasons(chat("Iza no tonga?", "Tonga i Brumhilda omaly"), raw)


def test_fake_pool_name_passes(raw):
    assert reasons(chat("Iza no tonga?", "Tonga i Hasina omaly hariva"), raw) == []


@pytest.mark.parametrize("text, reason", [
    ("Antsoy amin'ny 034 12 345 67 aho", "long_number"),
    ("Alefaso any amin'ny vola.test@gmail.com", "contact"),
    ("Jereo ny www.example.com/sary", "contact"),
    ("Sala\u0903ma tompoko", "hidden_char"),
])
def test_numbers_contacts_and_hidden_characters_drop(raw, text, reason):
    assert reason in reasons(chat("Inona no vaovao?", text), raw)


def test_hidden_character_normalise_would_drop_still_drops(raw):
    """The gate keeps the chat as written, so a zero-width space normalise() removes still counts."""
    assert "hidden_char" in reasons(chat("Inona no vaovao?", "Salama\u200b tompoko"), raw)


def test_emoji_sequence_is_not_hidden(raw):
    assert reasons(chat("Inona no vaovao?", "Salama tompoko \U0001f468\u200d\U0001f469\u200d\U0001f467 \u2764\ufe0f"), raw) == []


def test_a_price_is_kept(raw):
    assert reasons(chat("Ohatrinona?", "1 500 000 Ar ny vidiny"), raw) == []


def test_eight_word_run_of_the_raw_export_drops(raw):
    assert "leak_8gram" in reasons(chat("Inona no tadidiko?", RAW_LINE), raw)


def test_seven_shared_words_are_not_a_leak(raw):
    seven = " ".join(RAW_LINE.split()[:7])
    assert "leak_8gram" not in reasons(chat("Inona no tadidiko?", f"{seven} sa tsia"), raw)


def test_assistant_turn_with_no_malagasy_drifts(raw):
    assert "drift" in reasons(chat("Salama", "fr: je suis au travail"), raw)


def test_user_turn_in_french_is_not_drift(raw):
    assert reasons(chat("fr: tu fais quoi", "Miasa aho izao"), raw) == []


@pytest.mark.parametrize("text", ["Izaho dia ChatGPT", "Mampiasa Gemini aho matetika"])
def test_assistant_claiming_or_naming_another_assistant_drops(raw, text):
    assert "identity" in reasons(chat("Iza ianao?", text), raw)


def test_denying_another_assistant_passes(raw):
    assert "identity" not in reasons(chat("ChatGPT ve ianao?", "Tsia, tsy ChatGPT aho"), raw)


def test_every_reason_is_listed(raw):
    record = chat("Iza?", "fr: Brumhilda zebulon 0341234567")
    assert set(reasons(record, raw)) <= set(DROP_REASONS)
    assert {"real_name", "unknown_name", "long_number", "drift"} <= set(reasons(record, raw))


def test_broken_chat_stops_further_checks(raw):
    assert reasons({"id": "synth-x", "messages": "nope"}, raw) == ["broken"]


def test_run_gate_counts_and_keeps_only_clean_ids(raw):
    good, other = clean_chat("tsara"), clean_chat("faly")
    leak = chat("Inona?", RAW_LINE)
    named = chat("Iza?", "Tonga i Zebulon")
    result = run_gate([good, leak, other, named], raw, says_malagasy, sample_size=2, seed=7)
    report = result.report
    assert result.kept_ids == [good["id"], other["id"]]
    assert report["checked"] == 4 and report["kept"] == 2 and report["dropped"] == 2
    assert report["dropped_by_reason"]["leak_8gram"] == 1
    assert report["dropped_by_reason"]["real_name"] == 1
    assert set(report["dropped_by_reason"]) == set(DROP_REASONS)
    assert report["dropped_ids"] == {leak["id"]: ["leak_8gram"], named["id"]: ["real_name", "unknown_name"]}
    assert report["status"] == "fail"  # any 8-word match fails the gate, not only the chat


def test_run_gate_passes_with_name_drops_only(raw):
    named = chat("Iza?", "Tonga i Zebulon")
    report = run_gate([clean_chat(), named], raw, says_malagasy, sample_size=1, seed=7).report
    assert report["status"] == "pass"
    assert report["dropped"] == 1


def test_duplicate_ids_are_broken(raw):
    good = clean_chat()
    result = run_gate([good, good], raw, says_malagasy, sample_size=1, seed=7)
    assert result.kept_ids == [good["id"]]
    assert result.report["dropped_by_reason"]["broken"] == 1


def test_sample_is_fixed_by_seed_and_counts_only(raw):
    records = [clean_chat(tag) for tag in ("tsara", "faly", "vizaka", "reraka", "sambatra")]
    one = run_gate(records, raw, says_malagasy, sample_size=3, seed=11).report["sample"]
    two = run_gate(records, raw, says_malagasy, sample_size=3, seed=11).report["sample"]
    assert one == two
    assert one["size"] == 3 and one["seed"] == 11
    assert all(isinstance(v, int) for k, v in one.items() if k != "ids")


def test_sample_counts_unlisted_lowercase_words_without_dropping(raw):
    odd = chat("Salama", "Salama tompoko, xqzvb ihany")
    report = run_gate([odd], raw, says_malagasy, sample_size=1, seed=1).report
    assert report["kept"] == 1
    assert report["sample"]["chats_with_unlisted_word"] == 1


def test_report_never_holds_chat_text(raw):
    leak = chat("Inona?", RAW_LINE)
    named = chat("Iza?", "Tonga i Zebulon")
    report = run_gate([leak, named, clean_chat()], raw, says_malagasy, sample_size=3, seed=1).report
    text = repr(report)
    for word in ("Zebulon", "zebulon", "mitondra", "Tonga", "Salama", "vizaka"):
        assert word not in text


def test_sample_larger_than_the_set_takes_every_chat(raw):
    report = run_gate([clean_chat()], raw, says_malagasy, sample_size=100, seed=1).report
    assert report["sample"]["size"] == 1
