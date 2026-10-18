"""Synthetic-chat helpers. Every chat and model output here is made up for the test; none is real."""

import json

import pytest

from fym1.data import synth
from fym1.data.known_words import KNOWN_WORDS, word_list
from fym1.data.synth import (
    DROP_REASONS,
    Seed,
    build_prompt,
    examples_for,
    filter_conversations,
    parse_conversation,
    seed_grid,
    synth_stats,
    validate_stats,
)


def says_malagasy(text):
    """Stand-in for is_malagasy: French lines in these fixtures start with "fr:"."""
    return not text.startswith("fr:")


def conv(*lines):
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": line} for i, line in enumerate(lines)]


REAL = [
    {"id": "r1", "messages": conv("Salama ry namana, aiza ianao zao?", "Mbola any am-piasana aho fa efa ho vita kely")},
    {"id": "r2", "messages": conv("De ahoana ny lalao omaly hariva tany an-tanana?", "Tsara be fa resy izahay 2-1 😂")},
    {"id": "r3", "messages": conv("Aza adino ny rdv rahampitso maraina", "Ok, tsy hadino zany")},
]


# --- seed grid ---------------------------------------------------------------------------------------------


def test_seed_grid_is_deterministic_and_sized():
    assert seed_grid(50, seed=7) == seed_grid(50, seed=7)
    assert len(seed_grid(50, seed=7)) == 50
    assert seed_grid(50, seed=7) != seed_grid(50, seed=8)


def test_seed_grid_prefix_is_stable_when_n_grows():
    """A resumed run with a larger target must give the same seeds to the indices already generated."""
    assert seed_grid(500, seed=3)[:40] == seed_grid(40, seed=3)


def test_seed_grid_covers_every_topic_before_repeating():
    seeds = seed_grid(len(synth.TOPICS), seed=1)
    assert {s.topic for s in seeds} == set(synth.TOPICS)


def test_seed_fields_come_from_the_grid():
    for s in seed_grid(200, seed=2):
        assert s.topic in synth.TOPICS and s.register in synth.REGISTERS
        assert s.persona_user in synth.PERSONAS and s.persona_assistant in synth.PERSONAS
        assert synth.MIN_TURNS <= s.turns <= synth.MAX_TURNS and s.turns % 2 == 0
        assert set(s.names) <= set(synth.ALLOWED_NAMES)


def test_seed_round_trips_through_json():
    s = seed_grid(1, seed=0)[0]
    assert Seed.from_dict(json.loads(json.dumps(s.as_dict()))) == s


# --- few-shot picks and prompt -------------------------------------------------------------------------------


def test_examples_for_is_deterministic_per_index_and_varies_across_indices():
    picks = [tuple(r["id"] for r in examples_for(i, REAL, k=2, seed=5)) for i in range(20)]
    assert picks == [tuple(r["id"] for r in examples_for(i, REAL, k=2, seed=5)) for i in range(20)]
    assert len(set(picks)) > 1
    assert all(len(set(p)) == 2 for p in picks)


def test_examples_for_skips_overlong_real_chats():
    long_chat = {"id": "long", "messages": conv("a" * 3000, "b")}
    for i in range(30):
        assert "long" not in [r["id"] for r in examples_for(i, REAL + [long_chat], k=3, seed=0)]


def test_examples_for_needs_enough_real_chats():
    with pytest.raises(ValueError, match="few-shot"):
        examples_for(0, REAL[:1], k=2, seed=0)


def test_prompt_has_seed_examples_and_format_rules():
    s = Seed("market", "student", "shop owner", "teasing", 4, ("Soa", "Rindra"))
    prompt = build_prompt(s, [r["messages"] for r in REAL[:2]])
    assert "market" in prompt and "student" in prompt and "shop owner" in prompt and "teasing" in prompt
    assert "Soa, Rindra" in prompt and "4" in prompt
    assert "A: Salama ry namana, aiza ianao zao?" in prompt
    assert "B: Tsara be fa resy izahay 2-1 😂" in prompt
    assert prompt.rstrip().endswith(synth.NEW_CHAT_HEADER)


def test_prompt_puts_multiline_turns_on_one_line():
    s = seed_grid(1, seed=0)[0]
    prompt = build_prompt(s, [conv("Salama\nde aona?", "Tsara\nfa reraka")])
    assert "A: Salama / de aona?" in prompt and "B: Tsara / fa reraka" in prompt


# --- parser --------------------------------------------------------------------------------------------------


def test_parse_plain_output():
    out = "A: Salama ry zalahy\nB: De aona?\nA: Mbola miasa ve ialahy?\nB: Eny fa efa ho vita 😂"
    assert parse_conversation(out) == conv("Salama ry zalahy", "De aona?", "Mbola miasa ve ialahy?",
                                           "Eny fa efa ho vita 😂")


def test_parse_ignores_preamble_fences_bold_and_trailing_notes():
    out = ("Here is the chat:\n```\n**A:** Salama\n**B :** Eny salama\n```\n\n"
           "I hope this matches the style you wanted.")
    assert parse_conversation(out) == conv("Salama", "Eny salama")


def test_parse_joins_continuation_lines_and_merges_same_speaker():
    out = "A: Salama\nmbola tsy natory ve?\nA: hey\nB: Tsy mbola"
    assert parse_conversation(out) == conv("Salama\nmbola tsy natory ve?\nhey", "Tsy mbola")


def test_parse_trims_to_start_with_user_and_end_with_assistant():
    out = "B: aiza?\nA: eto\nB: ok\nA: dia?"
    assert parse_conversation(out) == conv("eto", "ok")


@pytest.mark.parametrize("out", ["", "no speakers at all", "A: only one side", "A: \nB: "])
def test_parse_returns_none_when_no_two_turn_chat(out):
    assert parse_conversation(out) is None


# --- filters -------------------------------------------------------------------------------------------------


def run(*convs, real=REAL):
    return filter_conversations(list(convs), REAL if real is REAL else real, is_malagasy=says_malagasy)


def test_clean_chat_is_kept_with_a_synth_id_in_chat_schema():
    kept, dropped = run(conv("Salama Soa, aiza ianao?", "Eto an-trano fotsiny, mijery tele"))
    assert sum(dropped.values()) == 0
    assert set(kept[0]) == {"id", "messages"} and kept[0]["id"].startswith("synth-")
    assert kept[0]["messages"] == conv("Salama Soa, aiza ianao?", "Eto an-trano fotsiny, mijery tele")


def test_unparseable_output_is_counted():
    kept, dropped = run(None)
    assert kept == [] and dropped["unparseable"] == 1


def test_assistant_turn_in_french_drops_the_chat_but_user_french_is_fine():
    _, dropped = run(conv("Salama", "fr: je suis à la maison"))
    assert dropped["not_malagasy"] == 1
    kept, _ = run(conv("fr: tu es où?", "Eto an-trano aho"))
    assert len(kept) == 1


@pytest.mark.parametrize("messages", [
    conv("a", "b" * (synth.MAX_TURN_CHARS + 1)),
    conv(*["Salama"] * (synth.MAX_TURNS + 2)),
    conv("Salama", "   "),
])
def test_length_bounds(messages):
    _, dropped = run(messages)
    assert dropped["length"] == 1


def test_phone_and_email_in_output_are_anonymised_then_dropped_as_placeholders():
    _, dropped = run(conv("Antsoy aho amin'ny 034 12 345 67", "Ok"), conv("Mandefasa any amin'ny x@y.mg", "Ok"))
    assert dropped["placeholder"] == 2


def test_placeholder_copied_from_examples_is_dropped():
    _, dropped = run(conv("Ity ny laharako <TEL>", "Ok"), conv("Salama <NAME>", "Ok"))
    assert dropped["placeholder"] == 2


def test_everyday_capitalised_word_at_line_start_is_kept():
    kept, dropped = run(conv("Merci be!", "Tsara be"), conv("Bonjour, aiza ianao?", "Eto aho"))
    assert len(kept) == 2 and dropped["unknown_name"] == 0


def test_unknown_name_like_token_is_dropped_not_replaced():
    kept, dropped = run(conv("Salama Zebulon, aiza ianao?", "Eto aho"), conv("Zebulon, aiza ianao?", "Eto aho"))
    assert kept == [] and dropped["unknown_name"] == 2


def test_fake_pool_names_are_kept():
    kept, _ = run(conv("Salama Rindra! Merci be", "Ok Soa"))
    assert len(kept) == 1


CORPUS_ONLY = "zokitra"  # synthetic: no real list has it; added to mg_corpus alone below


def test_corpus_only_word_in_a_name_slot_is_dropped(monkeypatch):
    corpus = KNOWN_WORDS["mg_corpus"]
    monkeypatch.setitem(KNOWN_WORDS, "mg_corpus", word_list(corpus.folded | {CORPUS_ONLY}))
    kept, dropped = run(conv("Salama", f"Efa nandeha i {CORPUS_ONLY}"))
    assert kept == [] and dropped["unknown_name"] == 1
    kept, _ = run(conv("Salama", f"Tsara ny {CORPUS_ONLY}"))  # outside a name slot the corpus clears it
    assert len(kept) == 1


def test_any_shared_eight_word_run_with_a_real_chat_drops_it():
    copied = "ary hoy izy hoe mbola any am-piasana aho fa efa ho vita kely"
    _, dropped = run(conv("Salama", copied))
    assert dropped["leak_8gram"] == 1
    seven = "Mbola any am-piasana aho fa efa"  # 7 words with "am-piasana" split in two = 7 \w-words
    kept, _ = run(conv("Salama", seven + " tokoa ve"))
    assert len(kept) == 1


def test_exact_duplicates_after_normalising_case_space_and_punctuation():
    kept, dropped = run(conv("Salama!", "Tsara be"), conv("salama", "tsara  be!!"))
    assert len(kept) == 1 and dropped["duplicate"] == 1


def test_near_duplicates_are_dropped_and_the_first_one_kept():
    a = conv("Salama, inona no vaovao any aminareo any an-tanana androany?", "Tsy misy fa mangatsiaka be ny andro")
    b = conv("Salama, inona no vaovao any aminareo any an-tanana androany?", "Tsy misy fa mangatsiaka be ny andro zao")
    far = conv("Efa nihinana ve ianao?", "Tsy mbola fa ho avy aho")
    kept, dropped = run(a, b, far)
    assert [k["messages"] for k in kept] == [a, far] and dropped["near_duplicate"] == 1


def test_first_failing_filter_wins_and_every_drop_has_a_known_reason():
    kept, dropped = run(None, conv("Salama", "fr: non"), conv("Salama", "Ok"))
    assert set(dropped) == set(DROP_REASONS)
    assert dropped["unparseable"] == 1 and dropped["not_malagasy"] == 1 and len(kept) == 1


def test_filter_is_deterministic():
    convs = [conv("Salama", "Ok"), conv("De aona", "Tsara"), None]
    assert run(*convs) == run(*convs)


def test_filter_needs_real_chats_to_check_leaks_against():
    with pytest.raises(ValueError):
        filter_conversations([conv("Salama", "Ok")], [], is_malagasy=says_malagasy)


# --- stats ---------------------------------------------------------------------------------------------------


def test_stats_are_counts_only_and_valid():
    kept, dropped = run(conv("Salama", "Ok"), None)
    stats = synth_stats(raw=2, kept=len(kept), dropped=dropped, by_generator={"google/gemma-4-12B-it": 2},
                        meta={"seed": 0, "few_shot_k": 3, "target_raw": 2500, "dtype": "float16",
                              "code_sha": "abc", "data_revision": "def", "gen_seconds": 10.5, "new_tokens": 900})
    validate_stats(stats)
    assert stats["raw"] == 2 and stats["kept"] == 1 and stats["dropped"]["unparseable"] == 1
    assert stats["kept"] + sum(stats["dropped"].values()) == stats["raw"]


@pytest.mark.parametrize("bad", [
    {"extra_text": "Salama ry namana"},
    {"dropped": {"secret_reason": 1}},
    {"kept": "12"},
    {"by_generator": {"google/gemma-4-12B-it": "many"}},
])
def test_validate_stats_rejects_unknown_keys_and_non_counts(bad):
    kept, dropped = run(conv("Salama", "Ok"))
    stats = synth_stats(raw=1, kept=1, dropped=dropped, by_generator={"google/gemma-4-12B-it": 1}, meta={})
    for key, value in bad.items():
        stats[key] = {**stats[key], **value} if isinstance(value, dict) and key in stats else value
    with pytest.raises(ValueError):
        validate_stats(stats)


def test_stats_reject_counts_that_do_not_add_up():
    stats = synth_stats(raw=5, kept=1, dropped={r: 0 for r in DROP_REASONS}, by_generator={}, meta={})
    with pytest.raises(ValueError, match="add up"):
        validate_stats(stats)


# --- generator licence ---------------------------------------------------------------------------------------

AFRIQUE = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"


@pytest.mark.parametrize("hub_id, licence", [(AFRIQUE, "cc-by-4.0"), ("google/gemma-4-12B-it", "apache-2.0")])
def test_permissive_generators_are_allowed(hub_id, licence):
    assert synth.generator_licence(hub_id) == licence
    assert synth.generator_licence(hub_id, card_licence=licence.upper()) == licence


@pytest.mark.parametrize("hub_id", ["google/gemma-3-27b-it", "google/gemma-3-12b-it", "someone/unknown-7b"])
def test_other_generators_are_refused(hub_id):
    """Gemma 3's terms would make fy-m1 a Gemma derivative; unknown models have no checked licence."""
    with pytest.raises(ValueError, match="not an allowed generator"):
        synth.generator_licence(hub_id)


@pytest.mark.parametrize("card", [None, "", "gemma", "other"])
def test_generator_refused_when_model_card_licence_differs(card):
    with pytest.raises(ValueError, match="model card"):
        synth.generator_licence("google/gemma-4-12B-it", card_licence=card)


def test_allowed_generators_are_permissive():
    assert set(synth.GENERATOR_LICENCES.values()) <= synth.PERMISSIVE_LICENCES
    assert not any("gemma-3" in hub_id for hub_id in synth.GENERATOR_LICENCES)


def test_generator_licences_match_model_selection():
    from pathlib import Path

    selection = json.loads((Path(__file__).parents[2] / "results" / "model_selection.json").read_text())
    measured = {c["model"]: c["license"].lower() for c in selection["candidates"]}
    for hub_id, licence in synth.GENERATOR_LICENCES.items():
        assert measured[hub_id] == licence


def test_stats_refuse_a_non_permissive_generator():
    stats = synth_stats(raw=1, kept=1, dropped={r: 0 for r in DROP_REASONS},
                        by_generator={"google/gemma-3-27b-it": 1}, meta={})
    with pytest.raises(ValueError, match="not an allowed generator"):
        validate_stats(stats)


def test_stats_record_the_generator_licence():
    stats = synth_stats(raw=1, kept=1, dropped={r: 0 for r in DROP_REASONS}, by_generator={AFRIQUE: 1},
                        meta={"generator": AFRIQUE, "generator_licence": "cc-by-4.0"})
    validate_stats(stats)


# --- fp16 overflow -------------------------------------------------------------------------------------------


def test_sampling_overflow_is_recognised():
    error = RuntimeError("probability tensor contains either `inf`, `nan` or element < 0")
    assert synth.is_sampling_overflow(error)


@pytest.mark.parametrize("error", [RuntimeError("CUDA out of memory"), ValueError("probability tensor contains"),
                                   RuntimeError("bad input")])
def test_other_errors_are_not_sampling_overflow(error):
    assert not synth.is_sampling_overflow(error)


# --- filter_raw_rows (04_synth's last step, also scripts/filter_synth.py) -------------------------------------


def test_filter_raw_rows_keeps_parsed_chats_and_builds_valid_stats():
    gen = "google/gemma-4-12B-it"
    rows = [{"output": out, "generator": gen, "dtype": "float32", "gen_seconds": 2.25, "new_tokens": 7}
            for out in ("A: Manao ahoana?\nB: Tsara be, misaotra", "tsy misy resaka eto")]  # fmt: skip
    kept, stats = synth.filter_raw_rows(rows, REAL, says_malagasy, meta={"seed": 0, "code_sha": "abc"})
    assert [r["messages"][1]["content"] for r in kept] == ["Tsara be, misaotra"]
    assert (stats["raw"], stats["kept"], stats["dropped"]["unparseable"]) == (2, 1, 1)
    assert (stats["seed"], stats["code_sha"], stats["generator"], stats["dtype"]) == (0, "abc", gen, "float32")
    assert (stats["generator_licence"], stats["gen_seconds"], stats["new_tokens"]) == ("apache-2.0", 4.5, 14)
    validate_stats(stats)
