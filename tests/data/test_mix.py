"""The r2 mix filters. Every chat, name and line here is made up for the test."""

import pytest

from fym1.data import mix, name_scan

RAW_NAMES, _, _ = name_scan.name_tokens(["Zebulon Quixbert"])
LONG_EVAL = "Lazao amiko ny tantaran'ny vohitra kely tany amin'ny morontsiraka atsinanana"  # 13 plain words
EVAL = mix.EvalPrompts(["Iza no filoha voalohany?", LONG_EVAL])


def says_malagasy(text):
    """Stand-in for is_malagasy: French/English lines in these fixtures start with "fr:"."""
    return not text.startswith("fr:")


def chat(*turns):
    messages = [{"role": ("user", "assistant")[i % 2], "content": t} for i, t in enumerate(turns)]
    return {"id": mix.record_id(messages), "messages": messages}


@pytest.mark.parametrize("text", [
    "Ny anarako dia Koto", "My name is Bob", "Je m'appelle Paul", "Izaho dia Koto", "Zaho i Bema", "Moi c'est Paul",
])  # fmt: skip
def test_name_claims_are_caught(text):
    assert mix.name_claim(text)


@pytest.mark.parametrize("text", ["Izaho dia fy-m1, mpanampy AI", "Inona no anaranao?", "Tsara be izany"])
def test_no_name_claim(text):
    assert not mix.name_claim(text)


@pytest.mark.parametrize("text", [
    "Ho tonga aho rahariva", "Zah efa any an-trano", "Mbola miandry aho", "Ok j'arrive", "I'm coming home",
    "Tsara, ho avy izahay rahampitso",
])  # fmt: skip
def test_personal_logistics_are_caught(text):
    assert mix.personal_logistics(text)


@pytest.mark.parametrize("text", [
    "Eto aho hanampy anao", "Tonga ny fahavaratra", "Andriamanitra no tonga teto", "Radama I nanjaka tany",
    "Afaka manoratra tononkalo aho",
])  # fmt: skip
def test_no_personal_logistics(text):
    assert not mix.personal_logistics(text)


@pytest.mark.parametrize("text", [
    "Hiantso anao amin'ny 2h aho", "Je t'appelle à 8h", "I'll call you at 7pm", "Zaho 14h30 vao afaka",
    "I can do it at 9", "Je finis vers 18H",
])  # fmt: skip
def test_first_person_clock_times_are_logistics(text):
    assert mix.personal_logistics(text)


@pytest.mark.parametrize("text", [
    "Manomboka amin'ny 2 ora ny fivoriana", "Le film commence à 8h", "The shop opens at 7pm", "Aho 8hours",
])  # fmt: skip
def test_clock_time_without_the_speaker_is_not_logistics(text):
    assert not mix.personal_logistics(text)


def test_logistics_needs_speaker_and_place_in_one_clause():
    assert not mix.personal_logistics("Tsy azoko aho. Lasa ny fiara")


def test_fake_names_are_capitalised_pool_names():
    assert mix.fake_name("Ahoana Haja?")
    assert mix.fake_name("Vola be izany")  # a pool name that is also a word: fail closed
    assert not mix.fake_name("mila vola aho")
    assert not mix.fake_name("Salama tompoko")


def test_persona_reasons_keep_order():
    assert mix.persona_reasons("Haja ny anarako, ho tonga aho rahariva") == [
        "name_claim", "personal_logistics", "fake_name"]  # fmt: skip
    assert mix.persona_reasons("Tsara be izany") == []


@pytest.mark.parametrize(("user", "reply", "expected"), [
    ("Nahoana no mangatsiaka be?", "Ririnina izao", True),
    ("Mandeha amin'ny tsena aho", "Ok, mividiana mofo", True),
    ("Tena tsara ilay sakafo omaly", "Sakafo inona izany?", True),
    ("Tena tsara ilay sakafo omaly", "Lasa ny fiara", False),
    ("Tena tsara ilay sakafo omaly", "😂", False),
])  # fmt: skip
def test_responds(user, reply, expected):
    assert mix.responds(user, reply) is expected


def test_real_pairs_keep_only_clean_answers():
    records = [
        chat("Inona no vaovao?", "Tsy misy fa salama tsara", "Tena tsara ilay sakafo", "Lasa ny fiara"),
        chat("Aiza ianao?", "Ho tonga aho rahariva"),
        chat("Ahoana?", "Haja no niantso"),
        chat("Ahoana?", "fr: all good thanks"),
        chat("Inona no vaovao?", "Tsy misy fa salama tsara"),  # same pair again
        chat("Iza no tonga?", "Zebulon no tonga"),
        chat("Iza ianao?", "ChatGPT aho"),
    ]
    kept, drops = mix.real_pairs(records, says_malagasy, RAW_NAMES, EVAL)
    assert [r["messages"][1]["content"] for r in kept] == ["Tsy misy fa salama tsara"]
    assert all(r["id"].startswith("mix-") and len(r["messages"]) == 2 for r in kept)
    assert dict(drops) == {"eval_prompt": 0, "not_a_reply": 1, "drift": 1, "identity": 1, "name_claim": 0,
                           "personal_logistics": 1, "fake_name": 1, "real_name": 1, "duplicate": 1}  # fmt: skip


def test_cut_persona_keeps_the_prefix_before_the_first_persona_reply():
    record = chat("Salama", "Salama tsara", "Aiza ianao?", "Ho tonga aho rahariva", "Ok", "Veloma")
    cut, reason = mix.cut_persona(record)
    assert reason == "personal_logistics"
    assert [t["content"] for t in cut["messages"]] == ["Salama", "Salama tsara"]
    assert cut["id"] == mix.record_id(cut["messages"])


def test_cut_persona_drops_a_chat_with_nothing_left_and_keeps_a_clean_one():
    assert mix.cut_persona(chat("Iza ianao?", "Haja aho")) == (None, "fake_name")
    clean = chat("Salama", "Salama tsara")
    assert mix.cut_persona(clean) == (clean, None)


def test_synth_rows_counts():
    rows, counts = mix.synth_rows([chat("Salama", "Salama tsara"), chat("Iza ianao?", "Haja aho"),
                                   chat("Salama", "Eny e", "Aiza?", "Mbola miandry aho"),
                                   chat("Salama", "Eny e", "iza no filoha voalohany", "Tsy haiko")], EVAL)  # fmt: skip
    assert len(rows) == 2
    assert counts == {"eval_prompt": 1, "whole": 1, "cut": 1, "dropped": 1, "reason_name_claim": 0,
                      "reason_personal_logistics": 1, "reason_fake_name": 1}  # fmt: skip


def aya(inputs, targets, code="plt", kind="original-annotations"):
    return {"inputs": inputs, "targets": targets, "language_code": code, "annotation_type": kind}


def test_aya_malagasy_takes_plateau_malagasy_original_annotations():
    rows = [aya("a", "b"), aya("a", "b", kind="re-annotations"), aya("a", "b", code="eng")]
    kept, counts = mix.aya_malagasy(rows)
    assert kept == [rows[0]]
    assert counts == {"malagasy_rows": 2, "original_annotations": 1, "re_annotations_left_out": 1}


def test_aya_rows_gates():
    rows = [
        aya("Inona no renivohitr'i Madagasikara?", "Antananarivo no renivohitra"),
        aya("IZA  no filoha voalohany ?!", "Philibert Tsiranana"),
        aya("Inona no renivohitr'i Madagasikara?", "Antananarivo no renivohitra"),
        aya("Lazao", "x" * (mix.AYA_MAX_CHARS + 1)),
        aya("fr: quelle est la capitale?", "Antananarivo"),
        aya("Inona no renivohitra?", "fr: the capital is Tana"),
        aya("Iza ianao?", "ChatGPT aho"),
        aya("Iza ianao?", "Ny anarako dia Koto"),
        aya("Aiza ianao?", "Efa any an-trano aho"),
        aya("Iza no mpanoratra?", "Haja no nanoratra azy"),
        aya("Iza no mpanoratra?", "Zebulon no nanoratra azy"),
    ]
    kept, drops = mix.aya_rows(rows, says_malagasy, RAW_NAMES, EVAL)
    assert len(kept) == 1
    assert dict(drops) == {reason: 1 for reason in mix.AYA_DROP_REASONS}


@pytest.mark.parametrize("text", [
    "Iza no filoha voalohany?",
    "iza   no FILOHA voalohany",  # case, spaces, punctuation ignored
    "Iza no filoha voalohany...!!",
    "Azafady, lazao amiko ny tantaran'ny vohitra kely tany amin'ny morontsiraka",  # an 8-word run of LONG_EVAL
])
def test_eval_prompts_match_the_prompt_or_an_eight_word_run(text):
    assert EVAL.matches(text)


@pytest.mark.parametrize("text", [
    "Iza no filoha voalohany tany Madagasikara?",  # longer than the prompt, under 8 words shared
    "Iza no filoha?",
    "Lazao amiko ny tantaran'ny vohitra kely",  # 7 words of LONG_EVAL ("tantaran'ny" counts as 2)
    "",
])
def test_eval_prompts_leave_other_text(text):
    assert not EVAL.matches(text)


def test_eval_prompts_refuse_an_empty_set():
    with pytest.raises(ValueError):
        mix.EvalPrompts(["", "?!"])


def test_eval_prompt_is_the_first_drop_reason():
    assert mix.AYA_DROP_REASONS[0] == mix.REAL_DROP_REASONS[0] == "eval_prompt"
    kept, drops = mix.real_pairs([chat("Iza no filoha voalohany?", "Tsy haiko"), chat("Salama", "Salama tsara")],
                                 says_malagasy, RAW_NAMES, EVAL)  # fmt: skip
    assert [r["messages"][0]["content"] for r in kept] == ["Salama"]
    assert drops["eval_prompt"] == 1


def test_capped_counts_hold_each_share_of_the_whole_mix():
    caps = mix.capped_counts(700, {"real": 1000, "aya": 1000}, {"real": 0.1, "aya": 0.2})
    assert caps == {"real": 100, "aya": 200}
    total = 700 + caps["real"] + caps["aya"]
    assert caps["real"] <= 0.1 * total and caps["aya"] <= 0.2 * total


def test_capped_counts_give_a_short_source_all_it_has_and_recompute_the_rest():
    caps = mix.capped_counts(700, {"real": 30, "aya": 1000}, {"real": 0.1, "aya": 0.2})
    assert caps["real"] == 30
    assert caps["aya"] == int(0.2 * (700 + 30) / 0.8)


def test_capped_counts_refuse_shares_of_one_or_more():
    with pytest.raises(ValueError):
        mix.capped_counts(10, {"a": 5, "b": 5}, {"a": 0.5, "b": 0.5})


def test_sample_is_seeded_and_keeps_order():
    rows = [{"id": str(i)} for i in range(50)]
    picked = mix.sample(rows, 10)
    assert picked == mix.sample(rows, 10) and len(picked) == 10
    assert [r["id"] for r in picked] == sorted((r["id"] for r in picked), key=int)
    assert mix.sample(rows, 99) == rows


def test_repeat_counts_in_mix_rows():
    rows = {"identity": mix.with_repeat([{"id": "a"}, {"id": "b"}], 5), "synth": [{"id": "c"}]}
    assert mix.mix_rows(rows) == 11
