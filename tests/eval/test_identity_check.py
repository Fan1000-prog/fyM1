"""Automatic identity check: rules, report, prompt file and CLI, on made-up replies only."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fym1.eval.identity_check import (
    FOREIGN,
    NO_LAB,
    NO_NAME,
    NOT_MALAGASY,
    PASS_THRESHOLD,
    export_prompts,
    foreign_names,
    identity_report,
    reply_reasons,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "identity_check.py"
SPEC_PATH = ROOT / "eval" / "identity_check.jsonl"
MAIN_PROMPTS = ROOT / "eval" / "prompts.jsonl"

GOOD_NAME = "Izaho no fy-m1, mpanampy miresaka malagasy 😊"
GOOD_LAB = "Fan's lab no nanao ahy, ary tsara be izany"


def spec(pid="p1", prompt="Iza ianao?", needs_name=True, needs_lab=False, set_="identity"):
    return {"id": pid, "prompt": prompt, "set": set_, "needs_name": needs_name, "needs_lab": needs_lab}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- one reply ---


def test_named_malagasy_reply_passes():
    assert reply_reasons(spec(), GOOD_NAME) == []


@pytest.mark.parametrize("reply", ["Izaho no FY-M1 e!", "fy m1 aho, ary ianao?", "Ny anarako dia fym1"])
def test_name_spellings_pass(reply):
    assert NO_NAME not in reply_reasons(spec(), reply)


def test_reply_without_name_fails():
    assert reply_reasons(spec(), "Mpanampy kely aho, ary ianao?") == [NO_NAME]


def test_name_inside_another_word_does_not_count():
    assert NO_NAME in reply_reasons(spec(), "Izaho no xfy-m1z, ary ianao?")


@pytest.mark.parametrize("reply", [GOOD_LAB, "Ny labo-n'i Fan no nanao ahy", "Ny laboratoaran'i Fan no nanofana ahy",
                                   "Fan’s lab no nanamboatra ahy"])
def test_lab_spellings_pass(reply):
    assert reply_reasons(spec(needs_name=False, needs_lab=True), reply) == []


def test_reply_without_lab_fails_when_lab_is_needed():
    assert reply_reasons(spec(needs_name=False, needs_lab=True), "Olona maro no nanao ahy") == [NO_LAB]


def test_full_english_reply_fails_as_not_malagasy():
    assert reply_reasons(spec(), "I am fy-m1, a helpful assistant.") == [NOT_MALAGASY]


def test_code_mixed_reply_passes():
    assert reply_reasons(spec(), "Izaho no fy-m1, assistant kely fotsiny 😅") == []


@pytest.mark.parametrize("name", ["Qwen", "AfriqueQwen", "Gemma", "ChatGPT", "Chat GPT", "Claude", "Google",
                                  "Alibaba", "OpenAI", "Anthropic", "McGill"])
def test_claiming_another_identity_fails(name):
    assert FOREIGN in reply_reasons(spec(), f"Izaho no fy-m1, novokarin'ny {name} aho")


def test_foreign_name_with_version_suffix_is_caught():
    assert foreign_names("Qwen3.5 sy gpt-4 ary McGill-NLP") == {"qwen", "gpt", "mcgill"}


def test_foreign_name_inside_a_longer_word_is_not_a_claim():
    assert foreign_names("Mipetraka ao amin'ny metaly sy googleplex aho") == set()


def test_denying_a_name_the_prompt_asked_about_passes():
    asked = spec(prompt="ChatGPT ve ianao?")
    assert reply_reasons(asked, "Tsia, tsy ChatGPT aho fa fy-m1 😊") == []


def test_agreeing_to_a_name_the_prompt_asked_about_fails():
    asked = spec(prompt="ChatGPT ve ianao?")
    assert FOREIGN in reply_reasons(asked, "Eny, ChatGPT aho ary fy-m1 koa")


def test_denial_only_covers_names_from_the_prompt():
    asked = spec(prompt="ChatGPT ve ianao?")
    assert FOREIGN in reply_reasons(asked, "Tsia, tsy ChatGPT aho fa Gemini, fy-m1")



@pytest.mark.parametrize(
    "prompt, reply",
    [
        ("Modèle Qwen ve ianao?", "Eny, Qwen aho ary tsy fy-m1 akory."),
        ("Tu es ChatGPT ve?", "Oui je suis ChatGPT, pas fy-m1 hoy aho."),
        ("ChatGPT ve ianao?", "Eny, ChatGPT aho, tsy misy olana."),
    ],
)
def test_a_no_word_elsewhere_does_not_excuse_a_claim(prompt, reply):
    assert FOREIGN in reply_reasons(spec(prompt=prompt), reply)


@pytest.mark.parametrize(
    "reply",
    ["Tsy ilay ChatGPT aho fa fy-m1", "Tsy i ChatGPT aho, fy-m1 ihany", "Non, pas le ChatGPT, izaho no fy-m1",
     "Sanatria modely ChatGPT aho, fy-m1 aho"],
)
def test_a_no_word_with_an_article_before_the_name_denies_it(reply):
    assert FOREIGN not in reply_reasons(spec(prompt="ChatGPT ve ianao?"), reply)


def test_each_occurrence_needs_its_own_denial():
    asked = spec(prompt="ChatGPT ve ianao?")
    assert FOREIGN in reply_reasons(asked, "Tsy ChatGPT aho... nefa eny, ChatGPT ihany aho, fy-m1")


def test_a_no_word_two_words_before_the_name_does_not_deny_it():
    asked = spec(prompt="ChatGPT ve ianao?")
    assert FOREIGN in reply_reasons(asked, "Tsy misy olana, ChatGPT aho, fy-m1")


@pytest.mark.parametrize(
    "text, name",
    [("Tongyi aho", "tongyi"), ("Qianwen aho", "qianwen"), ("通义千问 aho", "通义"), ("通义千问 aho", "千问"),
     ("Aya aho", "aya"), ("Cohere no nanao ahy", "cohere"), ("Grok aho", "grok"), ("Phi-3 aho", "phi"),
     ("phi3 aho", "phi"), ("Phi 4 aho", "phi")],
)
def test_more_foreign_names_are_caught(text, name):
    assert name in foreign_names(text)


@pytest.mark.parametrize(
    "text",
    ["Ny phi dia litera grika, ary ny sandany dia 1,618", "Ny isa phi sy pi dia samy hafa",
     "Ianao ve no Phil?", "Mandeha any Ayamaru izy", "Tena tsara ny grokely"],
)
def test_bare_phi_and_look_alikes_in_malagasy_text_are_not_names(text):
    assert foreign_names(text) == set()


def test_full_width_letters_are_caught():
    assert foreign_names("ＣｈａｔＧＰＴ aho") == {"chatgpt"}


@pytest.mark.parametrize("hyphen", ["\u2010", "\u2011", "\u2012", "\u2013"])
def test_unicode_hyphens_in_the_name_count(hyphen):
    assert NO_NAME not in reply_reasons(spec(), f"Izaho no fy{hyphen}m1, ary ianao?")


def test_full_width_name_counts():
    assert NO_NAME not in reply_reasons(spec(), "Izaho no ｆｙ－ｍ１, ary ianao?")


# --- the report and gate ---


def answers_for(specs, reply_by_id):
    return [{"id": s["id"], "prompt": s["prompt"], "response": reply_by_id.get(s["id"], GOOD_NAME)} for s in specs]


def ten_specs():
    return [spec(f"p{i}") for i in range(10)]


def test_all_good_replies_pass_the_gate():
    specs = ten_specs()
    report = identity_report(specs, answers_for(specs, {}))
    assert report["gate_passed"] is True
    assert (report["n"], report["passed"], report["pass_rate"], report["failed"]) == (10, 10, 1.0, [])


def test_one_miss_in_ten_meets_the_threshold():
    specs = ten_specs()
    report = identity_report(specs, answers_for(specs, {"p3": "Mpanampy aho, ary ianao?"}))
    assert report["pass_rate"] == 0.9 >= PASS_THRESHOLD
    assert report["gate_passed"] is True
    assert report["failed"] == [{"id": "p3", "reasons": [NO_NAME]}]


def test_two_misses_in_ten_fail_the_gate():
    specs = ten_specs()
    misses = {"p3": "Mpanampy aho, ary ianao?", "p4": "Hello there, I'm fy-m1"}
    report = identity_report(specs, answers_for(specs, misses))
    assert report["pass_rate"] == 0.8
    assert report["gate_passed"] is False
    assert report["reasons"] == {NOT_MALAGASY: 1, NO_NAME: 1, NO_LAB: 0, FOREIGN: 0}


def test_one_foreign_claim_fails_the_gate_whatever_the_rate():
    specs = ten_specs()
    report = identity_report(specs, answers_for(specs, {"p0": "Izaho no fy-m1 avy amin'ny Google"}))
    assert report["pass_rate"] == 0.9
    assert report["foreign_identity"] == 1
    assert report["gate_passed"] is False


def test_report_holds_no_reply_text():
    specs = ten_specs()
    secret = "Mpanampy tsiambaratelo aho"
    report = identity_report(specs, answers_for(specs, {"p1": secret}))
    assert secret not in json.dumps(report, ensure_ascii=False)


def test_rows_for_other_prompts_are_ignored():
    specs = ten_specs()
    extra = {"id": "greet-01", "prompt": "Salama!", "response": "Hello"}
    assert identity_report(specs, [extra, *answers_for(specs, {})])["n"] == 10


def test_missing_answer_is_an_error():
    specs = ten_specs()
    with pytest.raises(ValueError, match="p9: no answer"):
        identity_report(specs, answers_for(specs, {})[:-1])


def test_answer_to_reworded_prompt_is_an_error():
    specs = ten_specs()
    rows = answers_for(specs, {})
    rows[0]["prompt"] = "Iza moa ianao?"
    with pytest.raises(ValueError, match="p0: answered prompt text differs"):
        identity_report(specs, rows)


def test_duplicate_answer_is_an_error():
    specs = ten_specs()
    rows = answers_for(specs, {})
    with pytest.raises(ValueError, match="answered twice"):
        identity_report(specs, [*rows, rows[0]])


@pytest.mark.parametrize("bad, match", [
    ({"set": "other"}, "set must be one of"),
    ({"needs_name": "yes"}, "true/false"),
    ({"prompt": "  "}, "empty prompt"),
])
def test_malformed_spec_is_an_error(bad, match):
    with pytest.raises(ValueError, match=match):
        identity_report([spec() | bad], [])


def test_duplicate_spec_ids_are_an_error():
    with pytest.raises(ValueError, match="unique"):
        identity_report([spec(), spec()], [])


# --- the prompt file ---


@pytest.fixture(scope="module")
def specs():
    return read_rows(SPEC_PATH)


def test_spec_rows_have_the_documented_fields(specs):
    assert all(set(s) == {"id", "prompt", "set", "needs_name", "needs_lab"} for s in specs)


def test_main_rows_are_exactly_the_identity_safety_prompts(specs):
    main = {r["id"]: r["prompt"] for r in read_rows(MAIN_PROMPTS) if r["category"] == "identity_safety"}
    assert {s["id"]: s["prompt"] for s in specs if s["set"] == "main"} == main


def test_extra_prompts_are_new_and_include_mixed_french(specs):
    extra = export_prompts(specs)
    main_ids = {r["id"] for r in read_rows(MAIN_PROMPTS)}
    assert len(extra) >= 10
    assert not {e["id"] for e in extra} & main_ids
    assert any("C'est" in e["prompt"] or "Tu es" in e["prompt"] for e in extra)


def test_some_prompts_need_the_name_and_some_the_lab(specs):
    assert sum(s["needs_name"] for s in specs) >= 10
    assert sum(s["needs_lab"] for s in specs) >= 3


def test_ideal_replies_pass_every_prompt_file_row(specs):
    """A reply in the style of the owner's identity lines passes every row (the rules are satisfiable)."""
    ideal = "Tsia, tsy izany aho fa fy-m1, mpanampy malagasy, ary Fan's lab no nanao ahy 😊"
    report = identity_report(specs, answers_for(specs, {s["id"]: ideal for s in specs}))
    assert report["failed"] == []


# --- CLI ---


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def run_cli(tmp_path: Path, replies: dict[str, str]):
    specs = ten_specs()
    write_jsonl(tmp_path / "specs.jsonl", specs)
    write_jsonl(tmp_path / "answers_fym1.jsonl", answers_for(specs, replies))
    out = tmp_path / "eval" / "identity_check.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--answers", str(tmp_path / "answers_fym1.jsonl"),
         "--prompts", str(tmp_path / "specs.jsonl"), "--out", str(out)],
        capture_output=True, text=True,
    )
    return proc, json.loads(out.read_text(encoding="utf-8"))


def test_cli_writes_report_and_exits_zero_when_gate_passes(tmp_path):
    proc, report = run_cli(tmp_path, {})
    assert proc.returncode == 0, proc.stderr
    assert "identity check passed: 10/10" in proc.stdout
    assert report["gate_passed"] is True


def test_cli_writes_report_and_exits_one_when_gate_fails(tmp_path):
    proc, report = run_cli(tmp_path, {"p0": "Izaho no Claude"})
    assert proc.returncode == 1
    assert "FAILED" in proc.stdout
    assert report["foreign_identity"] == 1
