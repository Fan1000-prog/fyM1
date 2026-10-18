"""Blind greeting gate: module and CLI, on made-up answers only."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fym1.eval.greeting_gate import blind, gate_rows, greeting_answers

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "greeting_gate.py"
PROMPTS = ["Salama e!", "De aona?"]


def baseline(slug: str, answers: list[str], prompts: list[str] = PROMPTS) -> dict:
    return {
        "model": {"slug": slug, "hub_id": f"org/{slug}"},
        "greetings": {"source": "eval/prompts.jsonl greetings", "sampling": "temp 0.7",
                      "answers": [{"prompt": p, "answer": a} for p, a in zip(prompts, answers)]},
    }


MODEL_A = baseline("model-a", ["Salama tsara 😅 Ary ianao?", "Tsara be fa reraka kely"])
MODEL_B = baseline("model-b", ["Hello, how are you?", "  Je vais bien, merci !\n"])


def judgement(blind_id: str, nat: int, cor: int, note: str = "ok") -> dict:
    return {"id": blind_id, "naturalness": nat, "correctness": cor, "note": note}


def test_greeting_answers_numbers_prompts_by_position():
    slug, rows = greeting_answers(MODEL_A)
    assert slug == "model-a"
    assert [r["id"] for r in rows] == ["g01", "g02"]
    assert rows[1] == {"id": "g02", "prompt": "De aona?", "response": "Tsara be fa reraka kely"}


def test_greeting_answers_rejects_missing_section():
    with pytest.raises(ValueError, match="no greetings"):
        greeting_answers({"model": {"slug": "x"}})


def test_blind_hides_model_names_and_keeps_every_answer():
    items, key, _ = blind([MODEL_A, MODEL_B], seed=1)
    assert len(items) == len(key) == 4
    assert all(set(item) == {"id", "prompt", "response"} for item in items)
    text = json.dumps(items)
    assert "model-a" not in text and "model-b" not in text
    assert sorted(v["model"] for v in key.values()) == ["model-a", "model-a", "model-b", "model-b"]
    assert "Je vais bien, merci !" in {item["response"] for item in items}  # whitespace stripped


def test_blind_is_seeded():
    assert blind([MODEL_A, MODEL_B], seed=7) == blind([MODEL_A, MODEL_B], seed=7)
    orders = {tuple(v["model"] for v in blind([MODEL_A, MODEL_B], seed=s)[1].values()) for s in range(10)}
    assert len(orders) > 1


def test_blind_lists_reworded_prompts():
    other = baseline("model-c", ["a", "b"], prompts=["Salama e!", "De aona ry zalahy?"])
    assert blind([MODEL_A, other], seed=1)[2] == ["g02"]
    assert blind([MODEL_A, MODEL_B], seed=1)[2] == []


def test_blind_rejects_a_different_prompt_count():
    short = baseline("model-c", ["a"], prompts=["Salama e!"])
    with pytest.raises(ValueError, match="same prompt ids"):
        blind([MODEL_A, short], seed=1)


def test_blind_rejects_same_model_twice():
    with pytest.raises(ValueError, match="given twice"):
        blind([MODEL_A, MODEL_A], seed=1)


def scored(items, key, nat_a=5, nat_b=1, cor=4):
    return [judgement(i["id"], nat_a if key[i["id"]]["model"] == "model-a" else nat_b, cor) for i in items]


def test_gate_rows_per_model():
    items, key, _ = blind([MODEL_A, MODEL_B], seed=3)
    rows = gate_rows(items, key, scored(items, key))
    assert rows["model-a"] == {"naturalness": 5.0, "correctness": 4.0, "judge_mean": 4.5,
                               "is_malagasy_rate": 1.0, "drift_rate": 0.0, "n": 2}
    assert rows["model-b"]["judge_mean"] == 2.5
    assert rows["model-b"]["is_malagasy_rate"] == 0.0
    assert rows["model-b"]["drift_rate"] == 1.0


def test_gate_rows_reports_shared_prompts_apart():
    items, key, _ = blind([MODEL_A, MODEL_B], seed=3)
    judged = [judgement(i["id"], 5 if key[i["id"]]["prompt_id"] == "g01" else 1, 3) for i in items]
    rows = gate_rows(items, key, judged, variant_ids=["g02"])
    assert rows["model-a"]["naturalness"] == 3.0
    assert rows["model-a"]["shared_prompts"] == {"naturalness": 5.0, "correctness": 3.0, "judge_mean": 4.0, "n": 1}
    assert "shared_prompts" not in gate_rows(items, key, judged)["model-a"]


def test_gate_rows_rejects_partial_judging():
    items, key, _ = blind([MODEL_A, MODEL_B], seed=3)
    with pytest.raises(ValueError, match="partial"):
        gate_rows(items, key, scored(items, key)[:-1])


def test_gate_rows_rejects_out_of_range_score():
    items, key, _ = blind([MODEL_A, MODEL_B], seed=3)
    judged = scored(items, key)
    judged[0]["naturalness"] = 6
    with pytest.raises(ValueError, match="1-5"):
        gate_rows(items, key, judged)


def test_gate_rows_requires_a_reason():
    items, key, _ = blind([MODEL_A, MODEL_B], seed=3)
    judged = scored(items, key)
    judged[0]["note"] = " "
    with pytest.raises(ValueError, match="note"):
        gate_rows(items, key, judged)


def test_gate_rows_rejects_key_mismatch():
    items, key, _ = blind([MODEL_A, MODEL_B], seed=3)
    with pytest.raises(ValueError, match="same ids"):
        gate_rows(items[:-1], key, scored(items, key))


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True)


def test_cli_blind_then_score(tmp_path):
    paths = []
    for result in (MODEL_A, MODEL_B):
        path = tmp_path / f"{result['model']['slug']}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        paths.append(str(path))
    done = run("blind", *paths, "--dir", "gate", "--seed", "5", cwd=tmp_path)
    assert done.returncode == 0, done.stderr

    blind_text = (tmp_path / "gate" / "blind.jsonl").read_text(encoding="utf-8")
    assert "model-a" not in blind_text and "model-b" not in blind_text
    key = json.loads((tmp_path / "gate" / "key.json").read_text(encoding="utf-8"))
    assert key["seed"] == 5
    assert key["variant_prompt_ids"] == []
    items = [json.loads(line) for line in blind_text.splitlines()]
    scores = scored(items, key["key"])
    (tmp_path / "gate" / "scores.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores))

    done = run("score", "--judge", "test judge", "--dir", "gate", "--out", "gate.json", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    report = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert report["judge"] == "test judge"
    assert report["blind_seed"] == 5
    assert report["models"]["model-a"]["n"] == 2
    assert report["models"]["model-b"]["drift_rate"] == 1.0


def test_cli_score_needs_a_judge_label(tmp_path):
    done = run("score", "--judge", " ", cwd=tmp_path)
    assert done.returncode != 0
    assert "judge" in done.stderr
