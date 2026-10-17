import json

import pytest

from fym1.eval.scoring import (
    assemble_scores,
    blind_batch,
    chat_texts,
    judge_means,
    judge_summary,
    leak_gate_passes,
    leak_review,
    model_name,
    read_jsonl,
)


def rows(*responses):
    return [{"id": f"p{i}", "prompt": f"prompt {i}", "response": r} for i, r in enumerate(responses)]


ANSWERS = {"base": rows("Hello there", "Salama!"), "fym1": rows("De aona?", "Salama!")}


def test_model_name_from_answers_file(tmp_path):
    assert model_name(tmp_path / "answers_fym1.jsonl") == "fym1"


def test_model_name_rejects_other_files(tmp_path):
    with pytest.raises(ValueError):
        model_name(tmp_path / "scores.jsonl")


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"id": "p0"}\n\n{"id": "p1"}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"id": "p0"}, {"id": "p1"}]


def test_blind_batch_hides_model_and_covers_every_answer():
    items, key = blind_batch(ANSWERS, seed=42)
    assert len(items) == 4
    assert all(set(item) == {"id", "prompt", "response"} for item in items)
    assert sorted((k["model"], k["prompt_id"]) for k in key.values()) == [
        ("base", "p0"), ("base", "p1"), ("fym1", "p0"), ("fym1", "p1"),
    ]


def test_blind_batch_is_reproducible_and_shuffled():
    items_a, key_a = blind_batch(ANSWERS, seed=42)
    items_b, key_b = blind_batch(ANSWERS, seed=42)
    assert (items_a, key_a) == (items_b, key_b)
    order = [key_a[item["id"]]["model"] for item in items_a]
    assert order != ["base", "base", "fym1", "fym1"]


def test_judge_means_per_model():
    _, key = blind_batch(ANSWERS, seed=0)
    judgements = [
        {"id": blind_id, "naturalness": 2 if k["model"] == "base" else 5, "correctness": 4}
        for blind_id, k in key.items()
    ]
    assert judge_means(judgements, key) == {
        "base": {"naturalness": 2.0, "correctness": 4.0, "n": 2},
        "fym1": {"naturalness": 5.0, "correctness": 4.0, "n": 2},
    }


def test_judge_means_rejects_unknown_id():
    _, key = blind_batch(ANSWERS, seed=0)
    with pytest.raises(ValueError, match="unknown"):
        judge_means([{"id": "nope", "naturalness": 3, "correctness": 3}], key)


@pytest.mark.parametrize("score", [0, 6, 3.5, "4"])
def test_judge_means_rejects_out_of_range_scores(score):
    _, key = blind_batch(ANSWERS, seed=0)
    blind_id = next(iter(key))
    with pytest.raises(ValueError, match="1-5"):
        judge_means([{"id": blind_id, "naturalness": score, "correctness": 3}], key)


def test_judge_means_rejects_duplicate_judgement():
    _, key = blind_batch(ANSWERS, seed=0)
    blind_id = next(iter(key))
    row = {"id": blind_id, "naturalness": 3, "correctness": 3}
    with pytest.raises(ValueError, match="twice"):
        judge_means([row, row], key)


def test_judge_summary_labels_the_judge():
    means = {"base": {"naturalness": 2.0, "correctness": 4.0, "n": 2}}
    assert judge_summary(means, judge="in-session agents") == {
        "judge": "in-session agents",
        "base": {"naturalness": 2.0, "correctness": 4.0, "n": 2},
    }


def test_judge_summary_requires_a_label():
    with pytest.raises(ValueError, match="judge"):
        judge_summary({}, judge=" ")


def test_judge_summary_rejects_reserved_model_name():
    with pytest.raises(ValueError, match="reserved"):
        judge_summary({"judge": {"naturalness": 3.0, "correctness": 3.0, "n": 1}}, judge="x")


def test_assemble_scores_flat_per_model_with_drift_and_metrics():
    scores = assemble_scores(ANSWERS, metrics={"fym1": {"leak_rate": 0.0, "belebele_acc": 0.61}})
    assert scores == {
        "n_prompts": 2,
        "base": {"drift_rate": 0.5},
        "fym1": {"drift_rate": 0.0, "leak_rate": 0.0, "belebele_acc": 0.61},
    }
    json.dumps(scores)  # must be writable as-is


def test_assemble_scores_requires_same_prompts_for_every_model():
    answers = {"base": rows("Salama!"), "fym1": rows("Salama!", "De aona?")}
    with pytest.raises(ValueError, match="same prompt ids"):
        assemble_scores(answers, metrics={})


def test_assemble_scores_rejects_metrics_for_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        assemble_scores(ANSWERS, metrics={"gpt": {"leak_rate": 0.0}})


def test_assemble_scores_rejects_unknown_metric_name():
    with pytest.raises(ValueError, match="unknown metric"):
        assemble_scores(ANSWERS, metrics={"fym1": {"belebele": 0.6}})


def test_assemble_scores_rejects_reserved_model_name():
    with pytest.raises(ValueError, match="n_prompts"):
        assemble_scores({"n_prompts": rows("Salama!")}, metrics={})


def test_chat_texts_takes_every_message_content():
    records = [
        {"id": "c1", "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]},
        {"id": "c2", "messages": [{"role": "user", "content": "z"}]},
    ]
    assert chat_texts(records) == ["x", "y", "z"]


def test_leak_review_starts_with_nothing_reviewed():
    matches = [{"ngram": "a b c d e f g h", "ids": ["p1"], "verdict": None}]
    assert leak_review(matches) == {"matches": 1, "approved_generic": 0, "rejected": 0}


def test_leak_review_counts_owner_verdicts():
    matches = [
        {"ngram": "n1", "ids": ["p1"], "verdict": "generic"},
        {"ngram": "n2", "ids": ["p1"], "verdict": "rejected"},
        {"ngram": "n3", "ids": ["p2", "p3"], "verdict": "generic"},
        {"ngram": "n4", "ids": ["p3"]},
    ]
    assert leak_review(matches) == {"matches": 4, "approved_generic": 2, "rejected": 1}


def test_leak_review_rejects_unknown_verdict():
    with pytest.raises(ValueError, match="verdict"):
        leak_review([{"ngram": "n1", "ids": ["p1"], "verdict": "ok"}])


def test_leak_review_rejects_a_duplicated_ngram():
    row = {"ngram": "n1", "ids": ["p1"], "verdict": "generic"}
    with pytest.raises(ValueError, match="more than once"):
        leak_review([row, row])


@pytest.mark.parametrize(
    "review, passes",
    [
        ({"matches": 0, "approved_generic": 0, "rejected": 0}, True),
        ({"matches": 2, "approved_generic": 2, "rejected": 0}, True),
        ({"matches": 2, "approved_generic": 0, "rejected": 0}, False),  # not reviewed yet
        ({"matches": 2, "approved_generic": 1, "rejected": 0}, False),  # partly reviewed
        ({"matches": 2, "approved_generic": 1, "rejected": 1}, False),  # a real leak
    ],
)
def test_leak_gate_passes_only_with_no_matches_or_all_generic(review, passes):
    assert leak_gate_passes(review) is passes


@pytest.mark.parametrize(
    "review",
    [
        {"matches": 1, "approved_generic": 2, "rejected": 0},
        {"matches": 1, "approved_generic": 1, "rejected": 1},
        {"matches": -1, "approved_generic": 0, "rejected": 0},
        {"matches": 1.0, "approved_generic": 1, "rejected": 0},
        {"matches": 1, "approved_generic": 1},
    ],
)
def test_leak_gate_rejects_inconsistent_review(review):
    with pytest.raises(ValueError):
        leak_gate_passes(review)
