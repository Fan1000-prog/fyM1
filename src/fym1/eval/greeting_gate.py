"""Blind greeting gate for the base-model pick (replaces the owner's 1-5 greeting ratings).

`blind`: pool every candidate's baseline greeting answers, shuffle with a fixed seed, give each an
opaque id. The judge sees only `{"id", "prompt", "response"}`; a separate key maps id -> model.
`score`: unblind the judge's rubric scores (src/fym1/eval/judge_rubric.md) with the key and add the
automatic language checks, one row per model.
"""

from statistics import mean

from fym1.eval.lang import is_malagasy
from fym1.eval.metrics import drift_rate
from fym1.eval.scoring import JUDGE_CRITERIA, blind_batch, judge_means


def greeting_answers(result: dict) -> tuple[str, list[dict]]:
    """`(slug, rows)` from one results/baseline/<slug>.json; rows are `{"id", "prompt", "response"}`.

    The prompt id is the answer's position, so every model's n-th answer is the same prompt.
    """
    slug = result["model"]["slug"]
    answers = result.get("greetings", {}).get("answers")
    if not answers:
        raise ValueError(f"{slug}: baseline file has no greetings answers")
    rows = [{"id": f"g{i:02d}", "prompt": a["prompt"], "response": a["answer"]}
            for i, a in enumerate(answers, 1)]
    return slug, rows


def variant_prompt_ids(answers: dict[str, list[dict]]) -> list[str]:
    """Prompt ids whose text is not the same for every model (a run made with an older prompt file)."""
    by_id: dict[str, set[str]] = {}
    for rows in answers.values():
        for row in rows:
            by_id.setdefault(row["id"], set()).add(row["prompt"])
    return sorted(pid for pid, texts in by_id.items() if len(texts) > 1)


def blind(results: list[dict], seed: int) -> tuple[list[dict], dict[str, dict], list[str]]:
    """Blinded items, their key and the variant prompt ids, from the candidates' baseline results.

    The judge scores each reply against its own prompt, so a reworded prompt is still judged fairly;
    the variant ids are returned so `gate_rows` can also report the like-for-like shared prompts.
    """
    answers: dict[str, list[dict]] = {}
    for result in results:
        slug, rows = greeting_answers(result)
        if slug in answers:
            raise ValueError(f"model {slug!r} given twice")
        answers[slug] = rows
    items, key = blind_batch(answers, seed)
    return items, key, variant_prompt_ids(answers)


def _check_notes(judgements: list[dict]) -> None:
    for judgement in judgements:
        note = judgement.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"{judgement.get('id')}: judge note (short reason) is missing")


def _criteria_means(judgements: list[dict], key: dict[str, dict]) -> dict[str, dict]:
    """`judge_means` plus `judge_mean`, the mean of the criteria (every reply has both)."""
    means = judge_means(judgements, key)
    for model_means in means.values():
        model_means["judge_mean"] = float(mean(model_means[c] for c in JUDGE_CRITERIA))
    return means


def gate_rows(items: list[dict], key: dict[str, dict], judgements: list[dict],
              variant_ids: list[str] = ()) -> dict[str, dict]:
    """Per model: judge means, `judge_mean`, is_malagasy rate, drift rate, n.

    With `variant_ids`, each row also gets `shared_prompts`: the judge means over only the prompts
    every model saw with the same text.
    """
    if {item["id"] for item in items} != set(key):
        raise ValueError("blinded items and key do not hold the same ids")
    _check_notes(judgements)
    means = _criteria_means(judgements, key)
    shared_means = None
    if variant_ids:
        shared_key = {bid: v for bid, v in key.items() if v["prompt_id"] not in set(variant_ids)}
        shared_means = _criteria_means([j for j in judgements if j["id"] in shared_key], shared_key)
    responses: dict[str, list[str]] = {}
    for item in items:
        responses.setdefault(key[item["id"]]["model"], []).append(item["response"])
    rows = {}
    for model, model_means in means.items():
        replies = responses[model]
        rows[model] = {
            **{c: model_means[c] for c in (*JUDGE_CRITERIA, "judge_mean")},
            "is_malagasy_rate": sum(map(is_malagasy, replies)) / len(replies),
            "drift_rate": drift_rate(replies),
            "n": model_means["n"],
        }
        if shared_means is not None:
            rows[model]["shared_prompts"] = shared_means[model]
    return rows
