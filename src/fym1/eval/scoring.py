"""Build results/eval/scores.json and results/eval/judge.json (schema read by paper/make_numbers.py).

Answer files are `answers_<model>.jsonl`, one row per prompt: `{"id", "prompt", "response"}`.
scores.json: `{"n_prompts", "<model>": {"drift_rate", <NOTEBOOK_METRICS>...}}`.
judge.json:  `{"judge", "<model>": {"naturalness", "correctness", "n"}}`.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from fym1.eval.metrics import drift_rate

JUDGE_CRITERIA = ("naturalness", "correctness")
ANSWERS_PREFIX = "answers_"
# Metrics measured in notebooks/02_eval_export.ipynb; any of them may still be missing (pending).
NOTEBOOK_METRICS = frozenset(
    {"leak_rate", "perplexity", "belebele_acc", "chrf_plt_eng", "chrf_eng_plt"}
)
N_PROMPTS = "n_prompts"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def model_name(answers_path: Path) -> str:
    """`answers_fym1.jsonl` -> `fym1`."""
    stem = answers_path.stem
    if not stem.startswith(ANSWERS_PREFIX) or stem == ANSWERS_PREFIX:
        raise ValueError(f"expected answers_<model>.jsonl, got {answers_path.name}")
    return stem.removeprefix(ANSWERS_PREFIX)


def blind_batch(answers: dict[str, list[dict]], seed: int) -> tuple[list[dict], dict[str, dict]]:
    """Mix every model's answers in a seeded random order with neutral ids.

    Returns the items the judge sees (`id`, `prompt`, `response` only) and the key
    that maps each neutral id back to `{"model", "prompt_id"}`.
    """
    pooled = [(model, row) for model, model_rows in answers.items() for row in model_rows]
    random.Random(seed).shuffle(pooled)
    items, key = [], {}
    for i, (model, row) in enumerate(pooled, 1):
        blind_id = f"j{i:04d}"
        items.append({"id": blind_id, "prompt": row["prompt"], "response": row["response"]})
        key[blind_id] = {"model": model, "prompt_id": row["id"]}
    return items, key


def _check_score(judgement: dict, criterion: str) -> int:
    score = judgement.get(criterion)
    if type(score) is not int or not 1 <= score <= 5:
        raise ValueError(
            f"{judgement.get('id')}: {criterion} must be an integer 1-5, got {score!r}"
        )
    return score


def judge_means(judgements: list[dict], key: dict[str, dict]) -> dict[str, dict]:
    """Mean judge score per model and criterion, unblinded with the key from `blind_batch`."""
    scores: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    for judgement in judgements:
        blind_id = judgement.get("id")
        if blind_id not in key:
            raise ValueError(f"judgement for unknown id {blind_id!r}")
        if blind_id in seen:
            raise ValueError(f"id {blind_id!r} judged twice")
        seen.add(blind_id)
        model = key[blind_id]["model"]
        for criterion in JUDGE_CRITERIA:
            scores[model][criterion].append(_check_score(judgement, criterion))
    return {
        model: {c: float(mean(by_criterion[c])) for c in JUDGE_CRITERIA}
        | {"n": len(by_criterion[JUDGE_CRITERIA[0]])}
        for model, by_criterion in sorted(scores.items())
    }


def judge_summary(means: dict[str, dict], judge: str) -> dict:
    """judge.json content: who judged, then the per-model means from `judge_means`."""
    if not judge.strip():
        raise ValueError("judge label is empty; say who judged (e.g. in-session agents)")
    if "judge" in means:
        raise ValueError("model name 'judge' is reserved")
    return {"judge": judge} | means


def _check_metrics(metrics: dict[str, dict], models: set[str]) -> None:
    unknown_models = set(metrics) - models
    if unknown_models:
        raise ValueError(f"metrics for unknown model(s): {sorted(unknown_models)}")
    unknown_names = {name for values in metrics.values() for name in values} - NOTEBOOK_METRICS
    if unknown_names:
        raise ValueError(
            f"unknown metric name(s) {sorted(unknown_names)}; expected {sorted(NOTEBOOK_METRICS)}"
        )


def assemble_scores(answers: dict[str, list[dict]], metrics: dict[str, dict]) -> dict:
    """scores.json content: drift_rate per model from its answers, plus notebook metrics.

    Every model must have answered the same prompts, so the numbers are comparable.
    """
    if not answers:
        raise ValueError("no answers to score")
    if N_PROMPTS in answers:
        raise ValueError(f"model name {N_PROMPTS!r} is reserved")
    prompt_ids = {model: sorted(row["id"] for row in rows) for model, rows in answers.items()}
    if len({tuple(ids) for ids in prompt_ids.values()}) != 1:
        raise ValueError("every model must answer the same prompt ids")
    _check_metrics(metrics, set(answers))

    scores = {N_PROMPTS: len(next(iter(prompt_ids.values())))}
    for model, rows in sorted(answers.items()):
        scores[model] = {"drift_rate": drift_rate([row["response"] for row in rows])}
        scores[model] |= metrics.get(model, {})
    return scores
