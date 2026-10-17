"""Shape checks for the public test prompts (eval/prompts.jsonl)."""

import json
from collections import Counter
from pathlib import Path

import pytest

from fym1.eval.lang import is_malagasy

PROMPTS_PATH = Path(__file__).resolve().parents[2] / "eval" / "prompts.jsonl"
CATEGORY_COUNTS = {
    "greeting": 50,
    "everyday": 40,
    "banter": 20,
    "facts": 20,
    "identity_safety": 20,
}


@pytest.fixture(scope="module")
def prompts():
    with PROMPTS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_rows_have_exactly_id_prompt_category(prompts):
    assert all(set(row) == {"id", "prompt", "category"} for row in prompts)


def test_ids_are_unique(prompts):
    assert len({row["id"] for row in prompts}) == len(prompts)


def test_category_split_matches_plan(prompts):
    assert Counter(row["category"] for row in prompts) == CATEGORY_COUNTS


def test_prompts_are_unique_and_non_empty(prompts):
    texts = [row["prompt"].strip() for row in prompts]
    assert all(texts)
    assert len(set(texts)) == len(texts)


# Owner-approved prompts that is_malagasy gets wrong. Kept as written (is_malagasy is not
# tuned to the test set); they show where the drift metric undercounts Malagasy.
# greet-35: French plus the Malagasy particle "e", which the word list leaves out on purpose.
KNOWN_LANGUAGE_MISSES = {"greet-35"}


def test_every_prompt_is_malagasy_except_known_misses(prompts):
    misses = {row["id"] for row in prompts if not is_malagasy(row["prompt"])}
    assert misses == KNOWN_LANGUAGE_MISSES
