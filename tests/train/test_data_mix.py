import random

import pytest

from fym1.eval.perplexity import held_out_rows
from fym1.train.data_mix import data_files, split_rows


def test_data_files_under_a_folder_in_repo_order():
    repo = ["chats.jsonl", "mix-r2/synth.jsonl", "mix-r2/mix_stats.json", "mix-r2/aya.jsonl", "system_prompt.txt"]
    assert data_files(repo, "mix-r2/") == ["mix-r2/synth.jsonl", "mix-r2/aya.jsonl"]
    assert data_files(repo, "") == ["chats.jsonl", "mix-r2/synth.jsonl", "mix-r2/aya.jsonl"]


def test_data_files_refuse_a_bad_or_empty_folder():
    with pytest.raises(ValueError):
        data_files(["mix-r2/a.jsonl"], "mix-r2")
    with pytest.raises(ValueError):
        data_files(["chats.jsonl"], "mix-r3/")


def test_split_without_repeat_is_the_r1_split():
    rows = [{"id": str(i)} for i in range(100)]
    expected = list(rows)
    random.Random(42).shuffle(expected)
    train, eval_rows = split_rows(rows, 0.02, 42)
    assert eval_rows == expected[:2] and train == expected[2:]
    assert eval_rows == held_out_rows(rows, 0.02, 42)  # 02_eval_export finds the same held-out rows
    assert rows == [{"id": str(i)} for i in range(100)]


def test_repeat_multiplies_train_rows_only():
    rows = [{"id": str(i), "repeat": 3} for i in range(50)]
    train, eval_rows = split_rows(rows, 0.02, 42)
    assert len(eval_rows) == 1 and len(train) == 49 * 3
    assert eval_rows[0]["id"] not in {r["id"] for r in train}


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", True])
def test_bad_repeat_is_refused(bad):
    with pytest.raises(ValueError):
        split_rows([{"id": "a"}, {"id": "b", "repeat": bad}, {"id": "c"}], 0.0001, 1)
