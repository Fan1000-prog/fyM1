"""Which data files a run reads, and the train/eval split with upweighted rows.

A row may carry `"repeat": k` (the r2 mix gives the identity chats one): it is trained on k times. The split is on
distinct rows first, so a repeated row is never in both train and eval, and an eval row counts once. Rows without
`repeat` give exactly the r1 split (shuffle with the seed, the first `round(n * eval_share)` rows are eval).
"""

import random


def data_files(repo_files: list[str], data_dir: str) -> list[str]:
    """The `*.jsonl` files under `data_dir` ("" = every `*.jsonl` in the repo, the r1 behaviour), in repo order."""
    if data_dir and not data_dir.endswith("/"):
        raise ValueError(f"data_dir must end with '/': {data_dir!r}")
    files = [f for f in repo_files if f.endswith(".jsonl") and f.startswith(data_dir)]
    if not files:
        raise ValueError(f"no .jsonl files under {data_dir!r}")
    return files


def repeat_of(row: dict) -> int:
    repeat = row.get("repeat", 1)
    if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
        raise ValueError(f"row {row.get('id')!r}: repeat must be a positive int, got {repeat!r}")
    return repeat


def split_rows(rows: list[dict], eval_share: float, seed: int) -> tuple[list[dict], list[dict]]:
    """(train rows with each row repeated `repeat` times, eval rows once each); the input list is not changed."""
    for row in rows:  # a bad repeat is refused wherever the row lands
        repeat_of(row)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_eval = max(1, round(len(shuffled) * eval_share))
    eval_rows, train_once = shuffled[:n_eval], shuffled[n_eval:]
    train = [row for row in train_once for _ in range(repeat_of(row))]
    return train, eval_rows
