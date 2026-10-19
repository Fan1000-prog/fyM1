"""Held-out-chat perplexity for notebooks/02_eval_export.ipynb, measured the way training scored eval loss.

01_train holds out the first `max(1, round(n * share))` rows after `random.Random(seed).shuffle` of every
training row, and trains (and scores eval loss) on assistant replies only (Unsloth `train_on_responses_only`
with `fym1.train.chat_format.turn_markers`). Perplexity here counts the same tokens: those after each
response header up to the next user header, so `perplexity(fym1) ~ exp(best eval loss)`.
"""

import math
import random


def held_out_rows(rows: list, share: float, seed: int) -> list:
    """The eval split of 01_train: the first max(1, round(n * share)) rows of a seeded shuffle (input untouched)."""
    if not rows:
        raise ValueError("no rows to split")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:max(1, round(len(shuffled) * share))]


def response_spans(text: str, instruction_part: str, response_part: str) -> list[tuple[int, int]]:
    """Character spans of assistant replies: after each response header, up to the next user header or the end."""
    if not instruction_part or not response_part:
        raise ValueError("instruction_part and response_part must be non-empty")
    spans = []
    start = text.find(response_part)
    while start != -1:
        begin = start + len(response_part)
        end = text.find(instruction_part, begin)
        end = len(text) if end == -1 else end
        spans.append((begin, end))
        start = text.find(response_part, end)
    return spans


def token_mask(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[bool]:
    """True for tokens (char offsets from a fast tokenizer) that start inside a reply span; padding (0, 0) is False."""
    return [end > start and any(lo <= start < hi for lo, hi in spans) for start, end in offsets]


def perplexity(nll_sum: float, n_tokens: int) -> float:
    """exp(mean negative log-likelihood per scored token)."""
    if n_tokens <= 0:
        raise ValueError("no scored tokens")
    return math.exp(nll_sum / n_tokens)
