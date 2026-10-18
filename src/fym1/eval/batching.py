"""Batched GPU work that survives an out-of-memory batch without inventing results."""

from collections.abc import Callable, Sequence


class StillOutOfMemory(Exception):
    """Out of memory even one item at a time: the caller records the section as "oom", never a number."""


def run_batched(items: Sequence, batch_size: int, run: Callable[[list], list],
                is_oom: Callable[[Exception], bool], clear: Callable[[], None]) -> list:
    """`run` maps a batch to one result per item. A batch that runs out of memory is retried one item at a
    time after `clear()` (e.g. torch.cuda.empty_cache); results keep the input order."""
    out = []
    for i in range(0, len(items), batch_size):
        batch = list(items[i:i + batch_size])
        try:
            out += run(batch)
            continue
        except Exception as e:
            if not is_oom(e):
                raise
        clear()
        for j, item in enumerate(batch):
            try:
                out += run([item])
            except Exception as e:
                if not is_oom(e):
                    raise
                raise StillOutOfMemory(f"out of memory at batch size 1 (item {i + j})") from None
    return out
