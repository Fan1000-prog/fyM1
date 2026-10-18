import pytest

from fym1.eval.batching import StillOutOfMemory, run_batched


class FakeOOM(Exception):
    pass


def is_oom(e):
    return isinstance(e, FakeOOM)


def test_runs_in_order_in_batches():
    seen = []

    def run(batch):
        seen.append(list(batch))
        return [x * 10 for x in batch]

    assert run_batched([1, 2, 3, 4, 5], 2, run, is_oom, lambda: None) == [10, 20, 30, 40, 50]
    assert seen == [[1, 2], [3, 4], [5]]


def test_oom_batch_is_cleared_and_retried_one_at_a_time():
    calls, cleared = [], []

    def run(batch):
        calls.append(list(batch))
        if len(batch) > 1 and 3 in batch:
            raise FakeOOM()
        return [x * 10 for x in batch]

    assert run_batched([1, 2, 3, 4], 2, run, is_oom, lambda: cleared.append(1)) == [10, 20, 30, 40]
    assert calls == [[1, 2], [3, 4], [3], [4]]
    assert cleared == [1]


def test_still_out_of_memory_at_batch_size_one():
    def run(batch):
        raise FakeOOM()

    with pytest.raises(StillOutOfMemory):
        run_batched([1, 2], 2, run, is_oom, lambda: None)


def test_other_errors_are_not_swallowed():
    def run(batch):
        raise ValueError("bug")

    with pytest.raises(ValueError):
        run_batched([1], 4, run, is_oom, lambda: None)


def test_on_retry_gets_the_size_of_each_retried_batch():
    retried = []

    def run(batch):
        if len(batch) > 1 and 3 in batch:
            raise FakeOOM()
        return batch

    run_batched([1, 2, 3, 4, 5], 2, run, is_oom, lambda: None, on_retry=retried.append)
    assert retried == [2]


@pytest.mark.parametrize("bad", [lambda batch: batch[:-1], lambda batch: batch + batch])
def test_result_count_must_match_the_batch(bad):
    with pytest.raises(ValueError, match="results for a batch"):
        run_batched([1, 2], 2, bad, is_oom, lambda: None)


def test_result_count_is_checked_on_the_one_at_a_time_retry():
    def run(batch):
        if len(batch) > 1:
            raise FakeOOM()
        return []

    with pytest.raises(ValueError, match="0 results for a batch of 1"):
        run_batched([1, 2], 2, run, is_oom, lambda: None)
