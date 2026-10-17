from fym1.train.supervise import best_eval, eval_losses, should_stop

HISTORY = [
    {"step": 10, "loss": 3.0},
    {"step": 100, "eval_loss": 2.0},
    {"step": 200, "eval_loss": 1.5},
    {"step": 300, "eval_loss": 1.6},
    {"step": 400, "eval_loss": 1.7},
]


def test_eval_losses_keeps_eval_entries_in_order():
    assert eval_losses(HISTORY) == [(100, 2.0), (200, 1.5), (300, 1.6), (400, 1.7)]


def test_should_stop_after_two_rises_in_a_row():
    assert should_stop([2.0, 1.5, 1.6, 1.7])
    assert not should_stop([2.0, 1.5, 1.6])
    assert not should_stop([2.0, 1.5, 1.6, 1.55])
    assert not should_stop([])


def test_should_stop_on_nan():
    assert should_stop([2.0, float("nan")])


def test_best_eval_is_lowest_loss_or_none():
    assert best_eval(HISTORY) == (200, 1.5)
    assert best_eval([{"step": 1, "loss": 2.0}]) is None
