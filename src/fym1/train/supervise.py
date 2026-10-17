"""Training supervision rules from the plan: stop when eval loss rises 2 evals in a row (or is NaN)."""

import math


def eval_losses(log_history: list[dict]) -> list[tuple[int, float]]:
    """(step, eval_loss) pairs from a Trainer `state.log_history`, in logged order."""
    return [(h["step"], h["eval_loss"]) for h in log_history if "eval_loss" in h]


def should_stop(losses: list[float], rises: int = 2) -> bool:
    if losses and math.isnan(losses[-1]):
        return True
    tail = losses[-(rises + 1):]
    return len(tail) == rises + 1 and all(b > a for a, b in zip(tail, tail[1:]))


def best_eval(log_history: list[dict]) -> tuple[int, float] | None:
    """(step, eval_loss) with the lowest finite eval loss, or None if there was no eval."""
    finite = [(s, l) for s, l in eval_losses(log_history) if not math.isnan(l)]
    return min(finite, key=lambda e: e[1]) if finite else None
