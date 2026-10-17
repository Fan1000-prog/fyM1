"""Turn arena votes (written by spaces/arena) into a fyM1-vs-base win rate with a 95% Wilson interval."""

import math
from collections.abc import Iterable, Mapping

Z_95 = 1.959963984540054
OTHER_MODEL = {"base": "fym1", "fym1": "base"}


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial rate; well-behaved for small `trials` and rates near 0 or 1."""
    if trials <= 0:
        raise ValueError(f"wilson_interval needs at least one trial, got {trials}")
    rate = successes / trials
    denominator = 1 + z**2 / trials
    centre = (rate + z**2 / (2 * trials)) / denominator
    half_width = z * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def side_models(a_is: str) -> dict[str, str]:
    """Which model sat on side A and side B of a pair."""
    return {"A": a_is, "B": OTHER_MODEL[a_is]}


def first_vote_per_session_and_pair(votes: Iterable[dict]) -> list[dict]:
    """Drop repeat votes; the Space blocks them, but its memory resets when it restarts."""
    kept = {}
    for vote in votes:
        kept.setdefault((vote["session_id"], vote["pair_id"]), vote)
    return list(kept.values())


def tally(votes: Iterable[dict], a_is_by_pair: Mapping[str, str]) -> dict:
    """Win counts, win rate (ties excluded) with 95% CI, and wrong-answer flags per model."""
    unique_votes = first_vote_per_session_and_pair(votes)
    wins = {"base": 0, "fym1": 0}
    wrong_flags = {"base": 0, "fym1": 0}
    same = 0
    for vote in unique_votes:
        if vote["pair_id"] not in a_is_by_pair:
            raise ValueError(f"vote for unknown pair {vote['pair_id']!r}; tally against the pairs file the arena served")
        models = side_models(a_is_by_pair[vote["pair_id"]])
        if vote["choice"] == "same":
            same += 1
        else:
            wins[models[vote["choice"]]] += 1
        for side in vote["wrong"]:
            wrong_flags[models[side]] += 1
    decisive = wins["fym1"] + wins["base"]
    return {
        "n_votes": len(unique_votes),
        "n_sessions": len({vote["session_id"] for vote in unique_votes}),
        "fym1_wins": wins["fym1"],
        "base_wins": wins["base"],
        "same": same,
        "fym1_win_rate_excl_ties": wins["fym1"] / decisive if decisive else None,
        "ci95": list(wilson_interval(wins["fym1"], decisive)) if decisive else None,
        "wrong_flags": wrong_flags,
    }
