"""Arena data offline: build the blind pairs from eval answers, and tally votes into a fyM1 win rate."""

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime

Z_95 = 1.959963984540054
OTHER_MODEL = {"base": "fym1", "fym1": "base"}
# Reading a prompt and two answers takes longer than this; a faster median means a script or a button masher.
MIN_MEDIAN_SECONDS_BETWEEN_VOTES = 2.0


def build_pairs(base_rows: Iterable[dict], fym1_rows: Iterable[dict], rng: random.Random) -> tuple[list[dict], list[str]]:
    """Blind pairs (`{"pair_id", "prompt", "a", "b", "a_is"}`, sides shuffled) and the ids missing on one side.

    Rows are eval answers `{"id", "prompt", "response"}`; ids are sorted so a seed always gives the same file.
    """
    base = {row["id"]: row for row in base_rows}
    fym1 = {row["id"]: row for row in fym1_rows}
    pairs = []
    for pair_id in sorted(base.keys() & fym1.keys()):
        if base[pair_id]["prompt"] != fym1[pair_id]["prompt"]:
            raise ValueError(f"id {pair_id!r} has different prompts in the two answer files")
        a_is = rng.choice(["base", "fym1"])
        answers = {"base": base[pair_id]["response"], "fym1": fym1[pair_id]["response"]}
        pairs.append(
            {
                "pair_id": pair_id,
                "prompt": base[pair_id]["prompt"],
                "a": answers[a_is],
                "b": answers[OTHER_MODEL[a_is]],
                "a_is": a_is,
            }
        )
    return pairs, sorted(base.keys() ^ fym1.keys())


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


def median_seconds_between_votes(session_votes: list[dict]) -> float | None:
    """Median gap between one session's consecutive votes; None with fewer than two votes."""
    times = sorted(datetime.fromisoformat(vote["ts"]) for vote in session_votes)
    gaps = [(later - earlier).total_seconds() for earlier, later in zip(times, times[1:])]
    return statistics.median(gaps) if gaps else None


def count_choices(votes: Iterable[dict], a_is_by_pair: Mapping[str, str]) -> dict:
    """Wins per model, ties, and wrong-answer flags per model."""
    wins = {"base": 0, "fym1": 0}
    wrong_flags = {"base": 0, "fym1": 0}
    same = 0
    for vote in votes:
        models = side_models(a_is_by_pair[vote["pair_id"]])
        if vote["choice"] == "same":
            same += 1
        else:
            wins[models[vote["choice"]]] += 1
        for side in vote["wrong"]:
            wrong_flags[models[side]] += 1
    return {"fym1_wins": wins["fym1"], "base_wins": wins["base"], "same": same, "wrong_flags": wrong_flags}


def tally(votes: Iterable[dict], a_is_by_pair: Mapping[str, str]) -> dict:
    """The results/eval/arena.json record (headline keys are the ones paper/numbers.py reads).

    Sessions whose median gap between votes is under MIN_MEDIAN_SECONDS_BETWEEN_VOTES are left out of the
    headline and reported in `flagged_sessions`. `fym1_win_rate` leaves ties ("same") out: fym1 wins /
    (fym1 wins + base wins), with its 95% Wilson interval in `ci_low`/`ci_high`; all three are None until
    there is a decisive vote.
    """
    sessions = defaultdict(list)
    for vote in first_vote_per_session_and_pair(votes):
        if vote["pair_id"] not in a_is_by_pair:
            raise ValueError(f"vote for unknown pair {vote['pair_id']!r}; tally against the pairs file the arena served")
        sessions[vote["session_id"]].append(vote)
    flagged = []
    kept = []
    for session_votes in sessions.values():
        median_gap = median_seconds_between_votes(session_votes)
        if median_gap is not None and median_gap < MIN_MEDIAN_SECONDS_BETWEEN_VOTES:
            flagged.append({"votes": len(session_votes), "median_seconds_between_votes": median_gap})
        else:
            kept.extend(session_votes)

    counts = count_choices(kept, a_is_by_pair)
    decisive = counts["fym1_wins"] + counts["base_wins"]
    ci_low, ci_high = wilson_interval(counts["fym1_wins"], decisive) if decisive else (None, None)
    return {
        "votes": len(kept),
        "raters": len({vote["session_id"] for vote in kept}),
        "fym1_win_rate": counts["fym1_wins"] / decisive if decisive else None,
        "ci_low": ci_low,
        "ci_high": ci_high,
        **counts,
        "excluded_sessions": len(flagged),
        "excluded_votes": sum(session["votes"] for session in flagged),
        "flagged_sessions": flagged,
        # Counts only: session ids stay in the private votes dataset.
        "votes_per_session": sorted((len(session_votes) for session_votes in sessions.values()), reverse=True),
    }
