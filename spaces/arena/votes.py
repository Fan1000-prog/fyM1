"""Arena logic with no Gradio and no network: reading pairs, picking one, turning a click into a vote."""

import json
import random
from collections.abc import Collection
from datetime import datetime, timezone
from pathlib import Path

CHOICES = frozenset({"A", "B", "same"})
SIDES = frozenset({"A", "B"})
MODELS = frozenset({"base", "fym1"})
PAIR_KEYS = ("pair_id", "prompt", "a", "b", "a_is")


def load_pairs(path: Path) -> list[dict]:
    """Every pair in `path` (JSON lines); fails on the first malformed line so a bad file never goes live."""
    pairs = []
    seen_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{line_number}"
        try:
            pair = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{where}: not valid JSON ({error})") from error
        missing = [key for key in PAIR_KEYS if key not in pair]
        if missing:
            raise ValueError(f"{where}: missing keys {missing}")
        if pair["a_is"] not in MODELS:
            raise ValueError(f"{where}: a_is must be one of {sorted(MODELS)}, got {pair['a_is']!r}")
        if pair["pair_id"] in seen_ids:
            raise ValueError(f"{where}: duplicate pair_id {pair['pair_id']!r}")
        seen_ids.add(pair["pair_id"])
        pairs.append(pair)
    if not pairs:
        raise ValueError(f"{path.name} has no pairs")
    return pairs


def pick_pair(
    pairs: list[dict], session_id: str, seen: set[tuple[str, str]], rng: random.Random
) -> dict | None:
    """A random pair this session has not voted on, or None when it has rated them all."""
    unseen = [pair for pair in pairs if (session_id, pair["pair_id"]) not in seen]
    return rng.choice(unseen) if unseen else None


def make_vote(
    pair_id: str,
    choice: str | None,
    session_id: str,
    seen: set[tuple[str, str]],
    wrong: Collection[str] = (),
    now: datetime | None = None,
) -> dict | None:
    """The vote to save, or None when nothing should be saved (no valid choice, bad flag, repeat vote).

    `wrong` lists the sides ("A"/"B") the rater flagged as factually wrong.
    """
    if choice not in CHOICES or not SIDES.issuperset(wrong) or (session_id, pair_id) in seen:
        return None
    return {
        "pair_id": pair_id,
        "choice": choice,
        "wrong": sorted(wrong),
        "session_id": session_id,
        "ts": (now or datetime.now(timezone.utc)).isoformat(),
    }
