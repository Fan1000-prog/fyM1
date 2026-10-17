"""Arena logic with no Gradio and no network: reading pairs, picking one, turning a click into a vote."""

import hashlib
import hmac
import json
import random
import secrets
from collections import Counter
from collections.abc import Collection
from datetime import datetime, timezone
from pathlib import Path

CHOICES = frozenset({"A", "B", "same"})
SIDES = frozenset({"A", "B"})
MODELS = frozenset({"base", "fym1"})
PAIR_KEYS = ("pair_id", "prompt", "a", "b", "a_is")
MAX_SESSION_ID_CHARS = 128


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


def sign(nonce: str, key: bytes) -> str:
    return hmac.new(key, nonce.encode(), hashlib.sha256).hexdigest()


def new_session_id(key: bytes) -> str:
    """A random id signed with the server key, so clients cannot invent their own."""
    nonce = secrets.token_hex(16)
    return f"{nonce}.{sign(nonce, key)}"


def is_valid_session_id(session_id, key: bytes) -> bool:
    """True only for an id this server minted (any type may arrive: the browser sends it)."""
    if not isinstance(session_id, str) or len(session_id) > MAX_SESSION_ID_CHARS:
        return False
    nonce, _, signature = session_id.partition(".")
    return bool(nonce) and hmac.compare_digest(signature, sign(nonce, key))


class VoteBook:
    """Server-side memory of who voted on what; decides whether a click becomes a saved vote.

    Not thread-safe: the caller holds the write lock around `record`.
    """

    def __init__(self, key: bytes, max_votes_per_session: int):
        self.key = key
        self.max_votes_per_session = max_votes_per_session
        self.seen: set[tuple[str, str]] = set()
        self.votes_cast: Counter[str] = Counter()

    def record(
        self, pair_id: str, choice: str | None, session_id, wrong: Collection[str] = (), now: datetime | None = None
    ) -> dict | None:
        """The vote to save and remember, or None for a forged id, a session at its cap, or what make_vote rejects."""
        if not is_valid_session_id(session_id, self.key):
            return None
        if self.votes_cast[session_id] >= self.max_votes_per_session:
            return None
        vote = make_vote(pair_id, choice, session_id, self.seen, wrong=wrong, now=now)
        if vote is not None:
            self.seen.add((session_id, pair_id))
            self.votes_cast[session_id] += 1
        return vote
