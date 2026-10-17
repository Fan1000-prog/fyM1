"""Turn Malagasy Wikipedia paragraphs into single-turn "tell me about" chats."""

import random
import re
from collections import Counter
from typing import NamedTuple

MIN_CHARS = 200
MAX_CHARS = 800
# Shorter articles are mostly bot-made stubs.
MIN_ARTICLE_CHARS = 1500
# A paragraph opening shared by more articles than this is a bot template.
MAX_TEMPLATE_REPEATS = 20
OPENING_WORDS = 6

# Malagasy wording to be checked by the owner before training.
PROMPT_TEMPLATES = (
    "Lazao ahy ny momba ny {title}.",
    "Inona no fantatrao momba ny {title}?",
    "Hazavao amiko kely ny momba ny {title}.",
    "Mba resaho kely ny {title} azafady.",
)


class Candidate(NamedTuple):
    id: str
    title: str
    paragraph: str


def is_stub(text: str) -> bool:
    return len(text) < MIN_ARTICLE_CHARS


def _opening(candidate: Candidate) -> str:
    """First words of the paragraph, with the title and numbers masked so templates line up."""
    text = re.sub(re.escape(candidate.title), "<title>", candidate.paragraph, flags=re.IGNORECASE)
    text = re.sub(r"\d+", "0", text)
    return " ".join(text.lower().split()[:OPENING_WORDS])


def drop_templated(candidates: list[Candidate]) -> list[Candidate]:
    """`candidates` without those whose opening repeats in over `MAX_TEMPLATE_REPEATS` of them."""
    counts = Counter(_opening(c) for c in candidates)
    return [c for c in candidates if counts[_opening(c)] <= MAX_TEMPLATE_REPEATS]


def pick_paragraph(text: str, rng: random.Random) -> str | None:
    """A random paragraph of `MIN_CHARS`-`MAX_CHARS` characters from an article, if any."""
    paragraphs = [p.strip() for p in text.split("\n")]
    fitting = [p for p in paragraphs if MIN_CHARS <= len(p) <= MAX_CHARS]
    return rng.choice(fitting) if fitting else None


def to_formal_chat(title: str, paragraph: str, rng: random.Random) -> list[dict[str, str]]:
    prompt = rng.choice(PROMPT_TEMPLATES).format(title=title)
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": paragraph}]
