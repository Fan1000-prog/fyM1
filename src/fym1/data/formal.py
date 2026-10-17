"""Turn Malagasy Wikipedia paragraphs into single-turn "tell me about" chats."""

import random

MIN_CHARS = 200
MAX_CHARS = 800

# Malagasy wording to be checked by the owner before training.
PROMPT_TEMPLATES = (
    "Lazao ahy ny momba ny {title}.",
    "Inona no fantatrao momba ny {title}?",
    "Hazavao amiko kely ny momba ny {title}.",
    "Mba resaho kely ny {title} azafady.",
)


def pick_paragraph(text: str, rng: random.Random) -> str | None:
    """A random paragraph of `MIN_CHARS`-`MAX_CHARS` characters from an article, if any."""
    paragraphs = [p.strip() for p in text.split("\n")]
    fitting = [p for p in paragraphs if MIN_CHARS <= len(p) <= MAX_CHARS]
    return rng.choice(fitting) if fitting else None


def to_formal_chat(title: str, paragraph: str, rng: random.Random) -> list[dict[str, str]]:
    prompt = rng.choice(PROMPT_TEMPLATES).format(title=title)
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": paragraph}]
