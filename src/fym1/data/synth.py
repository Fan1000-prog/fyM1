"""Synthetic everyday chats from an open-weight model, few-shot prompted with real (anonymised) chats.

Pure and CPU-testable; the GPU part lives in notebooks/04_synth.ipynb. The flow per conversation:
seed (topic x personas x register) -> prompt with k real examples -> model output -> parse -> filter.
Output rows use the chat schema of build_chats.py: `{"id", "messages": [{"role", "content"}, ...]}`,
user first, assistant last. Real chats are only ever few-shot examples and the leak reference here:
any conversation sharing an 8-word run with a real chat is dropped. Stats are counts only.
"""

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

from fym1.data.anonymize import FIRST_NAMES, SURNAMES, Anonymizer, compare_key
from fym1.data.leftovers import leftover_names
from fym1.data.segment import Conversation
from fym1.eval.lang import MALAGASY_WORDS
from fym1.eval.metrics import NgramIndex

# Everyday situations; English, so the generator reads them reliably. The chat itself is in Malagasy.
TOPICS = (
    "planning to meet up this weekend", "asking how the family is doing", "a football match last night",
    "prices at the market going up", "a taxi-be ride that took forever", "a power cut (delestage) at home",
    "cooking dinner tonight", "a birthday party", "exams and revision", "looking for a job",
    "a funny video someone shared", "borrowing a little money", "the rainy season and floods",
    "a wedding next month", "church on Sunday", "a new phone", "mobile money that did not go through",
    "a trip to the countryside (ambanivohitra)", "a sick child or relative", "music and a new song",
    "work running late", "fixing a motorbike", "buying school things for the kids", "a famadihana or family event",
    "watching a series", "the internet bundle running out", "going to the beach", "helping someone move house",
    "a misunderstanding between friends", "good news someone just got",
)  # fmt: skip
PERSONAS = (
    "a university student", "a market vendor", "a taxi driver", "an office worker", "a mother of two",
    "a retired uncle", "a young teacher", "a nurse", "a football fan", "a small shop owner",
    "a cousin living in town", "a friend from school",
)  # fmt: skip
REGISTERS = (
    "casual and friendly, short lines",
    "teasing and funny, lots of emoji",
    "warm and caring",
    "a bit rushed, very short texting spelling",
    "heavy French mixing, like young people in Tana",
)
ALLOWED_NAMES = FIRST_NAMES  # the anonymiser's fake pool: real chats show these names too
NAMES_PER_SEED = 2
MIN_TURNS, MAX_TURNS = 2, 12  # build_chats.py cuts conversations at 12 turns
SEED_TURNS = (2, 4, 4, 6, 6, 8, 10)  # requested turns, weighted to short chats like the real ones
MAX_TURN_CHARS = 500
MAX_CHAT_CHARS = 3000
MAX_EXAMPLE_CHARS = 1500  # few-shot examples longer than this make the prompt too long
NEW_CHAT_HEADER = "### New chat"

# Capitalised words that are not names: French and texting words people start lines with.
NOT_NAMES = frozenset(
    compare_key(w)
    for w in """
    merci salut bonjour bonsoir bonne bon oui non mais donc bref alors enfin voila voilà déjà deja désolé desole
    je tu il elle on nous vous ils moi toi lui quoi comment pourquoi quand ou où ça ca cest c'est trop grave
    super cool top vraiment courage bisous bises allez allo allô attends attend normal sérieux serieux pardon
    haha hahaha hihi mdr lol ptdr ouf bah ben hein ok okay oki dac stp svp rdv
    """.split()
) | frozenset(compare_key(w) for w in MALAGASY_WORDS)

# Synthetic chats are training data, so the generator's licence reaches fy-m1. Owner decision 2026-10-18: fy-m1
# ships under a simple open licence, so only permissively licensed generators (Gemma 3's terms of use would make
# fy-m1 a Gemma derivative). Licences as on each model card, lower case; a new generator is added here with its own.
PERMISSIVE_LICENCES = frozenset({"apache-2.0", "mit", "cc-by-4.0"})
GENERATOR_LICENCES = {
    "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1": "cc-by-4.0",
    "google/gemma-4-12B-it": "apache-2.0",
}
_NOT_FETCHED = object()

DROP_REASONS = (
    "unparseable", "length", "not_malagasy", "placeholder", "unknown_name", "leak_8gram", "duplicate",
    "near_duplicate",
)  # fmt: skip
LEAK_N = 8
NEAR_DUP_JACCARD = 0.7
_SHINGLE = 3
_PLACEHOLDER = re.compile(r"<[A-Z_]+>|\[[A-Z_]+\]")
_WORDS = re.compile(r"\w+")
_SPEAKER = re.compile(r"^[\s*_]*([AB])[\s*_]*[:：][\s*_]*(.*)$")
_ROLE = {"A": "user", "B": "assistant"}


@dataclass(frozen=True)
class Seed:
    topic: str
    persona_user: str
    persona_assistant: str
    register: str
    turns: int
    names: tuple[str, ...]

    def as_dict(self) -> dict:
        return {**asdict(self), "names": list(self.names)}

    @classmethod
    def from_dict(cls, d: dict) -> "Seed":
        return cls(**{**d, "names": tuple(d["names"])})


def seed_grid(n: int, seed: int) -> list[Seed]:
    """`n` seeds, every topic once per round. Prefix-stable: seed_grid(m)[:n] == seed_grid(n) for m >= n."""
    rng = random.Random(seed)
    seeds: list[Seed] = []
    while len(seeds) < n:
        for topic in rng.sample(TOPICS, len(TOPICS)):
            user, assistant = rng.sample(PERSONAS, 2)
            names = tuple(rng.sample(ALLOWED_NAMES, NAMES_PER_SEED))
            seeds.append(Seed(topic, user, assistant, rng.choice(REGISTERS), rng.choice(SEED_TURNS), names))
    return seeds[:n]


def generator_licence(hub_id: str, card_licence: str | None | object = _NOT_FETCHED) -> str:
    """The licence of an allowed generator, or ValueError. `card_licence` is the Hub model card's, when fetched:
    it must agree, so a relicensed model is refused until GENERATOR_LICENCES is checked again."""
    licence = GENERATOR_LICENCES.get(hub_id)
    if licence not in PERMISSIVE_LICENCES:
        raise ValueError(f"{hub_id} is not an allowed generator (permissive only): {sorted(GENERATOR_LICENCES)}")
    if card_licence is not _NOT_FETCHED and str(card_licence or "").lower() != licence:
        raise ValueError(f"{hub_id}: model card licence {card_licence!r} is not the checked {licence!r}")
    return licence


def is_sampling_overflow(error: BaseException) -> bool:
    """True for torch's sampling error on inf/nan probabilities (float16 overflow): the next generator should run."""
    return isinstance(error, RuntimeError) and "probability tensor contains" in str(error)


def _chars(messages: Conversation) -> int:
    return sum(len(turn["content"]) for turn in messages)


def examples_for(index: int, real: list[dict], k: int, seed: int) -> list[dict]:
    """The `k` real records shown as few-shot examples for seed `index`; same answer on every rerun."""
    eligible = sorted((r for r in real if _chars(r["messages"]) <= MAX_EXAMPLE_CHARS), key=lambda r: r["id"])
    if len(eligible) < k:
        raise ValueError(f"need {k} real chats of <= {MAX_EXAMPLE_CHARS} chars for few-shot, have {len(eligible)}")
    return random.Random(f"{seed}:{index}").sample(eligible, k)


def _render(messages: Conversation) -> str:
    speaker = {"user": "A", "assistant": "B"}
    return "\n".join(f"{speaker[t['role']]}: {' / '.join(t['content'].splitlines())}" for t in messages)


def build_prompt(seed: Seed, examples: list[Conversation]) -> str:
    """The user message asking the generator for one new chat in the style of `examples`."""
    shown = "\n\n".join(f"### Example {i}\n{_render(m)}" for i, m in enumerate(examples, 1))
    return f"""You write realistic Messenger chats between two people in Madagascar, in everyday Malagasy as people \
really text each other.

Rules:
- Write in Malagasy. French words and short French phrases mixed in are normal ("lancer-na", "rdv", "normal"), \
but every line by B must be mostly Malagasy, never a full French or English sentence.
- Texting style: mostly lower case, short forms and phonetic spelling (zany, ahn, bé, zah, aona), slang, emoji ok. \
Short lines.
- Do not copy any sentence from the examples. New wording, new details.
- No phone numbers, emails, links, addresses or @handles. Never write placeholders like <TEL> or <NAME>.
- If a name is needed, use only: {", ".join(seed.names)}.

The new chat:
- Topic: {seed.topic}.
- A is {seed.persona_user}. B is {seed.persona_assistant} and replies to A.
- Tone: {seed.register}.
- About {seed.turns} lines, alternating A then B, starting with A and ending with B.

Format: one line per message, each starting with "A: " or "B: ". Output only the chat.

Examples of the style (other people, other topics):

{shown}

{NEW_CHAT_HEADER}
"""


def parse_conversation(output: str) -> Conversation | None:
    """Model output -> user/assistant turns, or None when there is no user-then-assistant chat in it.

    Lines start with "A:" (user) or "B:" (assistant), optionally in bold. A line without a speaker
    continues the previous turn; after a blank line it is a trailing note and ignored, as is any preamble.
    """
    turns: Conversation = []
    open_turn = False
    for line in output.splitlines():
        if line.strip().startswith("```"):
            continue
        match = _SPEAKER.match(line)
        if match:
            role, text = _ROLE[match[1]], match[2].strip()
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] = "\n".join(filter(None, (turns[-1]["content"], text)))
            else:
                turns.append({"role": role, "content": text})
            open_turn = True
        elif not line.strip():
            open_turn = False
        elif open_turn:
            turns[-1]["content"] = "\n".join(filter(None, (turns[-1]["content"], line.strip())))
    turns = [t for t in turns if t["content"].strip()]
    turns = _merge(turns)
    start = 1 if turns and turns[0]["role"] == "assistant" else 0
    end = len(turns) - 1 if turns and turns[-1]["role"] == "user" else len(turns)
    turns = turns[start:end]
    return turns if len(turns) >= 2 else None


def _merge(turns: Conversation) -> Conversation:
    merged: Conversation = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1] = {**merged[-1], "content": merged[-1]["content"] + "\n" + turn["content"]}
        else:
            merged.append(dict(turn))
    return merged


def _record_id(messages: Conversation) -> str:
    text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return "synth-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _words(messages: Conversation) -> list[str]:
    text = " | ".join(t["content"] for t in messages)
    return _WORDS.findall(unicodedata.normalize("NFC", text).lower())


def _shingles(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)} or {tuple(words)}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _fits_length(messages: Conversation) -> bool:
    return (MIN_TURNS <= len(messages) <= MAX_TURNS
            and all(0 < len(t["content"].strip()) <= MAX_TURN_CHARS for t in messages)
            and _chars(messages) <= MAX_CHAT_CHARS)  # fmt: skip


def _has_unknown_name(messages: Conversation) -> bool:
    names = leftover_names((t["content"] for t in messages), fake_names=FIRST_NAMES + SURNAMES)
    return any(compare_key(word) not in NOT_NAMES for word in names)


def _first_failure(
    messages: Conversation | None, is_malagasy: Callable[[str], bool], anonymize: Anonymizer, leaks: NgramIndex
) -> tuple[str | None, Conversation | None]:
    """(drop reason or None, the anonymised conversation)."""
    if messages is None:
        return "unparseable", None
    if not _fits_length(messages):
        return "length", None
    if not all(is_malagasy(t["content"]) for t in messages if t["role"] == "assistant"):
        return "not_malagasy", None
    clean = [{"role": t["role"], "content": anonymize(t["content"])} for t in messages]
    if any(_PLACEHOLDER.search(t["content"]) for t in clean):
        return "placeholder", None
    if _has_unknown_name(clean):
        return "unknown_name", None
    if any(leaks.shares_with(t["content"]) for t in clean):
        return "leak_8gram", None
    return None, clean


def filter_conversations(
    conversations: Iterable[Conversation | None], real: list[dict], is_malagasy: Callable[[str], bool]
) -> tuple[list[dict], Counter]:
    """Kept `{"id", "messages"}` rows (input order) and drop counts per reason (first failing filter wins).

    `conversations` are parse_conversation() results (None = unparseable). `real` are the real chat
    records: any synthetic turn sharing an 8-word run with any real turn drops the conversation.
    """
    leaks = NgramIndex((t["content"] for r in real for t in r["messages"]), n=LEAK_N)
    anonymize = Anonymizer({})  # no real names known here: catches phones, emails, links, handles, mentions
    dropped = Counter({reason: 0 for reason in DROP_REASONS})
    kept: list[dict] = []
    seen_exact: set[str] = set()
    kept_shingles: list[set] = []
    for messages in conversations:
        reason, clean = _first_failure(messages, is_malagasy, anonymize, leaks)
        if clean is not None:
            words = _words(clean)
            shingles = _shingles(words)
            if " ".join(words) in seen_exact:
                reason = "duplicate"
            elif any(_jaccard(shingles, other) >= NEAR_DUP_JACCARD for other in kept_shingles):
                reason = "near_duplicate"
            else:
                seen_exact.add(" ".join(words))
                kept_shingles.append(shingles)
                kept.append({"id": _record_id(clean), "messages": clean})
        if reason:
            dropped[reason] += 1
    return kept, dropped


# results/synth_stats.json: counts and run settings only, never text.
STATS_META_KEYS = (
    "seed", "few_shot_k", "target_raw", "generator", "dtype", "gpus", "code_sha", "data_revision",
    "gen_seconds", "new_tokens", "generator_licence",
)  # fmt: skip
_STATS_KEYS = {"raw", "kept", "dropped", "by_generator", *STATS_META_KEYS}
_MODEL_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
_MAX_META_STR = 80


def synth_stats(raw: int, kept: int, dropped: Counter | dict, by_generator: dict, meta: dict) -> dict:
    """The counts-only stats dict; unknown meta keys are refused by validate_stats."""
    return {"raw": raw, "kept": kept, "dropped": {r: int(dropped.get(r, 0)) for r in DROP_REASONS},
            "by_generator": dict(sorted(by_generator.items())), **meta}  # fmt: skip


def filter_raw_rows(
    rows: list[dict], real: list[dict], is_malagasy: Callable[[str], bool], meta: dict
) -> tuple[list[dict], dict]:
    """04_synth's raw rows -> (kept `{"id", "messages"}` rows, validated counts-only stats).

    Each row carries the model `output`, `generator`, `dtype`, `gen_seconds` and `new_tokens`; `meta` gives the
    run settings the rows do not record (seed, few_shot_k, target_raw, gpus, code_sha, data_revision).
    """
    kept, dropped = filter_conversations((parse_conversation(r["output"]) for r in rows), real, is_malagasy)
    generators = sorted({(r["generator"], r["dtype"]) for r in rows})
    stats = synth_stats(
        raw=len(rows), kept=len(kept), dropped=dropped, by_generator=Counter(r["generator"] for r in rows),
        meta={**meta,
              "generator": ", ".join(g for g, _ in generators),
              "dtype": ", ".join(sorted({d for _, d in generators})),
              "generator_licence": ", ".join(sorted({generator_licence(g) for g, _ in generators})),
              "gen_seconds": round(sum(r["gen_seconds"] for r in rows), 1),
              "new_tokens": sum(r["new_tokens"] for r in rows)},
    )  # fmt: skip
    validate_stats(stats)
    return kept, stats


def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_stats(stats: dict) -> None:
    """Raise ValueError unless `stats` holds only known keys with counts or short run settings."""
    unknown = set(stats) - _STATS_KEYS
    if unknown:
        raise ValueError(f"unknown stats keys {sorted(unknown)}")
    if not (_is_count(stats.get("raw")) and _is_count(stats.get("kept"))):
        raise ValueError("raw and kept must be counts")
    dropped = stats.get("dropped", {})
    if set(dropped) != set(DROP_REASONS) or not all(map(_is_count, dropped.values())):
        raise ValueError(f"dropped must count exactly {DROP_REASONS}")
    by_generator = stats.get("by_generator", {})
    if not all(_MODEL_ID.match(k) and _is_count(v) for k, v in by_generator.items()):
        raise ValueError("by_generator must map model ids to counts")
    for hub_id in by_generator:
        generator_licence(hub_id)
    for key in STATS_META_KEYS:
        value = stats.get(key)
        ok = value is None or _is_count(value) or isinstance(value, float) and not isinstance(value, bool)
        ok = ok or isinstance(value, str) and len(value) <= _MAX_META_STR and "\n" not in value
        if not ok:
            raise ValueError(f"{key} must be a number or a short setting string")
    if stats["kept"] + sum(dropped.values()) != stats["raw"]:
        raise ValueError("kept + dropped must add up to raw")
