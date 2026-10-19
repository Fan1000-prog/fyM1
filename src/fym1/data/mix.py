"""The r2 training mix: assistant-style chats first, real chats for style only, Aya Malagasy capped.

Why (r1 diagnosis): `segment.to_conversations(owner=None)` (build_chats --all-senders) puts every member of the
family group thread in the assistant seat in turn, and the synthetic chats copy that two-friends shape. The model
learned to BE a family member: it gives itself a name and says when it will arrive. r2 keeps the casual style and
removes the persona.

Sources (one file each under MIX_DIR in fyM1-chats, rows `{"id", "messages"}` plus an optional `"repeat"`):

- `synth`: the gate-kept synthetic chats. Each chat is cut before its first assistant turn that plays a person
  (PERSONA_REASONS); a chat left with no user -> assistant pair is dropped.
- `identity`: the owner-written identity chats, each with `"repeat": IDENTITY_REPEAT` (01_train repeats train rows,
  never eval rows: `fym1.train.data_mix.split_rows`).
- `real`: single user -> assistant turn pairs cut from the anonymised real chats, kept only when the reply answers
  the user turn and passes every check in REAL_DROP_REASONS. Capped at REAL_MAX_SHARE of the mix.
- `aya`: CohereLabs/aya_dataset (pinned AYA_REVISION), Plateau Malagasy original annotations only, through the same
  checks (AYA_DROP_REASONS). Capped at AYA_MAX_SHARE of the mix.

Synth, real and Aya drop a row (a synth chat: the whole chat) whose user turn is a public eval prompt
(`eval_prompt`, checked first): an eval score must not come from an eval question trained with its answer. The
owner-written identity chats are not checked: teaching the identity answers is their job.

Pure and CPU-testable: the language test, the raw-export name tokens and the eval prompts are passed in. Counts
only, never text.
"""

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable

from fym1.data import name_scan
from fym1.data.synth import FAKE_POOL
from fym1.eval.identity_check import self_claims, unexcused_mentions

MIX_DIR = "mix-r2"
SOURCES = ("synth", "identity", "real", "aya")
IDENTITY_REPEAT = 5  # 16 owner-written chats x 5 = 80 training rows, about 6% of the mix (r1: 16 of 2,992, 0.5%)
REAL_MAX_SHARE = 0.10  # real chats are style only: at most 10% of the mix rows
AYA_MAX_SHARE = 0.20  # Aya is careful written Malagasy: at most 20%, so the casual sources stay 70%+
SEED = 20261019

AYA_REPO = "CohereLabs/aya_dataset"  # the human-annotated Aya Dataset (was CohereForAI/aya_dataset), not aya_collection
AYA_REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"
AYA_FILE = "data/train-00000-of-00001.parquet"
AYA_LANGUAGE_CODE = "plt"  # Plateau Malagasy; the card's table says "mlg", the rows say "plt"
AYA_ANNOTATION = "original-annotations"  # written from scratch by people; re-annotations edit machine output
AYA_MAX_CHARS = 1500  # prompt or answer longer than this is dropped (the casual turns are far shorter)

EVAL_RUN_WORDS = 8  # a user turn sharing this many words in a row with an eval prompt is that prompt

PERSONA_REASONS = ("name_claim", "personal_logistics", "fake_name")
REAL_DROP_REASONS = ("eval_prompt", "not_a_reply", "drift", "identity", *PERSONA_REASONS, "real_name")
AYA_DROP_REASONS = ("eval_prompt", "duplicate", "length", "style", "drift", "identity", *PERSONA_REASONS, "real_name")

_PLAIN_WORD = re.compile(r"\w+")  # the leak check's words (fym1.eval.metrics): no punctuation, digits kept
_CLAUSE = re.compile(r"[.!?;:,\n…]+")
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
_CAPITALISED = re.compile(r"\b[A-Z][a-z]+\b")
_FIRST_PERSON = frozenset({"aho", "izaho", "zaho", "zah", "za", "izahay", "zahay", "ahy", "je", "moi", "nous"})
# English "I" only with a verb after it: a bare "I" is often a Roman numeral ("Radama I").
_ENGLISH_I = re.compile(r"\bI(?:['’](?:m|ll|ve|d)|\s+(?:am|will|was|have|come|go|get|arrive|leave|can|need)\b)|\bIm\b")
_FRENCH_J = re.compile(r"\bj['’]", re.IGNORECASE)
# Words that place the speaker somewhere or at a time: arriving, coming, going home, being here or there, tonight.
_LOGISTICS = frozenset(
    """
    tonga avy hoavy mandeha handeha nandeha lasa mody hody nody miverina hiverina niverina mankany hankany nankany
    mankaty hankaty mankao miakatra hiakatra niakatra midina hidina nidina miandry hiandry niandry mivoaka hivoaka
    eto ety ao eo any aty ery an-trano rahariva anio androany rahampitso omaly maraina hariva alina atoandro ora
    minitra arrive arriver arrivé arrivée viens venir rentre rentrer passe passer vais aller chez maison soir demain
    bientôt route coming arriving way home tonight tomorrow there here leave leaving
    """.split()
)
# A clock time also places the speaker ("ho tonga amin'ny 2 ora aho", "je passe à 8h", "I'll call at 7pm", "14h30").
_CLOCK_TIME = re.compile(
    r"\bamin['’]ny\s+\d{1,2}\s*(?:ora|h)\b|\bà\s+\d{1,2}\s*h|\bat\s+\d{1,2}(?::\d{2})?\b|\b\d{1,2}\s*[ap]m\b"
    r"|\b\d{1,2}h\d{0,2}\b",
    re.IGNORECASE,
)
_HELPING = frozenset({"hanampy", "manampy", "ampiana", "hanampiana", "help", "aider", "aide", "assist"})
_NAME_PHRASE = re.compile(
    r"\banarako\b|\biantsoana\s+ahy\b|\bmy\s+name\b|\bcall\s+me\b|\bm['’]appelle\b|\bmoi\s+c['’]est\b",
    re.IGNORECASE,
)
_SELF_THEN_NAME = re.compile(r"\b(?:[Ii]zaho|[Zz]aho|[Zz]ah)\s+(?:(?:dia|no)\s+)?(?:i\s+)?([A-Z][a-z]+)\b")
_FAKE_NAMES = frozenset(FAKE_POOL)

_QUESTION_WORDS = frozenset(
    """
    ve inona iza aiza taiza oviana ahoana aona nahoana firy ohatrinona moa sa pourquoi comment quand où quoi qui
    combien what why how when where who
    """.split()
)
_REPLY_OPENERS = frozenset(
    """
    eny ie ia eka tsia tsy oui ouais non ok okey okay oki dac misaotra merci mety marina yes no yeah haha hihi
    aaa ah ahh efa mbola tena eh
    """.split()
)
_STOPWORDS = frozenset(
    """
    ny ilay ve fa dia ka ary sy amin ireo izany izao zao aho ahy ianao anao izy azy isika tsy efa mbola koa
    ihany kely tena le la les de des du un une et est pas the and you that this with have
    """.split()
)


def _words(text: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in _WORD.findall(text)]


def _has_first_person(clause: str) -> bool:
    return bool(_FIRST_PERSON.intersection(_words(clause)) or _ENGLISH_I.search(clause) or _FRENCH_J.search(clause))


def name_claim(text: str) -> bool:
    """The text gives the speaker a personal name: "anarako ...", "my name is", "je m'appelle", "Izaho Koto"."""
    return bool(_NAME_PHRASE.search(text) or _SELF_THEN_NAME.search(text))


def personal_logistics(text: str) -> bool:
    """A clause with the speaker ("aho", "zaho", "je", "I") and a place, a movement, a time word or a clock time
    ("ho tonga aho rahariva", "j'arrive", "I'm coming home", "je passe à 8h"), unless the clause is about helping
    ("Eto aho hanampy")."""
    for clause in _CLAUSE.split(text):
        words = {part for word in _words(clause) for part in (word, *word.split("'"))}  # "j'arrive" -> "arrive"
        places = words & _LOGISTICS or _CLOCK_TIME.search(clause)
        if _has_first_person(clause) and places and not words & _HELPING:
            return True
    return False


def fake_name(text: str) -> bool:
    """A capitalised fake-pool name (the anonymiser's names). Fail closed: also a capitalised common word that
    is a pool name ("Vola", "Soa")."""
    return any(word in _FAKE_NAMES for word in _CAPITALISED.findall(text))


def persona_reasons(text: str) -> list[str]:
    """Every way the text plays a person instead of an assistant, in PERSONA_REASONS order."""
    found = {"name_claim": name_claim(text), "personal_logistics": personal_logistics(text), "fake_name": fake_name(text)}
    return [reason for reason in PERSONA_REASONS if found[reason]]


def _plain_words(text: str) -> tuple[str, ...]:
    return tuple(_PLAIN_WORD.findall(unicodedata.normalize("NFC", text).lower()))


def _runs(words: tuple[str, ...]) -> set[tuple[str, ...]]:
    return {words[i : i + EVAL_RUN_WORDS] for i in range(len(words) - EVAL_RUN_WORDS + 1)}


class EvalPrompts:
    """The public eval prompts (eval/prompts.jsonl, eval/identity_check.jsonl), to keep them out of training.

    A text matches when its words (lower-cased, NFC, punctuation and spacing ignored) equal an eval prompt's, or
    share an EVAL_RUN_WORDS-word run with one."""

    def __init__(self, prompts: Iterable[str]):
        words = [_plain_words(prompt) for prompt in prompts]
        self.exact = frozenset(w for w in words if w)
        if not self.exact:  # an empty set would pass every row unchecked
            raise ValueError("no eval prompts with words: nothing to keep out of training")
        self.runs = frozenset(run for w in words for run in _runs(w))

    def matches(self, text: str) -> bool:
        words = _plain_words(text)
        return words in self.exact or not self.runs.isdisjoint(_runs(words))


def is_question(text: str) -> bool:
    return "?" in text or bool(_QUESTION_WORDS.intersection(_words(text)))


def responds(user: str, reply: str) -> bool:
    """The reply answers the user turn: the user asked something, or the reply opens like an answer ("Eny",
    "Ok", "Misaotra"), or both share a content word (4+ letters, not a common function word)."""
    reply_words = _words(reply)
    if not reply_words:
        return False
    if is_question(user) or reply_words[0] in _REPLY_OPENERS:
        return True
    content = {w for w in _words(user) if len(w) >= 4 and w not in _STOPWORDS}
    return bool(content.intersection(reply_words))


def claims_another_assistant(prompt: str, reply: str) -> bool:
    """The reply claims to be, or names unexcused, another assistant (the identity gate of the synth gate)."""
    return bool(self_claims(reply) or unexcused_mentions(prompt, reply))


def _has_real_name(texts: Iterable[str], raw_names: frozenset[str]) -> bool:
    return any(name_scan.count_name_hits(text, raw_names) for text in texts)


def record_id(messages: list[dict]) -> str:
    """A hash of the messages, like `synth.record_id`, with the mix's own prefix."""
    text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return "mix-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _pair_record(user: str, reply: str) -> dict:
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
    return {"id": record_id(messages), "messages": messages}


def _reply_reasons(user: str, reply: str, is_malagasy: Callable[[str], bool], raw_names: frozenset[str]) -> dict:
    """Checks shared by real pairs and Aya rows."""
    return {
        "drift": not is_malagasy(reply),
        "identity": claims_another_assistant(user, reply),
        **{reason: True for reason in persona_reasons(reply)},
        "real_name": _has_real_name((user, reply), raw_names),
    }


def real_pairs(records: Iterable[dict], is_malagasy: Callable[[str], bool], raw_names: frozenset[str],
               eval_prompts: EvalPrompts) -> tuple[list[dict], Counter]:  # fmt: skip
    """Kept (user, assistant) pairs of the real chats, and the dropped pairs per first reason (REAL_DROP_REASONS
    order; a pair counts once). Exact duplicate pairs are kept once and counted as `duplicate`."""
    kept, drops, seen = [], Counter(dict.fromkeys((*REAL_DROP_REASONS, "duplicate"), 0)), set()
    for record in records:
        turns = record["messages"]
        for i in range(1, len(turns)):
            if turns[i]["role"] != "assistant" or turns[i - 1]["role"] != "user":
                continue
            user, reply = turns[i - 1]["content"], turns[i]["content"]
            found = {"eval_prompt": eval_prompts.matches(user), "not_a_reply": not responds(user, reply),
                     **_reply_reasons(user, reply, is_malagasy, raw_names)}  # fmt: skip
            reason = next((r for r in REAL_DROP_REASONS if found.get(r)), None)
            row = _pair_record(user, reply)
            if reason is None and row["id"] in seen:
                reason = "duplicate"
            if reason:
                drops[reason] += 1
                continue
            seen.add(row["id"])
            kept.append(row)
    return kept, drops


def cut_persona(record: dict) -> tuple[dict | None, str | None]:
    """The chat cut before its first assistant turn that plays a person, and that turn's first reason. None when
    no user -> assistant pair is left; the reason is None when nothing was cut."""
    turns = record["messages"]
    for i, turn in enumerate(turns):
        reasons = persona_reasons(turn["content"]) if turn["role"] == "assistant" else []
        if reasons:
            kept = turns[: i - 1]  # also drop the user turn this reply answered: chats end on an assistant turn
            if len(kept) < 2:
                return None, reasons[0]
            return {"id": record_id(kept), "messages": kept}, reasons[0]
    return record, None


def synth_rows(records: Iterable[dict], eval_prompts: EvalPrompts) -> tuple[list[dict], dict[str, int]]:
    """Synth chats with the persona turns cut off. Counts: chats dropped for an eval prompt in a user turn, then
    chats kept whole, cut, dropped, and per first persona reason."""
    kept = []
    counts = Counter({"eval_prompt": 0, "whole": 0, "cut": 0, "dropped": 0,
                      **{f"reason_{r}": 0 for r in PERSONA_REASONS}})  # fmt: skip
    for record in records:
        if any(t["role"] == "user" and eval_prompts.matches(t["content"]) for t in record["messages"]):
            counts["eval_prompt"] += 1
            continue
        row, reason = cut_persona(record)
        if reason:
            counts[f"reason_{reason}"] += 1
        counts["whole" if reason is None else "cut" if row else "dropped"] += 1
        if row:
            kept.append(row)
    return kept, dict(counts)


def aya_malagasy(rows: Iterable[dict]) -> tuple[list[dict], dict[str, int]]:
    """The Plateau Malagasy rows, then only the original annotations. Counts of both."""
    malagasy = [r for r in rows if r.get("language_code") == AYA_LANGUAGE_CODE]
    original = [r for r in malagasy if r.get("annotation_type") == AYA_ANNOTATION]
    return original, {"malagasy_rows": len(malagasy), "original_annotations": len(original),
                      "re_annotations_left_out": len(malagasy) - len(original)}  # fmt: skip


def aya_rows(rows: Iterable[dict], is_malagasy: Callable[[str], bool], raw_names: frozenset[str],
             eval_prompts: EvalPrompts) -> tuple[list[dict], Counter]:  # fmt: skip
    """Aya rows (`inputs`, `targets`) as single-pair chats through AYA_DROP_REASONS; dropped rows per first reason.
    `style` drops a prompt with no Malagasy (full French/English); `drift` the same for the answer."""
    kept, drops, seen = [], Counter(dict.fromkeys(AYA_DROP_REASONS, 0)), set()
    for row in rows:
        user, reply = row["inputs"].strip(), row["targets"].strip()
        record = _pair_record(user, reply)
        found = {
            "eval_prompt": eval_prompts.matches(user),
            "duplicate": record["id"] in seen,
            "length": not user or not reply or max(len(user), len(reply)) > AYA_MAX_CHARS,
            "style": bool(user) and not is_malagasy(user),
            **_reply_reasons(user, reply, is_malagasy, raw_names),
        }
        seen.add(record["id"])
        reason = next((r for r in AYA_DROP_REASONS if found.get(r)), None)
        if reason:
            drops[reason] += 1
        else:
            kept.append(record)
    return kept, drops


def capped_counts(base_rows: int, available: dict[str, int], shares: dict[str, float]) -> dict[str, int]:
    """How many rows each capped source may add so that it is at most its share of the whole mix (base rows plus
    every capped source). A source with fewer rows than its cap gives all of them and the others are recomputed."""
    if sum(shares.values()) >= 1:
        raise ValueError("capped shares must add up to less than 1")
    taken: dict[str, int] = {}
    while True:
        open_sources = [s for s in shares if s not in taken]
        open_share = sum(shares[s] for s in open_sources)
        total = (base_rows + sum(taken.values())) / (1 - open_share)
        caps = {s: int(shares[s] * total) for s in open_sources}
        short = [s for s in open_sources if available[s] <= caps[s]]
        if not short:
            return {**taken, **caps}
        for source in short:
            taken[source] = available[source]


def sample(rows: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """A fixed-seed sample of `n` rows, in their input order."""
    if n >= len(rows):
        return list(rows)
    picked = set(random.Random(seed).sample(range(len(rows)), n))
    return [row for i, row in enumerate(rows) if i in picked]


def with_repeat(rows: Iterable[dict], repeat: int) -> list[dict]:
    return [{**row, "repeat": repeat} for row in rows]


def mix_rows(rows_by_source: dict[str, list[dict]]) -> int:
    """Training rows of the mix: each row counts `repeat` times."""
    return sum(row.get("repeat", 1) for rows in rows_by_source.values() for row in rows)
