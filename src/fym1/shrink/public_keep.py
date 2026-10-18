"""The vocabulary trim's keep set, built from public text only (security finding #13).

The trimmed tokenizer ships with the model, and its vocabulary shows which tokens the keep corpus used.
Built from the private chats, it would reveal which tokens appear in them. So the keep corpus is pinned
public text: Malagasy Wikipedia and HPLT Malagasy web text (the everyday and texting spellings), plus French
and English Wikipedia. Every source must name a dataset in PUBLIC_DATASETS at a pinned revision, and the
builder takes no conversations, so the chats cannot reach it. Text the keep set misses still encodes and
decodes unchanged through the byte tokens `tokens_to_keep` always keeps; it just costs more tokens
(measure with `tokens_per_word`).

The chat template is covered by rendering public lines as conversations (with each generation prompt),
using the public system prompt the demo ships with.
"""

import itertools
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from fym1.data.public_sources import (
    HPLT_DATASET,
    HPLT_FILES,
    HPLT_REVISION,
    WIKIPEDIA_DATASET,
    WIKIPEDIA_REVISION,
)
from fym1.shrink.vocab import chat_texts, tokens_to_keep

PUBLIC_DATASETS = frozenset({WIKIPEDIA_DATASET, HPLT_DATASET})
MIN_LINE_CHARS, MAX_LINE_CHARS = 20, 1000
_PINNED = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class PublicSource:
    """`lines` lines of one pinned public dataset: a Wikipedia `config`, or parquet `data_files` of a repo."""

    dataset: str
    revision: str
    lines: int
    config: str | None = None
    data_files: tuple[str, ...] = ()

    def __post_init__(self):
        if self.dataset not in PUBLIC_DATASETS:
            raise ValueError(f"{self.dataset!r} is not a public dataset the keep set may read: {sorted(PUBLIC_DATASETS)}")
        if not _PINNED.match(self.revision):
            raise ValueError("revision must be a pinned 40-character commit sha")

    @property
    def label(self) -> str:
        return f"{self.dataset}:{self.config or 'plt_Latn'}"


DEFAULT_SOURCES = (
    PublicSource(WIKIPEDIA_DATASET, WIKIPEDIA_REVISION, 25_000, config="20231101.mg"),
    PublicSource(HPLT_DATASET, HPLT_REVISION, 25_000, data_files=HPLT_FILES),
    PublicSource(WIKIPEDIA_DATASET, WIKIPEDIA_REVISION, 25_000, config="20231101.fr"),
    PublicSource(WIKIPEDIA_DATASET, WIKIPEDIA_REVISION, 25_000, config="20231101.en"),
)

Loader = Callable[..., Iterable[dict]]


def source_lines(source: PublicSource, load: Loader | None = None) -> list[str]:
    """The first `source.lines` lines of MIN_LINE_CHARS..MAX_LINE_CHARS characters, streamed."""
    if load is None:
        from datasets import load_dataset as load
    args = (source.config,) if source.config else ()
    kwargs = {"data_files": list(source.data_files)} if source.data_files else {}
    rows = load(source.dataset, *args, revision=source.revision, split="train", streaming=True, **kwargs)
    lines = (line for row in rows for line in row["text"].split("\n") if MIN_LINE_CHARS <= len(line) <= MAX_LINE_CHARS)
    return list(itertools.islice(lines, source.lines))


def _as_conversations(lines: list[str]) -> Iterator[list[dict]]:
    """Consecutive public lines as user/assistant turns, so the template wraps real text."""
    for start in range(0, len(lines), 2):
        pair = lines[start:start + 2]
        yield [{"role": role, "content": text} for role, text in zip(("user", "assistant"), pair, strict=False)]


def public_keep_ids(
    tokenizer, sources: Iterable[PublicSource], system_prompt: str = "", load: Loader | None = None, **template_kwargs
) -> tuple[list[int], dict[str, int]]:
    """Sorted keep ids from public text only, and the line count read from each source.

    Pass the chat-template options used for training and serving (e.g. `enable_thinking=False`).
    """
    lines, counts = [], {}
    for source in sources:
        read = source_lines(source, load)
        counts[source.label] = counts.get(source.label, 0) + len(read)
        lines += read
    corpus = itertools.chain(
        lines,
        chat_texts(tokenizer, _as_conversations(lines), system_prompt, **template_kwargs) if tokenizer.chat_template else (),
    )
    return tokens_to_keep(tokenizer, corpus, always_keep=set()), counts
