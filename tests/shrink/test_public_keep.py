"""The trim keep set comes from public text only (security finding #13).

A trimmed tokenizer is published with the model, so its vocabulary shows which tokens the keep corpus
used. Built from the private chats, it would reveal which tokens appear in them.
"""

import pytest
from synthetic_text import HELD_OUT, OTHER_LANGUAGES

from fym1.shrink.public_keep import (
    DEFAULT_SOURCES,
    PUBLIC_DATASETS,
    PublicSource,
    public_keep_ids,
    source_lines,
)
from fym1.shrink.vocab import tokens_per_word, trim_tokenizer

REVISION = "0" * 40
CHAT_LINE = HELD_OUT[0]  # synthetic stand-in for a private chat line


def _encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


class FakeHub:
    """Stands in for `datasets.load_dataset`: serves synthetic public rows, records every call."""

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def __call__(self, path, *args, **kwargs):
        self.calls.append((path, args, kwargs))
        return iter(self.rows)


def _source(lines=100, dataset="wikimedia/wikipedia"):
    return PublicSource(dataset, REVISION, lines, config="20231101.mg")


# --- sources -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", ["Fan1000-Prog/fyM1-chats", "Fan1000-Prog/fyM1-ckpt", "someone/chats", ""])
def test_a_non_public_dataset_is_refused(dataset):
    with pytest.raises(ValueError, match="public"):
        PublicSource(dataset, REVISION, 10)


@pytest.mark.parametrize("revision", ["main", "", "b04c8d1c"])
def test_an_unpinned_revision_is_refused(revision):
    with pytest.raises(ValueError, match="revision"):
        PublicSource("wikimedia/wikipedia", revision, 10, config="20231101.mg")


def test_default_sources_are_public_pinned_and_cover_mg_fr_en():
    assert {s.dataset for s in DEFAULT_SOURCES} <= PUBLIC_DATASETS
    assert {s.config for s in DEFAULT_SOURCES} >= {"20231101.mg", "20231101.fr", "20231101.en"}
    assert any(s.dataset.startswith("HPLT/") for s in DEFAULT_SOURCES)  # everyday Malagasy web text


def test_source_lines_splits_filters_and_caps():
    short, long = "x" * 5, "y" * 2000
    hub = FakeHub([{"text": f"{OTHER_LANGUAGES[0]}\n{short}\n{long}\n{OTHER_LANGUAGES[1]}"}, {"text": OTHER_LANGUAGES[2]}])
    assert source_lines(_source(lines=2), load=hub) == OTHER_LANGUAGES[:2]
    path, args, kwargs = hub.calls[0]
    assert (path, args) == ("wikimedia/wikipedia", ("20231101.mg",))
    assert kwargs["revision"] == REVISION and kwargs["streaming"] is True


def test_source_lines_reads_parquet_files_of_a_pinned_repo():
    hub = FakeHub([{"text": OTHER_LANGUAGES[0]}])
    source = PublicSource("HPLT/HPLT2.0_cleaned", REVISION, 5, data_files=("plt_Latn/a.parquet",))
    assert source_lines(source, load=hub) == OTHER_LANGUAGES[:1]
    path, args, kwargs = hub.calls[0]
    assert (path, args, kwargs["data_files"]) == ("HPLT/HPLT2.0_cleaned", (), ["plt_Latn/a.parquet"])


# --- keep set ----------------------------------------------------------------------------------------


def test_keep_set_reads_only_public_datasets(tokenizer):
    hub = FakeHub([{"text": line} for line in OTHER_LANGUAGES])
    public_keep_ids(tokenizer, [_source()], load=hub)
    assert hub.calls and {path for path, _, _ in hub.calls} <= PUBLIC_DATASETS


def test_a_chat_only_token_is_not_kept(tokenizer):
    """A token only the (synthetic) chat line uses stays out; the line still encodes and decodes via bytes."""
    hub = FakeHub([{"text": line} for line in OTHER_LANGUAGES])
    keep, counts = public_keep_ids(tokenizer, [_source()], load=hub)
    chat_ids = set(_encode(tokenizer, CHAT_LINE))
    public_ids = {i for line in OTHER_LANGUAGES for i in _encode(tokenizer, line)}
    assert chat_ids - public_ids, "fixture check: the chat line must use tokens public text does not"
    assert chat_ids - set(keep)
    new = trim_tokenizer(tokenizer, keep)
    assert new.decode(_encode(new, CHAT_LINE)) == tokenizer.decode(_encode(tokenizer, CHAT_LINE))
    assert counts == {"wikimedia/wikipedia:20231101.mg": len(OTHER_LANGUAGES)}


CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}Fuchs springt{% else %}Hund{% endif %}\n{% endif %}"
)


def test_template_text_is_kept(make_tokenizer):
    """The generation-prompt header encodes the same after trimming, though no public line holds it."""
    tok = make_tokenizer("byte_level")
    tok.chat_template = CHAT_TEMPLATE
    hub = FakeHub([{"text": line} for line in OTHER_LANGUAGES[2:]])  # no "Fuchs springt" in the public text
    keep, _ = public_keep_ids(tok, [_source()], system_prompt="Mpanampy", load=hub, enable_thinking=False)
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": "Mpanampy"}, {"role": "user", "content": OTHER_LANGUAGES[3]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    new = trim_tokenizer(tok, keep)
    assert [keep[i] for i in _encode(new, prompt)] == _encode(tok, prompt)


def test_public_keep_ids_takes_no_conversations():
    """The builder has no parameter a caller could hand chats through."""
    import inspect

    params = set(inspect.signature(public_keep_ids).parameters)
    assert params == {"tokenizer", "sources", "system_prompt", "load", "template_kwargs"}


# --- cost --------------------------------------------------------------------------------------------


def test_tokens_per_word_counts(tokenizer):
    lines = ["a b", "c"]
    cost = tokens_per_word(tokenizer, lines)
    tokens = sum(len(_encode(tokenizer, line)) for line in lines)
    assert cost == {"lines": 2, "words": 3, "tokens": tokens, "tokens_per_word": round(tokens / 3, 4)}


def test_tokens_per_word_needs_words(tokenizer):
    with pytest.raises(ValueError, match="no words"):
        tokens_per_word(tokenizer, ["", "  "])


def test_trimming_costs_tokens_on_chat_only_text(tokenizer):
    hub = FakeHub([{"text": line} for line in OTHER_LANGUAGES])
    keep, _ = public_keep_ids(tokenizer, [_source()], load=hub)
    new = trim_tokenizer(tokenizer, keep)
    assert tokens_per_word(new, HELD_OUT)["tokens"] > tokens_per_word(tokenizer, HELD_OUT)["tokens"]
