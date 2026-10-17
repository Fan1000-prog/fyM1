import json

import pytest
import sentencepiece as spm
from synthetic_text import HELD_OUT, KEEP_CORPUS

from fym1.shrink.vocab import chat_texts, tokens_to_keep, trim_sentencepiece, trim_tokenizer


def _encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


# --- tokens_to_keep -------------------------------------------------------------------------------


def test_special_tokens_always_kept(tokenizer):
    keep = set(tokens_to_keep(tokenizer, ["a"], always_keep=set()))
    assert set(tokenizer.all_special_ids) <= keep
    assert set(tokenizer.added_tokens_decoder) <= keep


def test_result_sorted_and_unique(tokenizer):
    keep = tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep={5, 5, 7})
    assert keep == sorted(set(keep))


def test_every_corpus_id_present(tokenizer):
    keep = set(tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep=set()))
    for line in KEEP_CORPUS:
        assert set(_encode(tokenizer, line)) <= keep


def test_always_keep_included(tokenizer):
    last = len(tokenizer) - 1
    assert last in tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep={last})


def test_byte_fallback_kept(tokenizer):
    """Any text must stay encodable, so the 256 byte tokens survive the trim."""
    keep = set(tokens_to_keep(tokenizer, ["a"], always_keep=set()))
    vocab = tokenizer.get_vocab()
    if "<0x00>" in vocab:
        # Gemma's converter stores some ASCII byte pieces as the character itself ("\t" for <0x09>).
        byte_tokens = {vocab.get(f"<0x{b:02X}>", vocab.get(chr(b))) for b in range(256)}
    else:
        from tokenizers.pre_tokenizers import ByteLevel

        byte_tokens = {vocab[c] for c in ByteLevel.alphabet()}
    assert None not in byte_tokens
    assert byte_tokens <= keep


def test_byte_piece_stored_as_character_kept(spm_dir):
    """A tab stays encodable when the vocabulary spells byte piece <0x09> as a literal tab."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tok_json = json.loads(AutoTokenizer.from_pretrained(spm_dir).backend_tokenizer.to_str())
    vocab = tok_json["model"]["vocab"]
    if "<0x09>" in vocab:
        vocab["\t"] = vocab.pop("<0x09>")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(tok_json)), unk_token="<unk>")
    keep = tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep=set())
    assert tokenizer.convert_tokens_to_ids("\t") in keep
    new = trim_tokenizer(tokenizer, keep)
    assert new.decode(_encode(new, "Mety\tbe")) == tokenizer.decode(_encode(tokenizer, "Mety\tbe"))


def test_trims_something(tokenizer):
    assert len(tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep=set())) < len(tokenizer)


def test_rejects_unknown_ids(tokenizer):
    with pytest.raises(ValueError):
        tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep={len(tokenizer) + 10})


# --- trim_tokenizer -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trimmed(tokenizer):
    keep = tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep=set())
    return keep, trim_tokenizer(tokenizer, keep)


def test_trimmed_size(trimmed):
    keep, new = trimmed
    assert len(new) == len(keep)


def test_corpus_encodes_to_same_tokens(tokenizer, trimmed):
    """Kept text gets exactly the same token sequence, just renumbered."""
    keep, new = trimmed
    for line in KEEP_CORPUS:
        assert [keep[i] for i in _encode(new, line)] == _encode(tokenizer, line)


def test_held_out_round_trip(tokenizer, trimmed):
    """Unseen text may split into more pieces but must decode to the same string."""
    _, new = trimmed
    for line in HELD_OUT:
        expected = tokenizer.decode(_encode(tokenizer, line))
        assert new.decode(_encode(new, line)) == expected
        assert expected.strip() == line.strip()


def test_special_tokens_renumbered(tokenizer, trimmed):
    keep, new = trimmed
    for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
        token = getattr(tokenizer, name, None)
        if token is not None:
            assert getattr(new, name) == token
            assert keep[new.convert_tokens_to_ids(token)] == tokenizer.convert_tokens_to_ids(token)
    for token in tokenizer.added_tokens_decoder.values():
        assert keep[new.convert_tokens_to_ids(token.content)] == tokenizer.convert_tokens_to_ids(token.content)


def test_template_specials_renumbered(tokenizer, trimmed):
    """Post-processor templates (e.g. Llama 3's begin-of-text) point at the new ids."""
    keep, new = trimmed
    line = KEEP_CORPUS[0]
    assert [keep[i] for i in new.encode(line)] == tokenizer.encode(line)


def test_rejects_keep_missing_merge_parent(tokenizer):
    """Dropping a token that a kept token is built from would make the kept token unreachable."""
    keep = tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep=set())
    model = json.loads(tokenizer.backend_tokenizer.to_str())["model"]
    vocab = model["vocab"]
    for merge in model["merges"]:
        left, right = merge if isinstance(merge, list) else merge.split(" ", 1)
        if vocab[left + right] in keep and len(left) > 1:
            with pytest.raises(ValueError, match="merge parents"):
                trim_tokenizer(tokenizer, [i for i in keep if i != vocab[left]])
            return
    pytest.fail("fixture has no kept token built from a multi-character part")


def test_rejects_unsorted_keep(tokenizer):
    keep = tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep=set())
    with pytest.raises(ValueError):
        trim_tokenizer(tokenizer, list(reversed(keep)))


def test_rejects_non_bpe():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel({"a": 0, "[UNK]": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    with pytest.raises(NotImplementedError):
        tokens_to_keep(wrapped, ["a"], always_keep=set())


# --- chat_texts -----------------------------------------------------------------------------------

# The generation prompt carries text (words from OTHER_LANGUAGES) that no message contains, like
# Gemma 4's "<|turn>model\n<|channel>thought\n<channel|>" header; enable_thinking changes it too.
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}Fuchs springt{% else %}Hund{% endif %}\n{% endif %}"
)
CONVERSATIONS = [
    [{"role": "user", "content": KEEP_CORPUS[0]}, {"role": "assistant", "content": KEEP_CORPUS[1]}],
    [{"role": "user", "content": KEEP_CORPUS[2]}, {"role": "assistant", "content": KEEP_CORPUS[3]}],
]


@pytest.fixture
def chat_tokenizer(make_tokenizer):
    tok = make_tokenizer("byte_level")
    tok.chat_template = CHAT_TEMPLATE
    return tok


def _prompt(tok, enable_thinking=False):
    messages = [{"role": "system", "content": "Mpanampy"}, {"role": "user", "content": KEEP_CORPUS[0]}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)


def test_chat_texts_cover_generation_prompt(chat_tokenizer):
    """A prompt as sent at inference encodes to the same tokens after trimming."""
    texts = list(chat_texts(chat_tokenizer, CONVERSATIONS, "Mpanampy", enable_thinking=False))
    keep = tokens_to_keep(chat_tokenizer, texts, always_keep=set())
    new = trim_tokenizer(chat_tokenizer, keep)
    assert [keep[i] for i in _encode(new, _prompt(chat_tokenizer))] == _encode(chat_tokenizer, _prompt(chat_tokenizer))


def test_whole_conversations_alone_miss_generation_prompt(chat_tokenizer):
    """Why chat_texts adds generation-prompt prefixes: without them the header re-tokenizes."""
    rendered = [chat_tokenizer.apply_chat_template(c, tokenize=False) for c in CONVERSATIONS]
    keep = tokens_to_keep(chat_tokenizer, rendered, always_keep=set())
    new = trim_tokenizer(chat_tokenizer, keep)
    assert [keep[i] for i in _encode(new, _prompt(chat_tokenizer))] != _encode(chat_tokenizer, _prompt(chat_tokenizer))


def test_chat_texts_pass_template_kwargs(chat_tokenizer):
    texts = list(chat_texts(chat_tokenizer, CONVERSATIONS[:1], enable_thinking=False))
    assert any("Fuchs springt" in t for t in texts)
    assert not any("Hund" in t for t in texts)
    assert texts[0].startswith("<|im_start|>user")  # no empty system turn when no system prompt


# --- trim_sentencepiece ---------------------------------------------------------------------------


def test_sentencepiece_model_matches_trimmed_tokenizer(spm_dir):
    """The trimmed tokenizer.model (read by llama.cpp's GGUF converter) agrees with tokenizer.json."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spm_dir)
    keep = tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep=set())
    original = (spm_dir / "tokenizer.model").read_bytes()
    old = spm.SentencePieceProcessor(model_proto=original)
    new = spm.SentencePieceProcessor(model_proto=trim_sentencepiece(original, keep))

    assert new.get_piece_size() == len([i for i in keep if i < old.get_piece_size()])
    for line in KEEP_CORPUS:
        assert [keep[i] for i in new.encode(line)] == old.encode(line)
    for line in HELD_OUT:
        assert new.decode(new.encode(line)) == old.decode(old.encode(line))
    for name in ("unk_id", "bos_id", "eos_id", "pad_id"):
        assert new.id_to_piece(getattr(new, name)()) == old.id_to_piece(getattr(old, name)())
