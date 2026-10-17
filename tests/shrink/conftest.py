"""Tiny synthetic tokenizers for the vocabulary-trimming tests.

Each fixture is trained here, in seconds, on made-up text, so no model download or private chat
text is needed. They cover the two tokenizer kinds the base-model shortlist uses:
byte-level BPE (Qwen, Llama 3) and SentencePiece BPE with byte fallback (Gemma, Llama 2, Mistral).
"""

import io
import json

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")

import sentencepiece as spm  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers  # noqa: E402
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402

from synthetic_text import TRAIN_TEXT  # noqa: E402


def _byte_level_tokenizer(ignore_merges: bool) -> PreTrainedTokenizerFast:
    """Byte-level BPE like Qwen / Llama 3, with a template that prepends a begin-of-text token."""
    tok = Tokenizer(models.BPE(ignore_merges=ignore_merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|begin_of_text|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=600,
        special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(TRAIN_TEXT, trainer=trainer)
    bos_id = tok.token_to_id("<|begin_of_text|>")
    tok.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A",
        pair="<|begin_of_text|> $A <|begin_of_text|> $B",
        special_tokens=[("<|begin_of_text|>", bos_id)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|begin_of_text|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )


def _sentencepiece_tokenizer(directory):
    """SentencePiece BPE with byte fallback, loaded the way Gemma's tokenizer is."""
    buf = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(TRAIN_TEXT),
        model_writer=buf,
        model_type="bpe",
        vocab_size=600,
        byte_fallback=True,
        add_dummy_prefix=False,  # as in Gemma's tokenizer.model
        character_coverage=1.0,
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        eos_piece="<eos>",
        bos_piece="<bos>",
        unk_piece="<unk>",
        user_defined_symbols=["<start_of_turn>", "<end_of_turn>"],
        minloglevel=2,
    )
    (directory / "tokenizer.model").write_bytes(buf.getvalue())
    (directory / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "GemmaTokenizer"}))
    return AutoTokenizer.from_pretrained(directory)


@pytest.fixture(scope="session")
def spm_dir(tmp_path_factory):
    """Directory holding the SentencePiece fixture (tokenizer.model + tokenizer_config.json)."""
    directory = tmp_path_factory.mktemp("spm")
    _sentencepiece_tokenizer(directory)
    return directory


@pytest.fixture(scope="session")
def make_tokenizer(spm_dir):
    """Build a fixture tokenizer by kind: byte_level, byte_level_ignore_merges or sentencepiece."""

    def make(kind: str):
        if kind == "sentencepiece":
            return AutoTokenizer.from_pretrained(spm_dir)
        return _byte_level_tokenizer(ignore_merges=kind.endswith("ignore_merges"))

    return make


@pytest.fixture(scope="session", params=["byte_level", "byte_level_ignore_merges", "sentencepiece"])
def tokenizer(request, make_tokenizer):
    return make_tokenizer(request.param)
