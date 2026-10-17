import copy
import json

import pytest

torch = pytest.importorskip("torch")

import sentencepiece as spm  # noqa: E402
from synthetic_text import HELD_OUT, KEEP_CORPUS  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
)

from fym1.shrink.model import trim_checkpoint, trim_model  # noqa: E402
from fym1.shrink.vocab import tokens_to_keep  # noqa: E402

# (architecture, tokenizer kind). Gemma 3 covers tied + scaled embeddings with a SentencePiece vocab.
CASES = [
    ("llama_untied", "byte_level"),
    ("llama_tied", "byte_level_ignore_merges"),
    ("gemma3", "sentencepiece"),
]
PADDED_ROWS = 8  # real checkpoints carry embedding rows past the last tokenizer id


def _tiny_model(arch, tokenizer):
    torch.manual_seed(0)
    common = dict(
        vocab_size=len(tokenizer) + PADDED_ROWS,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if arch == "gemma3":
        model = Gemma3ForCausalLM(Gemma3TextConfig(**common))
    else:
        model = LlamaForCausalLM(LlamaConfig(**common, tie_word_embeddings=arch == "llama_tied"))
    model.generation_config.eos_token_id = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    return model.eval()


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def case(request, make_tokenizer):
    arch, kind = request.param
    tokenizer = make_tokenizer(kind)
    original = _tiny_model(arch, tokenizer)
    keep = tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep=set())
    model, new_tokenizer = trim_model(copy.deepcopy(original), tokenizer, keep)
    return arch, tokenizer, original, keep, model, new_tokenizer


def _logits(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0]


def _assert_same_predictions(original, tokenizer, trimmed, new_tokenizer, keep):
    """On kept text the trimmed model gives the original logits, restricted to kept tokens."""
    for line in KEEP_CORPUS:
        old_ids = tokenizer.encode(line)
        new_ids = new_tokenizer.encode(line)
        assert [keep[i] for i in new_ids] == old_ids
        torch.testing.assert_close(_logits(trimmed, new_ids), _logits(original, old_ids)[:, keep])


def test_logits_match_on_kept_tokens(case):
    _, tokenizer, original, keep, model, new_tokenizer = case
    _assert_same_predictions(original, tokenizer, model, new_tokenizer, keep)


def test_rows_and_config_shrunk(case):
    _, _, _, keep, model, _ = case
    assert model.get_input_embeddings().weight.shape[0] == len(keep)
    assert model.get_output_embeddings().weight.shape[0] == len(keep)
    assert model.get_output_embeddings().out_features == len(keep)
    assert model.config.vocab_size == len(keep)


def test_tied_weights_stay_tied(case):
    arch, _, original, _, model, _ = case
    tied = arch != "llama_untied"
    assert (original.get_output_embeddings().weight is original.get_input_embeddings().weight) == tied
    assert (model.get_output_embeddings().weight is model.get_input_embeddings().weight) == tied


def test_embedding_class_kept(case):
    """Gemma scales embeddings inside its own Embedding subclass; swapping the module would drop that."""
    _, _, original, _, model, _ = case
    assert type(model.get_input_embeddings()) is type(original.get_input_embeddings())


def test_special_token_ids_remapped(case):
    _, _, _, _, model, new_tokenizer = case
    assert model.config.bos_token_id == new_tokenizer.bos_token_id
    assert model.config.eos_token_id == new_tokenizer.eos_token_id
    assert model.config.pad_token_id == new_tokenizer.pad_token_id
    assert model.generation_config.eos_token_id == [new_tokenizer.eos_token_id, new_tokenizer.pad_token_id]


def test_rejects_ids_past_embedding_rows(make_tokenizer):
    tokenizer = make_tokenizer("byte_level")
    model = _tiny_model("llama_untied", tokenizer)
    keep = tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep=set())
    model.resize_token_embeddings(keep[-1])
    with pytest.raises(ValueError, match="embedding rows"):
        trim_model(model, tokenizer, keep)


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def checkpoint(request, make_tokenizer, spm_dir, tmp_path_factory):
    arch, kind = request.param
    tokenizer = make_tokenizer(kind)
    original = _tiny_model(arch, tokenizer)
    src = tmp_path_factory.mktemp(f"{arch}_src")
    original.save_pretrained(src, max_shard_size="20KB")  # several shards, like a real 8B checkpoint
    tokenizer.save_pretrained(src)
    if kind == "sentencepiece":
        (src / "tokenizer.model").write_bytes((spm_dir / "tokenizer.model").read_bytes())
    keep = tokens_to_keep(tokenizer, KEEP_CORPUS, always_keep=set())
    dst = tmp_path_factory.mktemp(f"{arch}_dst")
    trim_checkpoint(src, dst, keep)
    return kind, src, dst, tokenizer, original, keep


def test_checkpoint_reloads_with_same_predictions(checkpoint):
    _, src, dst, tokenizer, original, keep = checkpoint
    assert len(list(src.glob("*.safetensors"))) > 1
    model = AutoModelForCausalLM.from_pretrained(dst).eval()
    new_tokenizer = AutoTokenizer.from_pretrained(dst)
    _assert_same_predictions(original, tokenizer, model, new_tokenizer, keep)
    for line in HELD_OUT:
        assert new_tokenizer.decode(new_tokenizer.encode(line, add_special_tokens=False)) == tokenizer.decode(
            tokenizer.encode(line, add_special_tokens=False)
        )


def test_checkpoint_index_and_generation_config(checkpoint):
    _, _, dst, _, _, _ = checkpoint
    new_tokenizer = AutoTokenizer.from_pretrained(dst)
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) == {p.name for p in dst.glob("*.safetensors")}
    generation = json.loads((dst / "generation_config.json").read_text())
    assert generation["eos_token_id"] == [new_tokenizer.eos_token_id, new_tokenizer.pad_token_id]


def test_checkpoint_sentencepiece_model_trimmed(checkpoint):
    kind, src, dst, _, _, keep = checkpoint
    if kind != "sentencepiece":
        assert not (dst / "tokenizer.model").exists()
        return
    old = spm.SentencePieceProcessor(model_file=str(src / "tokenizer.model"))
    new = spm.SentencePieceProcessor(model_file=str(dst / "tokenizer.model"))
    for line in KEEP_CORPUS:
        assert [keep[i] for i in new.encode(line)] == old.encode(line)
