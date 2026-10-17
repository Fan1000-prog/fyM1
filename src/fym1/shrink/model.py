"""Slice a causal LM's embedding and output rows down to a trimmed vocabulary.

`trim_model` works on a loaded model (used for the quality gates; bitsandbytes 4-bit loading keeps
both embedding matrices in 16-bit, so it works there too). `trim_checkpoint` works shard by shard on
a saved 16-bit checkpoint, so a 7-12B model can be trimmed in Colab RAM before GGUF export.
"""

import json
import shutil
from pathlib import Path

import torch

from fym1.shrink.vocab import trim_sentencepiece, trim_tokenizer

# Tensors holding one row per token in Llama / Qwen / Gemma checkpoints (tied models have no lm_head).
_VOCAB_TENSOR_SUFFIXES = ("embed_tokens.weight", "lm_head.weight", "lm_head.bias")
# Tokenizer files written by trim_tokenizer; never copied from the untrimmed source.
_TOKENIZER_FILE_PREFIXES = ("tokenizer", "added_tokens", "special_tokens_map", "vocab.json", "merges.txt")


def trim_model(model, tokenizer, keep_ids: list[int]):
    """Trim `model` in place and `tokenizer` to `keep_ids`; return (model, new tokenizer).

    Row `n` of the new embedding and output matrices is old row `keep_ids[n]`. The embedding
    module is kept (only its weight is replaced), so subclasses such as Gemma's scaled embedding
    keep working, and tied input/output weights stay tied.
    """
    new_tokenizer = trim_tokenizer(tokenizer, keep_ids)
    embeddings = model.get_input_embeddings()
    output = model.get_output_embeddings()
    rows = embeddings.weight.shape[0]
    if keep_ids[-1] >= rows:
        raise ValueError(f"keep_ids reaches id {keep_ids[-1]} but the model has {rows} embedding rows")
    tied = output is not None and output.weight is embeddings.weight
    new_id = {old: new for new, old in enumerate(keep_ids)}

    embeddings.weight = _select_rows(embeddings.weight, keep_ids)
    embeddings.num_embeddings = len(keep_ids)
    if embeddings.padding_idx is not None:
        embeddings.padding_idx = new_id.get(embeddings.padding_idx)
    if output is not None:
        if output.weight.shape[0] != rows:
            raise NotImplementedError("output layer and input embeddings have different row counts")
        output.weight = embeddings.weight if tied else _select_rows(output.weight, keep_ids)
        if output.bias is not None:
            output.bias = _select_rows(output.bias, keep_ids)
        output.out_features = len(keep_ids)

    _update_config(model.config, new_id)
    if getattr(model, "generation_config", None) is not None:
        _remap_token_ids(model.generation_config, new_id)
    return model, new_tokenizer


def trim_checkpoint(src: str | Path, dst: str | Path, keep_ids: list[int]) -> None:
    """Write a trimmed copy of the safetensors checkpoint + tokenizer in `src` to `dst`.

    Loads one shard at a time. Also trims `tokenizer.model` when present, because llama.cpp's
    GGUF converter reads it for SentencePiece models.
    """
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    new_id = {old: new for new, old in enumerate(keep_ids)}

    trim_tokenizer(AutoTokenizer.from_pretrained(src), keep_ids).save_pretrained(dst)
    if (src / "tokenizer.model").exists():
        (dst / "tokenizer.model").write_bytes(trim_sentencepiece((src / "tokenizer.model").read_bytes(), keep_ids))

    config = AutoConfig.from_pretrained(src)
    rows = config.get_text_config().vocab_size
    if keep_ids[-1] >= rows:
        raise ValueError(f"keep_ids reaches id {keep_ids[-1]} but the model has {rows} embedding rows")
    _update_config(config, new_id)
    config.save_pretrained(dst)
    if (src / "generation_config.json").exists():
        generation_config = GenerationConfig.from_pretrained(src)
        _remap_token_ids(generation_config, new_id)
        generation_config.save_pretrained(dst)

    _trim_shards(src, dst, keep_ids, rows)
    for path in src.iterdir():
        if path.is_file() and not (dst / path.name).exists() and not path.name.startswith(_TOKENIZER_FILE_PREFIXES):
            shutil.copy2(path, dst / path.name)


def _select_rows(weight: torch.Tensor, keep_ids: list[int]) -> torch.nn.Parameter:
    if not weight.is_floating_point() or type(weight) not in (torch.Tensor, torch.nn.Parameter):
        raise NotImplementedError(f"cannot slice a {type(weight).__name__} of {weight.dtype}; load it in 16-bit")
    index = torch.tensor(keep_ids, device=weight.device)
    return torch.nn.Parameter(weight.detach().index_select(0, index).clone(), requires_grad=weight.requires_grad)


def _update_config(config, new_id: dict[int, int]) -> None:
    text_config = config.get_text_config()  # the config itself for text-only models
    for cfg in [config] if text_config is config else [config, text_config]:
        _remap_token_ids(cfg, new_id)
    text_config.vocab_size = len(new_id)


def _remap_token_ids(config, new_id: dict[int, int]) -> None:
    """Renumber every `*_token_id(s)` / `*_token_index` field (bos, eos, pad, image tokens, ...)."""
    for key, value in config.to_dict().items():
        if value is None or not key.endswith(("_token_id", "_token_ids", "_token_index")):
            continue
        if not isinstance(value, int | list):
            continue
        ids = value if isinstance(value, list) else [value]
        missing = [i for i in ids if i not in new_id]
        if missing:
            raise ValueError(f"config {key}={value} refers to ids not in keep_ids: {missing}")
        remapped = [new_id[i] for i in ids]
        setattr(config, key, remapped if isinstance(value, list) else remapped[0])


def _trim_shards(src: Path, dst: Path, keep_ids: list[int], rows: int) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    shards = sorted(set(index["weight_map"].values())) if index else ["model.safetensors"]
    for shard in shards:  # names come from a downloaded file: never let them leave src / dst
        if not isinstance(shard, str) or Path(shard).name != shard or shard in ("", ".", ".."):
            raise ValueError(f"unsafe shard name in {index_path.name}: {shard!r}")
    keep_index = torch.tensor(keep_ids)
    trimmed, removed_bytes = [], 0

    for shard in shards:
        with safe_open(src / shard, framework="pt") as f:
            shapes = {name: f.get_slice(name).get_shape() for name in f.keys()}
            to_trim = [name for name, shape in shapes.items() if _is_vocab_tensor(name, shape, rows)]
            unexpected = [n for n, s in shapes.items() if n not in to_trim and len(s) > 0 and s[0] == rows]
            if unexpected:
                raise NotImplementedError(f"vocabulary-sized tensors this trimmer does not handle: {unexpected}")
            if not to_trim:
                shutil.copy2(src / shard, dst / shard)
                continue
            tensors = {name: f.get_tensor(name) for name in f.keys()}
            metadata = f.metadata()
        for name in to_trim:
            before = tensors[name]
            tensors[name] = before.index_select(0, keep_index).contiguous()
            removed_bytes += (before.numel() - tensors[name].numel()) * before.element_size()
        save_file(tensors, dst / shard, metadata=metadata or {"format": "pt"})
        trimmed.extend(to_trim)

    if not trimmed:
        raise ValueError(f"no embedding tensors with {rows} rows found in {src}")
    if index:
        index.setdefault("metadata", {})
        if "total_size" in index["metadata"]:
            index["metadata"]["total_size"] -= removed_bytes
        (dst / index_path.name).write_text(json.dumps(index, indent=2))


def _is_vocab_tensor(name: str, shape: list[int], rows: int) -> bool:
    return name.endswith(_VOCAB_TENSOR_SUFFIXES) and len(shape) > 0 and shape[0] == rows
