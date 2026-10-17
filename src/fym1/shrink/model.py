"""Slice a saved causal LM checkpoint's embedding and output rows down to a trimmed vocabulary.

Works shard by shard on a 16-bit safetensors checkpoint, so a 7-12B model can be trimmed in Colab RAM
before the quality gates and GGUF export.
"""

import json
import shutil
from pathlib import Path

import torch

from fym1.shrink.vocab import trim_sentencepiece, trim_tokenizer

# Tensors holding one row per token in Llama / Qwen / Gemma checkpoints (tied models have no lm_head).
_VOCAB_TENSOR_SUFFIXES = ("embed_tokens.weight", "lm_head.weight", "lm_head.bias")
# Tokenizer files written by trim_tokenizer; never copied from the untrimmed source.
_INDEX_NAME = "model.safetensors.index.json"
_TOKENIZER_FILE_PREFIXES = ("tokenizer", "added_tokens", "special_tokens_map", "vocab.json", "merges.txt")


def trim_checkpoint(src: str | Path, dst: str | Path, keep_ids: list[int]) -> None:
    """Write a trimmed copy of the safetensors checkpoint + tokenizer in `src` to `dst`.

    Loads one shard at a time. Also trims `tokenizer.model` when present, because llama.cpp's
    GGUF converter reads it for SentencePiece models.
    """
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    src, dst = Path(src), Path(dst)
    index, shards = _read_shard_index(src)  # validate untrusted shard names before writing anything
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

    _trim_shards(src, dst, index, shards, keep_ids, rows)
    for path in src.iterdir():
        if path.is_file() and not (dst / path.name).exists() and not path.name.startswith(_TOKENIZER_FILE_PREFIXES):
            shutil.copy2(path, dst / path.name)


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


def _read_shard_index(src: Path) -> tuple[dict | None, list[str]]:
    """The index (if sharded) and shard file names; names come from a downloaded file, so they
    must be plain file names that cannot leave `src` / `dst`."""
    index_path = src / _INDEX_NAME
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    shards = sorted(set(index["weight_map"].values())) if index else ["model.safetensors"]
    for shard in shards:
        if not isinstance(shard, str) or Path(shard).name != shard or shard in ("", ".", ".."):
            raise ValueError(f"unsafe shard name in {_INDEX_NAME}: {shard!r}")
    return index, shards


def _trim_shards(src: Path, dst: Path, index: dict | None, shards: list[str], keep_ids: list[int], rows: int) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

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
        (dst / _INDEX_NAME).write_text(json.dumps(index, indent=2))


def _is_vocab_tensor(name: str, shape: list[int], rows: int) -> bool:
    return name.endswith(_VOCAB_TENSOR_SUFFIXES) and len(shape) > 0 and shape[0] == rows
