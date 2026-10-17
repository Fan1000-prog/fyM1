"""Vocabulary trimming: keep only the tokens a language needs, without breaking BPE merges.

Supports Hugging Face fast tokenizers whose model is BPE, in both flavours used by the base-model
shortlist: byte-level BPE (Qwen, Llama 3) and SentencePiece-style BPE with byte fallback (Gemma,
Llama 2, Mistral).

Why merges need care: BPE builds a token by merging smaller tokens step by step. If a kept token's
parts are dropped, the merge that builds it can never fire and the token becomes unreachable, so
text that used it silently re-tokenizes differently. `tokens_to_keep` therefore closes the keep set
over every merge that produces a kept token, and `trim_tokenizer` refuses a keep set that is not
closed. With that closure, the trimmed merge list is an order-preserving subset of the original that
contains every merge the original applies to the kept text, so that text encodes to exactly the same
token sequence (renumbered). Unseen text still encodes, via byte tokens, and decodes unchanged.
"""

import json
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

_BYTE_FALLBACK_TOKEN = re.compile(r"^<0x[0-9A-F]{2}>$")
_ENCODE_BATCH_SIZE = 1000


def tokens_to_keep(tokenizer, corpus: Iterable[str], always_keep: set[int]) -> list[int]:
    """Sorted token ids to keep: ids seen in `corpus`, special and byte tokens, `always_keep`,
    closed over the BPE merges that build them."""
    tok_json = _tokenizer_json(tokenizer)
    model = _bpe_model(tok_json)
    out_of_range = sorted(i for i in always_keep if not 0 <= i < len(tokenizer))
    if out_of_range:
        raise ValueError(f"always_keep has ids outside the vocabulary: {out_of_range[:10]}")

    keep = set(always_keep) | _required_ids(tokenizer, tok_json)
    for batch in _batched(corpus, _ENCODE_BATCH_SIZE):
        for encoding in tokenizer.backend_tokenizer.encode_batch(batch, add_special_tokens=False):
            keep.update(encoding.ids)
    return sorted(_merge_closure(model, keep))


def chat_texts(tokenizer, conversations: Iterable[list[dict]], system_prompt: str = "", **template_kwargs) -> Iterator[str]:
    """Keep-corpus texts for chat data, rendered exactly as the model sees them.

    Yields each whole conversation and, before every assistant turn, the prefix rendered with
    `add_generation_prompt=True`: generation prompts (e.g. a "thought" channel header) only appear
    there. Pass the same `template_kwargs` (e.g. `enable_thinking=False`) used for training and serving.
    """
    for conversation in conversations:
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(conversation)
        yield tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
        for turn, message in enumerate(messages):
            if message["role"] == "assistant":
                yield tokenizer.apply_chat_template(
                    messages[:turn], tokenize=False, add_generation_prompt=True, **template_kwargs
                )


def trim_tokenizer(tokenizer, keep_ids: list[int]):
    """Return a new tokenizer holding only `keep_ids`; new id `n` is old id `keep_ids[n]`."""
    tok_json = _tokenizer_json(tokenizer)
    model = _bpe_model(tok_json)
    _check_keep_ids(tokenizer, tok_json, model, keep_ids)
    new_id = {old: new for new, old in enumerate(keep_ids)}
    _remap_tokenizer_json(tok_json, new_id)

    from transformers import AutoTokenizer

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        tokenizer.save_pretrained(directory)
        # Files that would let the tokenizer rebuild itself from the untrimmed vocabulary.
        for stale in ("tokenizer.model", "vocab.json", "merges.txt"):
            (directory / stale).unlink(missing_ok=True)
        (directory / "tokenizer.json").write_text(json.dumps(tok_json, ensure_ascii=False), encoding="utf-8")
        _remap_tokenizer_files(directory, new_id)
        return AutoTokenizer.from_pretrained(directory)


def trim_sentencepiece(model_proto: bytes, keep_ids: list[int]) -> bytes:
    """Trim a SentencePiece `tokenizer.model` to `keep_ids` (ids past its pieces are ignored).

    llama.cpp's GGUF converter reads `tokenizer.model`, not `tokenizer.json`, for SentencePiece
    models (Gemma), so it must be trimmed in step with the fast tokenizer.
    """
    from sentencepiece import sentencepiece_model_pb2

    if keep_ids != sorted(set(keep_ids)):
        raise ValueError("keep_ids must be sorted and unique")
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(model_proto)
    kept = [i for i in keep_ids if i < len(proto.pieces)]
    new_id = {old: new for new, old in enumerate(kept)}
    byte_type = sentencepiece_model_pb2.ModelProto.SentencePiece.BYTE
    dropped_bytes = [p.piece for i, p in enumerate(proto.pieces) if p.type == byte_type and i not in new_id]
    if dropped_bytes:
        raise ValueError(f"keep_ids drops SentencePiece byte pieces {dropped_bytes[:10]}")

    pieces = []
    for old in kept:
        piece = sentencepiece_model_pb2.ModelProto.SentencePiece()
        piece.CopyFrom(proto.pieces[old])
        pieces.append(piece)
    del proto.pieces[:]
    proto.pieces.extend(pieces)

    spec = proto.trainer_spec
    for field in ("unk_id", "bos_id", "eos_id", "pad_id"):
        old = getattr(spec, field)
        if old < 0:
            continue
        if old not in new_id:
            raise ValueError(f"SentencePiece {field}={old} is not in keep_ids")
        setattr(spec, field, new_id[old])
    spec.vocab_size = len(kept)
    return proto.SerializeToString()


def _tokenizer_json(tokenizer) -> dict:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise NotImplementedError("vocabulary trimming needs a fast (tokenizers-backed) tokenizer")
    return json.loads(backend.to_str())


def _bpe_model(tok_json: dict) -> dict:
    model = tok_json["model"]
    if model.get("type") != "BPE":
        raise NotImplementedError(f"only BPE tokenizers are supported, got {model.get('type')!r}")
    if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
        raise NotImplementedError("BPE with subword prefix/suffix markers is not supported")
    return model


def _merge_pairs(model: dict) -> list[tuple[str, str]]:
    """Merges as (left, right) pairs; tokenizer.json stores them as lists or as "left right"."""
    return [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in model["merges"]]


def _uses_byte_level(node) -> bool:
    if isinstance(node, dict):
        return node.get("type") == "ByteLevel" or any(_uses_byte_level(v) for v in node.values())
    if isinstance(node, list):
        return any(_uses_byte_level(v) for v in node)
    return False


def _required_ids(tokenizer, tok_json: dict) -> set[int]:
    """Ids that must survive any trim: added/special tokens, unk, and the base pieces that let
    unseen text still be encoded."""
    model = tok_json["model"]
    vocab = model["vocab"]
    required = {t["id"] for t in tok_json["added_tokens"]} | set(tokenizer.all_special_ids)
    if model.get("unk_token") in vocab:
        required.add(vocab[model["unk_token"]])

    if model.get("byte_fallback"):
        required |= {i for t, i in vocab.items() if _BYTE_FALLBACK_TOKEN.match(t)}
        # Some converters (Gemma) store an ASCII byte piece as the character itself, e.g. "\t" for <0x09>.
        required |= {vocab[chr(b)] for b in range(128) if f"<0x{b:02X}>" not in vocab and chr(b) in vocab}
    elif _uses_byte_level(tok_json.get("pre_tokenizer")):
        from tokenizers.pre_tokenizers import ByteLevel

        required |= {vocab[c] for c in ByteLevel.alphabet() if c in vocab}
    else:
        required |= {i for t, i in vocab.items() if len(t) == 1}
    return required


def _merge_closure(model: dict, ids: set[int]) -> set[int]:
    """`ids` plus, recursively, both parts of every merge that produces a token in the set."""
    vocab = model["vocab"]
    token_of = {i: t for t, i in vocab.items()}
    producers = defaultdict(list)
    for left, right in _merge_pairs(model):
        producers[left + right].append((left, right))

    closed = set(ids)
    stack = [token_of[i] for i in ids if i in token_of]
    while stack:
        for pair in producers.get(stack.pop(), ()):
            for part in pair:
                part_id = vocab[part]
                if part_id not in closed:
                    closed.add(part_id)
                    stack.append(part)
    return closed


def _check_keep_ids(tokenizer, tok_json: dict, model: dict, keep_ids: list[int]) -> None:
    if keep_ids != sorted(set(keep_ids)):
        raise ValueError("keep_ids must be sorted and unique")
    if keep_ids and not 0 <= keep_ids[0] <= keep_ids[-1] < len(tokenizer):
        raise ValueError("keep_ids has ids outside the vocabulary")
    keep = set(keep_ids)
    missing = _required_ids(tokenizer, tok_json) - keep
    if missing:
        raise ValueError(f"keep_ids drops {len(missing)} special/byte tokens, e.g. {sorted(missing)[:10]}")
    unreachable = _merge_closure(model, keep) - keep
    if unreachable:
        raise ValueError(
            f"keep_ids drops {len(unreachable)} merge parents of kept tokens, e.g. {sorted(unreachable)[:10]}; "
            "build keep_ids with tokens_to_keep"
        )


def _remap_tokenizer_json(tok_json: dict, new_id: dict[int, int]) -> None:
    model = tok_json["model"]
    model["vocab"] = {t: new_id[i] for t, i in model["vocab"].items() if i in new_id}
    vocab = model["vocab"]
    model["merges"] = [
        m
        for m, (left, right) in zip(model["merges"], _merge_pairs(model), strict=True)
        if left in vocab and right in vocab and left + right in vocab
    ]
    for token in tok_json["added_tokens"]:
        token["id"] = new_id[token["id"]]
    _remap_post_processor(tok_json.get("post_processor"), new_id)


def _remap_post_processor(node, new_id: dict[int, int]) -> None:
    """Rewrite token ids baked into post-processors (TemplateProcessing, Bert/Roberta processing)."""
    if isinstance(node, list):
        for item in node:
            _remap_post_processor(item, new_id)
        return
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key == "ids" and isinstance(value, list):
            node[key] = [new_id[i] for i in value]
        elif key in ("sep", "cls") and isinstance(value, list) and len(value) == 2:
            node[key] = [value[0], new_id[value[1]]]
        else:
            _remap_post_processor(value, new_id)


def _remap_tokenizer_files(directory: Path, new_id: dict[int, int]) -> None:
    """Rewrite ids stored next to tokenizer.json by `save_pretrained` (older transformers)."""
    config_path = directory / "tokenizer_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.pop("vocab_file", None)
        if "added_tokens_decoder" in config:
            config["added_tokens_decoder"] = {
                str(new_id[int(i)]): token for i, token in config["added_tokens_decoder"].items()
            }
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    added_path = directory / "added_tokens.json"
    if added_path.exists():
        added = json.loads(added_path.read_text(encoding="utf-8"))
        added_path.write_text(json.dumps({t: new_id[i] for t, i in added.items()}, ensure_ascii=False))


def _batched(items: Iterable[str], size: int) -> Iterator[list[str]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
