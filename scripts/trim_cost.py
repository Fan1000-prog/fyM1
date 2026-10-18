"""Measure what the public-text vocabulary trim costs: tokens per word, trimmed vs untrimmed tokenizer.

Usage:
    uv run python scripts/trim_cost.py --chats ~/malagasy-data/private/<run>/chats.jsonl \
        [--tokenizer MODEL_ID_OR_DIR] [--system-prompt-file PATH] [--out results/shrink/trim_cost.json]

The keep set comes from public text only (fym1.shrink.public_keep, security finding #13), so the anonymised chats
are a true held-out set here. They are private: this script writes and prints counts only, never a line of text
and never the chats path. The 150 public eval prompts are measured too. Needs the `shrink` group; downloads only
the tokenizer files and streams the pinned public text.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from fym1.shrink.public_keep import DEFAULT_SOURCES, public_keep_ids
from fym1.shrink.vocab import tokens_per_word, trim_tokenizer
from fym1.train.chat_format import CHAT_TEMPLATE_KWARGS

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "shrink" / "trim_cost.json"
PROMPTS = REPO / "eval" / "prompts.jsonl"
HELD_OUT_LINES = 2000
SEED = 42


def chosen_model() -> str:
    return json.loads((REPO / "results" / "model_selection.json").read_text("utf-8"))["chosen"]


def chat_lines(path: Path, n: int, seed: int = SEED) -> list[str]:
    """Up to `n` non-empty message texts from a chats.jsonl, shuffled with `seed`."""
    lines = []
    for raw in path.read_text("utf-8").splitlines():
        record = json.loads(raw)
        messages = record["messages"] if isinstance(record, dict) else record
        lines += [m["content"] for m in messages if m.get("content", "").strip()]
    random.Random(seed).shuffle(lines)
    return lines[:n]


def prompt_lines(path: Path = PROMPTS) -> list[str]:
    return [json.loads(raw)["prompt"] for raw in path.read_text("utf-8").splitlines() if raw.strip()]


def compare(untrimmed, trimmed, lines: list[str]) -> dict:
    """Counts only: tokens per word before and after, and how many lines decode unchanged."""
    before, after = tokens_per_word(untrimmed, lines), tokens_per_word(trimmed, lines)

    def decoded(tok, line):
        return tok.decode(tok.encode(line, add_special_tokens=False))

    return {
        "untrimmed": before,
        "trimmed": after,
        "token_ratio": round(after["tokens"] / before["tokens"], 4),
        "decode_equal": sum(decoded(trimmed, line) == decoded(untrimmed, line) for line in lines),
    }


def trim_cost(tokenizer, held_out: dict[str, list[str]], system_prompt: str = "", load=None, sources=DEFAULT_SOURCES):
    keep, counts = public_keep_ids(tokenizer, sources, system_prompt, load=load, **CHAT_TEMPLATE_KWARGS)
    trimmed = trim_tokenizer(tokenizer, keep)
    return {
        "keep_set": "public text only (fym1.shrink.public_keep)",
        "sources": [{"dataset": s.dataset, "config": s.config or "plt_Latn", "revision": s.revision} for s in sources],
        "public_lines_read": counts,
        "system_prompt_in_keep_corpus": bool(system_prompt),
        "vocab_before": len(tokenizer),
        "vocab_after": len(keep),
        "held_out": {name: compare(tokenizer, trimmed, lines) for name, lines in held_out.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chats", type=Path, required=True, help="anonymised chats.jsonl (private; never copied)")
    parser.add_argument("--tokenizer", default=None, help="model id or dir (default: results/model_selection.json)")
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--held-out", type=int, default=HELD_OUT_LINES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    model = args.tokenizer or chosen_model()
    tokenizer = AutoTokenizer.from_pretrained(model)
    system_prompt = args.system_prompt_file.read_text("utf-8").strip() if args.system_prompt_file else ""
    held_out = {"chats_anonymised_private": chat_lines(args.chats.expanduser(), args.held_out),
                "eval_prompts": prompt_lines()}
    result = {"tokenizer": model, **trim_cost(tokenizer, held_out, system_prompt)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, cost in result["held_out"].items():
        print(f"{name}: tokens/word {cost['untrimmed']['tokens_per_word']} -> {cost['trimmed']['tokens_per_word']} "
              f"({cost['untrimmed']['lines']} lines)")
    print(f"vocab {result['vocab_before']} -> {result['vocab_after']}; wrote {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
