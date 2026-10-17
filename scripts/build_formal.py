"""Malagasy Wikipedia -> single-turn "tell me about" chats (the small formal slice).

    uv run python scripts/build_formal.py --n 1500 --out ~/malagasy-data/private/formal.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from fym1.data.formal import pick_paragraph, to_formal_chat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1500, help="chats to write")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", default="20231101.mg", help="wikimedia/wikipedia config")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    articles = load_dataset("wikimedia/wikipedia", args.config, split="train").shuffle(seed=args.seed)

    records = []
    for article in articles:  # one paragraph per article, for variety
        paragraph = pick_paragraph(article["text"], rng)
        if paragraph is None:
            continue
        chat = to_formal_chat(article["title"], paragraph, rng)
        records.append({"id": f"wiki-{article['id']}", "messages": chat})
        if len(records) == args.n:
            break
    if len(records) < args.n:
        raise SystemExit(f"Only {len(records)} of {args.n} articles have a paragraph of the right length; lower --n.")

    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(json.dumps({"articles_total": articles.num_rows, "chats": len(records)}))


if __name__ == "__main__":
    main()
