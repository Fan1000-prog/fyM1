"""Malagasy Wikipedia -> the most widespread words, ranked (src/fym1/data/mg_common_words.txt).

The anonymiser uses it so names never wipe everyday words ("mbola", "raha", "ny").

    uv run python scripts/build_common_words.py
"""

import argparse
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from fym1.data.anonymize import fold

OUT = Path(__file__).resolve().parents[1] / "src/fym1/data/mg_common_words.txt"
LETTERS = re.compile(r"[^\W\d_]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top", type=int, default=3000)
    parser.add_argument("--config", default="20231101.mg", help="wikimedia/wikipedia config")
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    articles = load_dataset("wikimedia/wikipedia", args.config, split="train")
    articles_per_word = Counter()  # ranked by how many articles use a word, not raw count
    for article in articles:
        articles_per_word.update({fold(w) for w in LETTERS.findall(article["text"])})
    words = [w for w, _ in articles_per_word.most_common(args.top)]
    args.out.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"{len(words)} words -> {args.out}")


if __name__ == "__main__":
    main()
