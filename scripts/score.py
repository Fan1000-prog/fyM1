"""Build the blind judge batch, then results/eval/judge.json and results/eval/scores.json.

Privacy: answers_*.jsonl, judge_input.jsonl, judge_key.json and judge_raw.json show which
model wrote each answer. They are git-ignored, live in the private fyM1-ckpt repo under
results/, and stay private until the arena closes (v0.2). Download them into results/eval/
to run the steps below. Only judge.json and scores.json (aggregates) are committed.

Step 1, before judging:
  uv run python scripts/score.py blind \
      --answers results/eval/answers_base.jsonl results/eval/answers_fym1.jsonl \
      --out results/eval/judge_input.jsonl --key results/eval/judge_key.json

Judge agents score judge_input.jsonl with src/fym1/eval/judge_rubric.md and write a JSON list
of {"id", "naturalness", "correctness", "note"} to results/eval/judge_raw.json.

Step 2, unblind the judge scores:
  uv run python scripts/score.py judge \
      --raw results/eval/judge_raw.json --key results/eval/judge_key.json \
      --label "Claude Opus 5.5 (claude-opus-5-5), blind, run as judging agents in Claude Code; rubric in src/fym1/eval/judge_rubric.md" \
      --out results/eval/judge.json

  --label is the methods disclosure the paper cites for who judged; the value above is the one
  used for v0.1.

Step 3, automatic scores:
  uv run python scripts/score.py scores \
      --answers results/eval/answers_base.jsonl results/eval/answers_fym1.jsonl \
      --metrics results/eval/notebook_metrics.json --out results/eval/scores.json

notebook_metrics.json is {"<model>": {"leak_rate", "perplexity", "belebele_acc", "chrf_plt_eng",
"chrf_eng_plt"}} from notebooks/02_eval_export.ipynb. --metrics is optional; missing metrics stay
out of scores.json and the paper shows them as pending.

Leak gate (release rule, run on the owner's laptop):
  uv run python scripts/score.py leaks \
      --answers results/eval/answers_fym1.jsonl --train ~/malagasy-data/private/chats.jsonl \
      --matches results/eval/leak_matches.jsonl --review results/eval/leak_review.json

  leak_matches.jsonl (private, git-ignored) holds one {"id", "ngram", "verdict": null} row per
  8-gram that a reply shares with the training chats: only the 8 words, never the chat around
  them. The owner sets each "verdict" to "generic" (an everyday phrase anyone would say) or
  "rejected" (copied from a real chat), then recounts:
  uv run python scripts/score.py leak-review \
      --matches results/eval/leak_matches.jsonl --review results/eval/leak_review.json

  leak_review.json is counts only: {"matches", "approved_generic", "rejected"}.
  uv run python scripts/score.py leak-gate --review results/eval/leak_review.json
  exits 0 only if there are no matches or every match is approved generic;
  otherwise 1 (do not ship).
"""

import argparse
import json
import sys
from pathlib import Path

from fym1.eval.scoring import (
    assemble_scores,
    blind_batch,
    chat_texts,
    judge_means,
    judge_summary,
    leak_gate_passes,
    leak_review,
    model_name,
    read_jsonl,
)
from fym1.eval.metrics import leak_matches

BLIND_SEED = 42


def load_answers(paths: list[Path]) -> dict[str, list[dict]]:
    answers = {model_name(path): read_jsonl(path) for path in paths}
    if len(answers) != len(paths):
        raise SystemExit("two answer files map to the same model name")
    return answers


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def cmd_blind(args: argparse.Namespace) -> None:
    items, key = blind_batch(load_answers(args.answers), seed=BLIND_SEED)
    write_jsonl(args.out, items)
    write_json(args.key, key)
    print(f"wrote {len(items)} blind items to {args.out}")


def cmd_judge(args: argparse.Namespace) -> None:
    means = judge_means(read_json(args.raw), read_json(args.key))
    write_json(args.out, judge_summary(means, judge=args.label))
    print(f"wrote {args.out}")


def cmd_scores(args: argparse.Namespace) -> None:
    metrics = read_json(args.metrics) if args.metrics else {}
    write_json(args.out, assemble_scores(load_answers(args.answers), metrics=metrics))
    print(f"wrote {args.out}")


def cmd_leaks(args: argparse.Namespace) -> None:
    if args.matches.exists():
        # It may already hold the owner's verdicts; never overwrite them silently.
        raise SystemExit(f"{args.matches} exists; move it away first to start a new leak review")
    train = [text for path in args.train for text in chat_texts(read_jsonl(path.expanduser()))]
    matches = [match | {"verdict": None} for match in leak_matches(read_jsonl(args.answers), train)]
    write_jsonl(args.matches, matches)
    write_json(args.review, leak_review(matches))
    print(f"{len(matches)} leaked 8-gram(s); see {args.matches} (private)")


def cmd_leak_review(args: argparse.Namespace) -> None:
    review = leak_review(read_jsonl(args.matches))
    write_json(args.review, review)
    print(f"wrote {args.review}: {review}")


def cmd_leak_gate(args: argparse.Namespace) -> None:
    review = read_json(args.review)
    if not leak_gate_passes(review):
        sys.exit(f"leak gate BLOCKED: {review}; every match must be reviewed and approved generic")
    print(f"leak gate passed: {review}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(required=True)

    blind = sub.add_parser("blind", help="shuffle answers into a blind batch for the judge")
    blind.add_argument("--answers", type=Path, nargs="+", required=True)
    blind.add_argument("--out", type=Path, required=True)
    blind.add_argument("--key", type=Path, required=True)
    blind.set_defaults(run=cmd_blind)

    judge = sub.add_parser("judge", help="unblind judge scores into judge.json")
    judge.add_argument("--raw", type=Path, required=True)
    judge.add_argument("--key", type=Path, required=True)
    judge.add_argument("--label", required=True, help="who judged, as cited in the paper")
    judge.add_argument("--out", type=Path, required=True)
    judge.set_defaults(run=cmd_judge)

    scores = sub.add_parser("scores", help="write scores.json")
    scores.add_argument("--answers", type=Path, nargs="+", required=True)
    scores.add_argument("--metrics", type=Path)
    scores.add_argument("--out", type=Path, required=True)
    scores.set_defaults(run=cmd_scores)

    leaks = sub.add_parser("leaks", help="list leaked 8-grams for owner review")
    leaks.add_argument("--answers", type=Path, required=True, help="the model to ship")
    leaks.add_argument("--train", type=Path, nargs="+", required=True, help="training chats")
    leaks.add_argument("--matches", type=Path, required=True)
    leaks.add_argument("--review", type=Path, required=True)
    leaks.set_defaults(run=cmd_leaks)

    review = sub.add_parser("leak-review", help="recount owner verdicts into leak_review.json")
    review.add_argument("--matches", type=Path, required=True)
    review.add_argument("--review", type=Path, required=True)
    review.set_defaults(run=cmd_leak_review)

    gate = sub.add_parser("leak-gate", help="exit non-zero unless the leak gate passes")
    gate.add_argument("--review", type=Path, required=True)
    gate.set_defaults(run=cmd_leak_gate)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
