"""Build the blind judge batch, then results/eval/judge.json and results/eval/scores.json.

Step 1, before judging:
  uv run python scripts/score.py blind \
      --answers results/eval/answers_base.jsonl results/eval/answers_fym1.jsonl \
      --out results/eval/judge_input.jsonl --key results/eval/judge_key.json

Judge agents score judge_input.jsonl with src/fym1/eval/judge_rubric.md and write a JSON list
of {"id", "naturalness", "correctness", "note"} to results/eval/judge_raw.json.

Step 2, unblind the judge scores:
  uv run python scripts/score.py judge \
      --raw results/eval/judge_raw.json --key results/eval/judge_key.json \
      --label "<who judged>" --out results/eval/judge.json

Step 3, automatic scores:
  uv run python scripts/score.py scores \
      --answers results/eval/answers_base.jsonl results/eval/answers_fym1.jsonl \
      --metrics results/eval/notebook_metrics.json --out results/eval/scores.json

notebook_metrics.json is {"<model>": {"leak_rate", "perplexity", "belebele_acc", "chrf_plt_eng",
"chrf_eng_plt"}} from notebooks/02_eval_export.ipynb. --metrics is optional; missing metrics stay
out of scores.json and the paper shows them as pending.
"""

import argparse
import json
from pathlib import Path

from fym1.eval.scoring import (
    assemble_scores,
    blind_batch,
    judge_means,
    judge_summary,
    model_name,
    read_jsonl,
)

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


def cmd_blind(args: argparse.Namespace) -> None:
    items, key = blind_batch(load_answers(args.answers), seed=BLIND_SEED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
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

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
