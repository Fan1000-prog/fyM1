"""Build the blind judge batch, then assemble results/eval/scores.json.

Step 1, before judging:
  uv run python scripts/score.py blind \
      --answers results/eval/answers_base.jsonl results/eval/answers_fym1.jsonl \
      --out results/eval/judge_input.jsonl --key results/eval/judge_key.json

Judge agents score judge_input.jsonl with src/fym1/eval/judge_rubric.md and write a JSON list
of {"id", "naturalness", "correctness", "note"} to results/eval/judge.json.

Step 2, after judging:
  uv run python scripts/score.py assemble \
      --answers results/eval/answers_base.jsonl results/eval/answers_fym1.jsonl \
      --metrics results/eval/notebook_metrics.json \
      --judge results/eval/judge.json --key results/eval/judge_key.json \
      --out results/eval/scores.json

notebook_metrics.json is {"<model>": {"leak_rate": ..., "perplexity": ..., ...}} from
notebooks/02_eval_export.ipynb. --metrics and --judge are optional; missing parts are left out.
"""

import argparse
import json
from pathlib import Path

from fym1.eval.scoring import assemble_scores, blind_batch, judge_means, model_name, read_jsonl

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


def cmd_assemble(args: argparse.Namespace) -> None:
    if args.judge and not args.key:
        raise SystemExit("--judge needs --key (the judge_key.json written by `blind`)")
    metrics = read_json(args.metrics) if args.metrics else {}
    judge = judge_means(read_json(args.judge), read_json(args.key)) if args.judge else {}
    scores = assemble_scores(load_answers(args.answers), metrics=metrics, judge=judge)
    write_json(args.out, scores)
    print(f"wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(required=True)

    blind = sub.add_parser("blind", help="shuffle answers into a blind batch for the judge")
    blind.add_argument("--answers", type=Path, nargs="+", required=True)
    blind.add_argument("--out", type=Path, required=True)
    blind.add_argument("--key", type=Path, required=True)
    blind.set_defaults(run=cmd_blind)

    assemble = sub.add_parser("assemble", help="write scores.json")
    assemble.add_argument("--answers", type=Path, nargs="+", required=True)
    assemble.add_argument("--metrics", type=Path)
    assemble.add_argument("--judge", type=Path)
    assemble.add_argument("--key", type=Path)
    assemble.add_argument("--out", type=Path, required=True)
    assemble.set_defaults(run=cmd_assemble)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
