"""Automatic identity check of fy-m1's answers (release gate).

Usage:
  uv run python scripts/identity_check.py \
      --answers results/eval/answers_fym1.jsonl results/eval/answers_fym1_identity.jsonl \
      [--prompts eval/identity_check.jsonl] [--out results/eval/identity_check.json]

answers_fym1.jsonl is the normal eval export (the 150 prompts, of which the 20 identity/safety ones
are checked); answers_fym1_identity.jsonl holds the export's answers to the extra identity prompts
(rows of eval/identity_check.jsonl with "set": "identity"). Both are private and git-ignored.

Writes results/eval/identity_check.json (counts, pass rate, threshold, gate verdict, failed ids with
reason codes; never reply text) and exits 1 when the gate fails. Rules: src/fym1/eval/identity_check.py.
"""

import argparse
import json
import sys
from pathlib import Path

from fym1.eval.identity_check import identity_report
from fym1.eval.scoring import read_jsonl

PROMPTS = Path("eval/identity_check.jsonl")
OUT = Path("results/eval/identity_check.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", type=Path, nargs="+", required=True)
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    answers = [row for path in args.answers for row in read_jsonl(path)]
    report = identity_report(read_jsonl(args.prompts), answers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verdict = "passed" if report["gate_passed"] else "FAILED"
    print(f"identity check {verdict}: {report['passed']}/{report['n']} passed "
          f"(threshold {report['threshold']:.0%}), {report['foreign_identity']} foreign identity claim(s); "
          f"wrote {args.out}")
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
