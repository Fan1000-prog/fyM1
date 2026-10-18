"""Blind greeting gate for the base-model pick.

Usage:
  python scripts/greeting_gate.py blind BASELINE.json [BASELINE.json ...] [--dir results/greeting_gate] [--seed N]
  python scripts/greeting_gate.py score --judge "<who judged>" [--dir results/greeting_gate]
                                        [--out results/greeting_gate.json]

`blind` writes <dir>/blind.jsonl (what the judge sees: id, prompt, response; no model names) and
<dir>/key.json (id -> model, prompt id). The judge writes <dir>/scores.jsonl in the rubric format
(src/fym1/eval/judge_rubric.md). `score` unblinds it and writes the per-model table.
"""

import argparse
import json
from pathlib import Path

from fym1.eval.greeting_gate import blind, gate_rows
from fym1.eval.scoring import read_jsonl

GATE_DIR = Path("results/greeting_gate")
OUT = Path("results/greeting_gate.json")
SEED = 20261018
RUBRIC = "src/fym1/eval/judge_rubric.md"
BLIND_FILE, KEY_FILE, SCORES_FILE = "blind.jsonl", "key.json", "scores.jsonl"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def run_blind(baselines: list[Path], gate_dir: Path, seed: int) -> None:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in baselines]
    items, key, variant_ids = blind(results, seed)
    gate_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(gate_dir / BLIND_FILE, items)
    meta = {"seed": seed, "prompts": sorted({r["greetings"]["source"] for r in results}),
            "sampling": sorted({r["greetings"]["sampling"] for r in results}),
            "variant_prompt_ids": variant_ids}
    (gate_dir / KEY_FILE).write_text(
        json.dumps(meta | {"key": key}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(items)} blinded items to {gate_dir / BLIND_FILE} and the key to {gate_dir / KEY_FILE}")


def run_score(gate_dir: Path, out: Path, judge: str) -> None:
    if not judge.strip():
        raise ValueError("--judge is empty; say who judged")
    items = read_jsonl(gate_dir / BLIND_FILE)
    key_file = json.loads((gate_dir / KEY_FILE).read_text(encoding="utf-8"))
    judgements = read_jsonl(gate_dir / SCORES_FILE)
    report = {
        "judge": judge,
        "rubric": RUBRIC,
        "blind_seed": key_file["seed"],
        "prompts": key_file["prompts"],
        "sampling": key_file["sampling"],
        "variant_prompt_ids": key_file["variant_prompt_ids"],
        "models": gate_rows(items, key_file["key"], judgements, key_file["variant_prompt_ids"]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} for {len(report['models'])} models")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    p_blind = sub.add_parser("blind")
    p_blind.add_argument("baselines", nargs="+", type=Path)
    p_blind.add_argument("--dir", type=Path, default=GATE_DIR)
    p_blind.add_argument("--seed", type=int, default=SEED)
    p_score = sub.add_parser("score")
    p_score.add_argument("--judge", required=True)
    p_score.add_argument("--dir", type=Path, default=GATE_DIR)
    p_score.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.step == "blind":
        run_blind(args.baselines, args.dir, args.seed)
    else:
        run_score(args.dir, args.out, args.judge)


if __name__ == "__main__":
    main()
