"""Turn the measured outputs in results/ into the numbers the paper and README cite.

Writes paper/numbers.tex (one \\newcommand per number) and, with --markdown, rewrites the
results table between the <!-- numbers:start --> / <!-- numbers:end --> markers in each given file.
A missing file or key renders as "pending", so the paper always builds and no number is
ever typed by hand.

Usage: uv run python paper/numbers.py [--results results] [--out paper/numbers.tex] [--markdown README.md MODEL_CARD.md]

Expected files (all optional; keys are dotted paths below):
  data_stats.json        threads, conversations, approved, identity, formal, synthetic, train_examples
  model_selection.json   chosen, chosen_size, license, candidates[{model, license, belebele_acc,
                         chrf_plt_eng, chrf_eng_plt, tokens_per_word, sniff_natural}]
  train_log.json         gpu, steps, epochs, wall_time_s, final_train_loss, best_eval_loss,
                         config.{lora_r, lora_alpha, lr, max_seq_len, effective_batch}
  eval/scores.json       n_prompts, {base,fym1}.{drift_rate, leak_rate, perplexity,
                         belebele_acc, chrf_plt_eng, chrf_eng_plt}
  eval/judge.json        judge, {base,fym1}.{naturalness, correctness}
  eval/arena.json        votes, raters, fym1_win_rate, ci_low, ci_high
  shrink.json            gates_passed, vocab_before, vocab_after, params_before, params_after,
                         gguf_bytes_before, gguf_bytes_after, tokens_per_sec_before, tokens_per_sec_after
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PENDING = "pending"
MACRO_PREFIX = "fy"
README_START = "<!-- numbers:start -->"
README_END = "<!-- numbers:end -->"

_DIGIT_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


# --- formatters: JSON value -> plain text (LaTeX escaping happens on output) ---


def count(v: Any) -> str:
    return f"{int(v):,}"


def fixed(digits: int) -> Callable[[Any], str]:
    return lambda v: f"{float(v):.{digits}f}"


def percent(v: Any) -> str:
    return f"{float(v) * 100:.1f}%"


def billions(v: Any) -> str:
    return f"{float(v) / 1e9:.2f}B"


def gigabytes(v: Any) -> str:
    return f"{float(v) / 1e9:.2f} GB"


def hours(v: Any) -> str:
    return f"{float(v) / 3600:.1f} h"


def text(v: Any) -> str:
    return str(v)


def passed(v: Any) -> str:
    if not isinstance(v, bool):
        raise TypeError(f"expected a bool, got {v!r}")
    return "passed" if v else "failed"


@dataclass(frozen=True)
class Number:
    name: str  # human-readable id, e.g. "drift-rate-fym1"; becomes the macro \fyDriftRateFymOne
    file: str  # path under results/
    key: str  # dotted path inside the JSON file
    fmt: Callable[[Any], str]


# fmt: off
NUMBERS: list[Number] = [
    # data (counts only, never text)
    Number("data-threads", "data_stats.json", "threads", count),
    Number("data-conversations", "data_stats.json", "conversations", count),
    Number("data-approved", "data_stats.json", "approved", count),
    Number("data-identity", "data_stats.json", "identity", count),
    Number("data-formal", "data_stats.json", "formal", count),
    Number("data-synthetic", "data_stats.json", "synthetic", count),
    Number("data-train-examples", "data_stats.json", "train_examples", count),
    # model selection
    Number("base-model", "model_selection.json", "chosen", text),
    Number("model-size", "model_selection.json", "chosen_size", text),
    Number("base-license", "model_selection.json", "license", text),
    # training
    Number("train-gpu", "train_log.json", "gpu", text),
    Number("train-steps", "train_log.json", "steps", count),
    Number("train-epochs", "train_log.json", "epochs", fixed(1)),
    Number("train-time", "train_log.json", "wall_time_s", hours),
    Number("train-loss", "train_log.json", "final_train_loss", fixed(3)),
    Number("eval-loss", "train_log.json", "best_eval_loss", fixed(3)),
    Number("lora-r", "train_log.json", "config.lora_r", count),
    Number("lora-alpha", "train_log.json", "config.lora_alpha", count),
    Number("learning-rate", "train_log.json", "config.lr", text),
    Number("max-seq-len", "train_log.json", "config.max_seq_len", count),
    Number("effective-batch", "train_log.json", "config.effective_batch", count),
    # automatic eval
    Number("eval-prompts", "eval/scores.json", "n_prompts", count),
    Number("drift-base", "eval/scores.json", "base.drift_rate", percent),
    Number("drift-fym1", "eval/scores.json", "fym1.drift_rate", percent),
    Number("leak-fym1", "eval/scores.json", "fym1.leak_rate", percent),
    Number("ppl-base", "eval/scores.json", "base.perplexity", fixed(2)),
    Number("ppl-fym1", "eval/scores.json", "fym1.perplexity", fixed(2)),
    Number("belebele-base", "eval/scores.json", "base.belebele_acc", percent),
    Number("belebele-fym1", "eval/scores.json", "fym1.belebele_acc", percent),
    Number("chrf-mg-en-base", "eval/scores.json", "base.chrf_plt_eng", fixed(1)),
    Number("chrf-mg-en-fym1", "eval/scores.json", "fym1.chrf_plt_eng", fixed(1)),
    Number("chrf-en-mg-base", "eval/scores.json", "base.chrf_eng_plt", fixed(1)),
    Number("chrf-en-mg-fym1", "eval/scores.json", "fym1.chrf_eng_plt", fixed(1)),
    # LLM judge
    Number("judge-model", "eval/judge.json", "judge", text),
    Number("natural-base", "eval/judge.json", "base.naturalness", fixed(2)),
    Number("natural-fym1", "eval/judge.json", "fym1.naturalness", fixed(2)),
    Number("correct-base", "eval/judge.json", "base.correctness", fixed(2)),
    Number("correct-fym1", "eval/judge.json", "fym1.correctness", fixed(2)),
    # human arena (v0.2)
    Number("arena-votes", "eval/arena.json", "votes", count),
    Number("arena-raters", "eval/arena.json", "raters", count),
    Number("arena-win-rate", "eval/arena.json", "fym1_win_rate", percent),
    Number("arena-ci-low", "eval/arena.json", "ci_low", percent),
    Number("arena-ci-high", "eval/arena.json", "ci_high", percent),
    # vocabulary trimming
    Number("trim-gates", "shrink.json", "gates_passed", passed),
    Number("vocab-before", "shrink.json", "vocab_before", count),
    Number("vocab-after", "shrink.json", "vocab_after", count),
    Number("params-before", "shrink.json", "params_before", billions),
    Number("params-after", "shrink.json", "params_after", billions),
    Number("gguf-before", "shrink.json", "gguf_bytes_before", gigabytes),
    Number("gguf-after", "shrink.json", "gguf_bytes_after", gigabytes),
    Number("tps-before", "shrink.json", "tokens_per_sec_before", fixed(1)),
    Number("tps-after", "shrink.json", "tokens_per_sec_after", fixed(1)),
]
# fmt: on

# README results table: (row label, base macro name or None, fyM1 macro name)
README_ROWS: list[tuple[str, str | None, str]] = [
    ("Replies that drift to full French/English (lower is better)", "drift-base", "drift-fym1"),
    ("Naturalness, LLM judge, 1-5", "natural-base", "natural-fym1"),
    ("Correctness, LLM judge, 1-5", "correct-base", "correct-fym1"),
    ("Belebele Malagasy reading (accuracy)", "belebele-base", "belebele-fym1"),
    ("FLORES chrF++ Malagasy to English", "chrf-mg-en-base", "chrf-mg-en-fym1"),
    ("FLORES chrF++ English to Malagasy", "chrf-en-mg-base", "chrf-en-mg-fym1"),
    ("Blind human votes: fyM1 win rate (v0.2)", None, "arena-win-rate"),
]

SELECTION_COLUMNS = ["belebele_acc", "chrf_plt_eng", "chrf_eng_plt", "tokens_per_word", "sniff_natural"]
SELECTION_FORMATS: dict[str, Callable[[Any], str]] = {
    "belebele_acc": percent,
    "chrf_plt_eng": fixed(1),
    "chrf_eng_plt": fixed(1),
    "tokens_per_word": fixed(2),
    "sniff_natural": fixed(1),
}


def latex_name(name: str) -> str:
    """'drift-fym1' -> 'fyDriftFymOne'. Macro names may hold letters only."""
    parts = re.findall(r"[A-Za-z]+|\d", name)
    words = [_DIGIT_WORDS[int(p)] if p.isdigit() else p.capitalize() for p in parts]
    if not words:
        raise ValueError(f"no letters or digits in macro name {name!r}")
    return MACRO_PREFIX + "".join(words)


def latex_escape(s: str) -> str:
    return "".join(_LATEX_SPECIALS.get(c, c) for c in s)


def load_results(results_dir: Path) -> dict[str, Any]:
    """Every *.json under results_dir, keyed by its path relative to results_dir."""
    loaded = {}
    for path in sorted(results_dir.rglob("*.json")):
        loaded[path.relative_to(results_dir).as_posix()] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def lookup(data: Any, key: str) -> Any:
    """Follow a dotted key; None if any step is missing."""
    for part in key.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def resolve(results: dict[str, Any], numbers: list[Number] = NUMBERS) -> dict[str, str | None]:
    """Macro id -> formatted plain-text value, or None when not measured yet."""
    values: dict[str, str | None] = {}
    for number in numbers:
        raw = lookup(results.get(number.file), number.key)
        values[number.name] = None if raw is None else number.fmt(raw)
    return values


def selection_rows(results: dict[str, Any]) -> list[list[str | None]]:
    candidates = lookup(results.get("model_selection.json"), "candidates") or []
    rows = []
    for c in candidates:
        row: list[str | None] = [lookup(c, "model"), lookup(c, "license")]
        for col in SELECTION_COLUMNS:
            raw = lookup(c, col)
            row.append(None if raw is None else SELECTION_FORMATS[col](raw))
        rows.append(row)
    return rows


def _tex_cell(value: str | None) -> str:
    return r"\textbf{pending}" if value is None else latex_escape(value)


def render_tex(values: dict[str, str | None], rows: list[list[str | None]]) -> str:
    lines = ["% Generated by paper/numbers.py from results/. Do not edit by hand."]
    for name, value in values.items():
        lines.append(rf"\newcommand{{\{latex_name(name)}}}{{{_tex_cell(value)}}}")
    if rows:
        body = r" \\ ".join(" & ".join(_tex_cell(v) for v in row) for row in rows) + r" \\"
    else:
        width = 2 + len(SELECTION_COLUMNS)
        body = rf"\multicolumn{{{width}}}{{c}}{{\textbf{{pending}}}} \\"
    lines.append(rf"\newcommand{{\{MACRO_PREFIX}SelectionRows}}{{{body}}}")
    return "\n".join(lines) + "\n"


def render_readme_table(values: dict[str, str | None]) -> str:
    def cell(name: str | None) -> str:
        if name is None:
            return "-"
        return values[name] or PENDING

    lines = ["| Measure | Base model | fyM1 |", "|---|---|---|"]
    lines += [f"| {label} | {cell(base)} | {cell(ours)} |" for label, base, ours in README_ROWS]
    return "\n".join(lines)


def inject_readme(readme: str, table: str) -> str:
    start, end = readme.find(README_START), readme.find(README_END)
    if start == -1 or end < start:
        raise ValueError(f"README needs {README_START} ... {README_END} markers")
    return readme[: start + len(README_START)] + "\n" + table + "\n" + readme[end:]


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=here.parent / "results")
    parser.add_argument("--out", type=Path, default=here / "numbers.tex")
    parser.add_argument("--markdown", type=Path, nargs="*", default=[], help="also refresh the results table in these files")
    args = parser.parse_args()

    results = load_results(args.results)
    values = resolve(results)
    args.out.write_text(render_tex(values, selection_rows(results)), encoding="utf-8")
    table = render_readme_table(values)
    for path in args.markdown:
        path.write_text(inject_readme(path.read_text(encoding="utf-8"), table), encoding="utf-8")
    measured = sum(v is not None for v in values.values())
    print(f"{args.out}: {measured}/{len(values)} numbers measured, {len(values) - measured} pending")


if __name__ == "__main__":
    main()
