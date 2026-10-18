"""Turn the measured outputs in results/ into the numbers the paper, README and model card cite.

Writes paper/numbers.tex (one \\newcommand per number). With --markdown it also refreshes each given
file: the results table between <!-- numbers:start --> and <!-- numbers:end -->, inline values
written as <!-- n:NAME -->value<!-- /n -->, and YAML front-matter lines ending in "# n:NAME".
A value not measured yet renders as "pending", so the paper always builds and no number is ever
typed by hand. Only the files listed below are read (never judge_key.json or judge_raw.json, which
would unblind the judge). A key missing from a file that exists is printed as a warning.
--strict (used at release) fails if any value is missing or the leak gate is not passed.

Usage: uv run python paper/make_numbers.py [--results results] [--out paper/numbers.tex]
                                           [--markdown README.md MODEL_CARD.md] [--strict]

Expected files (keys are dotted paths below):
  data_stats.json        threads, conversations, kept_messages, approved, identity, formal, synthetic,
                         train_examples
  data_gate.json         conversations_before_gate, kept, dropped, dropped_{long_number, contact,
                         too_many_unknown, too_many_fakes}, tokens_replaced, fakes_per_kept_{median, max}
  synth_stats.json       raw, kept, generator, generator_licence
  model_selection.json   chosen (HF repo id), chosen_size (e.g. "8B"), license, hf_license (HF license id),
                         candidates[{model, license, belebele_acc, belebele_chat_acc, chrf_plt_eng,
                         chrf_eng_plt, tokens_per_word, sec_per_step, greeting_gate.judge_mean}]
  greeting_gate.json     models.<model key>.{judge_mean, is_malagasy_rate, n}
  train_log.json         gpu, steps, epochs, wall_time_s, final_train_loss, best_eval_loss,
                         config.{lora_r, lora_alpha, lr, lr_scheduler, max_seq_len, effective_batch}
  eval/scores.json       n_prompts, {base,fym1}.{drift_rate, leak_rate, perplexity,
                         belebele_acc, chrf_plt_eng, chrf_eng_plt}
  eval/judge.json        judge, {base,fym1}.{naturalness, correctness}
  eval/leak_review.json  matches, approved_generic, rejected (counts of distinct matched 8-word runs)
  eval/arena.json        votes, raters, fym1_win_rate, ci_low, ci_high (early votes at v0.1)
  shrink.json            gates_passed, vocab_before, vocab_after, params_before, params_after,
                         gguf_bytes_before, gguf_bytes_after, tokens_per_sec_before, tokens_per_sec_after
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PENDING = "pending"
MACRO_PREFIX = "fy"
README_START = "<!-- numbers:start -->"
README_END = "<!-- numbers:end -->"
LEAK_FILE = "eval/leak_review.json"
INLINE = re.compile(r"<!-- n:([\w-]+) -->.*?<!-- /n -->")
YAML_LINE = re.compile(r"^(?P<head>\s*[\w-]+:[ \t]*)(?P<value>.*?)(?P<tail>[ \t]+#[ \t]*n:(?P<name>[\w-]+))$", re.M)

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


def _number(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"expected a number, got {v!r}")
    return float(v)


def count(v: Any) -> str:
    n = _number(v)
    if not n.is_integer():
        raise ValueError(f"expected a whole number, got {v!r}")
    return f"{int(n):,}"


def fixed(digits: int) -> Callable[[Any], str]:
    return lambda v: f"{_number(v):.{digits}f}"


def percent(v: Any) -> str:
    share = _number(v)
    if not 0 <= share <= 1:
        raise ValueError(f"expected a share between 0 and 1, got {v!r}")
    return f"{share * 100:.1f}%"


def billions(v: Any) -> str:
    return f"{_number(v) / 1e9:.2f}B"


def gigabytes(v: Any) -> str:
    return f"{_number(v) / 1e9:.2f} GB"


def hours(v: Any) -> str:
    return f"{_number(v) / 3600:.1f} h"


def text(v: Any) -> str:
    if not isinstance(v, str):
        raise TypeError(f"expected text, got {v!r}")
    return v


def passed(v: Any) -> str:
    if not isinstance(v, bool):
        raise TypeError(f"expected a bool, got {v!r}")
    return "passed" if v else "failed"


def model_repo(size: Any) -> str:
    return f"Fan1000-Prog/fy-m1-{text(size)}"


@dataclass(frozen=True)
class Pick:
    """A key step into a list: the first item whose `field` equals `value` (e.g. a candidate by model id)."""

    field: str
    value: str

    def __str__(self) -> str:
        return f"[{self.field}={self.value}]"


# A dotted path ("base.drift_rate"), or a tuple of steps when a step holds a dot or picks from a list.
Key = str | tuple[str | Pick, ...]


@dataclass(frozen=True)
class Number:
    name: str  # human-readable id, e.g. "drift-fym1"; becomes the macro \fyDriftFymOne
    file: str  # path under results/
    key: Key  # path inside the JSON file
    fmt: Callable[[Any], str]
    required: bool = True  # False: a comparison value that may stay pending at release (--strict ignores it)


QWEN = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"
GEMMA4 = "google/gemma-4-12B-it"
GEMMA3 = "google/gemma-3-12b-it"
# Model keys in greeting_gate.json (results/baseline/ file stems).
GREETING_KEYS = {QWEN: "afriqueqwen3.5-9b-instruct", GEMMA4: "gemma-4-12b-it", GEMMA3: "gemma-3-12b-it"}
# Short names for macro ids: qwen -> \fySelQwen..., gemma4 -> \fySelGemmaFour...
SHORT = {QWEN: "qwen", GEMMA4: "gemma4", GEMMA3: "gemma3"}


def candidate_numbers() -> list[Number]:
    """Per-candidate values the paper's selection section cites. Only the base's are required at release."""
    columns = [
        ("belebele", "belebele_acc", percent),
        ("belebele-chat", "belebele_chat_acc", percent),
        ("chrf-mg-en", "chrf_plt_eng", fixed(1)),
        ("chrf-en-mg", "chrf_eng_plt", fixed(1)),
        ("sec-step", "sec_per_step", fixed(1)),
    ]
    out = [
        Number(f"sel-{SHORT[m]}-{label}", "model_selection.json", ("candidates", Pick("model", m), key), fmt, m == QWEN)
        for m in SHORT
        for label, key, fmt in columns
    ]
    out += [
        Number(f"greet-{SHORT[m]}-{label}", "greeting_gate.json", ("models", GREETING_KEYS[m], key), fmt)
        for m in SHORT
        for label, key, fmt in [("mean", "judge_mean", fixed(2)), ("malagasy", "is_malagasy_rate", percent)]
    ]
    out.append(Number("greet-answers", "greeting_gate.json", ("models", GREETING_KEYS[QWEN], "n"), count))
    return out


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
    Number("data-kept-messages", "data_stats.json", "kept_messages", count),
    # strict automatic data gate (counts only)
    Number("gate-before", "data_gate.json", "conversations_before_gate", count),
    Number("gate-kept", "data_gate.json", "kept", count),
    Number("gate-dropped", "data_gate.json", "dropped", count),
    Number("gate-dropped-number", "data_gate.json", "dropped_long_number", count),
    Number("gate-dropped-contact", "data_gate.json", "dropped_contact", count),
    Number("gate-dropped-unknown", "data_gate.json", "dropped_too_many_unknown", count),
    Number("gate-dropped-fakes", "data_gate.json", "dropped_too_many_fakes", count),
    Number("gate-replaced", "data_gate.json", "tokens_replaced", count),
    Number("gate-fakes-median", "data_gate.json", "fakes_per_kept_median", count),
    Number("gate-fakes-max", "data_gate.json", "fakes_per_kept_max", count),
    # synthetic chats (counts and run settings only)
    Number("synth-raw", "synth_stats.json", "raw", count),
    Number("synth-kept", "synth_stats.json", "kept", count),
    Number("synth-generator", "synth_stats.json", "generator", text),
    Number("synth-licence", "synth_stats.json", "generator_licence", text),
    # model selection
    Number("base-model", "model_selection.json", "chosen", text),
    Number("model-size", "model_selection.json", "chosen_size", text),
    Number("model-repo", "model_selection.json", "chosen_size", model_repo),
    Number("base-license", "model_selection.json", "license", text),
    Number("base-license-hf", "model_selection.json", "hf_license", text),
    # training
    Number("train-gpu", "train_log.json", "gpu", text),
    Number("train-steps", "train_log.json", "steps", count),
    Number("train-epochs", "train_log.json", "epochs", fixed(1)),
    Number("train-time", "train_log.json", "wall_time_s", hours),
    Number("train-loss", "train_log.json", "final_train_loss", fixed(3)),
    Number("eval-loss", "train_log.json", "best_eval_loss", fixed(3)),
    Number("lora-r", "train_log.json", "config.lora_r", count),
    Number("lora-alpha", "train_log.json", "config.lora_alpha", count),
    Number("learning-rate", "train_log.json", "config.lr", lambda v: f"{_number(v):g}"),
    Number("lr-scheduler", "train_log.json", "config.lr_scheduler", text),
    Number("max-seq-len", "train_log.json", "config.max_seq_len", count),
    Number("effective-batch", "train_log.json", "config.effective_batch", count),
    # automatic eval
    Number("eval-prompts", "eval/scores.json", "n_prompts", count),
    Number("drift-base", "eval/scores.json", "base.drift_rate", percent),
    Number("drift-fym1", "eval/scores.json", "fym1.drift_rate", percent),
    Number("leak-fym1", "eval/scores.json", "fym1.leak_rate", percent),
    Number("leak-matches", LEAK_FILE, "matches", count),
    Number("leak-generic", LEAK_FILE, "approved_generic", count),
    Number("leak-rejected", LEAK_FILE, "rejected", count),
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
    # human arena: v0.1 reports the early votes in at release; the full panel's result is v0.2
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
    *candidate_numbers(),
]
# fmt: on

# README results table: (row label, base macro name or None, fy-m1 macro name)
README_ROWS: list[tuple[str, str | None, str]] = [
    ("Replies that drift to full French/English (lower is better)", "drift-base", "drift-fym1"),
    ("Naturalness, LLM judge, 1-5", "natural-base", "natural-fym1"),
    ("Correctness, LLM judge, 1-5", "correct-base", "correct-fym1"),
    ("Belebele Malagasy reading (accuracy)", "belebele-base", "belebele-fym1"),
    ("FLORES chrF++ Malagasy to English", "chrf-mg-en-base", "chrf-mg-en-fym1"),
    ("FLORES chrF++ English to Malagasy", "chrf-en-mg-base", "chrf-en-mg-fym1"),
    ("Blind human votes so far (early votes): fy-m1 win rate", None, "arena-win-rate"),
]

SELECTION_FILE = "model_selection.json"
RESULT_FILES = sorted({n.file for n in NUMBERS} | {SELECTION_FILE})
SELECTION_COLUMNS = [
    "belebele_acc", "belebele_chat_acc", "chrf_plt_eng", "chrf_eng_plt", "tokens_per_word", "greeting_gate.judge_mean",
]  # fmt: skip
SELECTION_FORMATS: dict[str, Callable[[Any], str]] = {
    "belebele_acc": percent,
    "belebele_chat_acc": percent,
    "chrf_plt_eng": fixed(1),
    "chrf_eng_plt": fixed(1),
    "tokens_per_word": fixed(2),
    "greeting_gate.judge_mean": fixed(2),
}


@dataclass(frozen=True)
class Resolved:
    values: dict[str, str | None]  # number name -> formatted plain text, None when not measured
    rows: list[list[str | None]]  # model-selection table
    missing_keys: list[str]  # "file: key" for keys absent from a file that exists
    missing: list[str]  # "file: key" for every required value not measured (what --strict fails on)


def leak_gate_problems(results: dict[str, Any]) -> list[str]:
    """Why the leak review blocks release: every matched run must be reviewed, and none rejected."""
    review = results.get(LEAK_FILE)
    counts = {key: lookup(review, key) for key in ("matches", "approved_generic", "rejected")}
    if any(v is None for v in counts.values()):
        return []  # not measured yet; reported as missing values
    matches, generic, rejected = (_number(counts[k]) for k in ("matches", "approved_generic", "rejected"))
    problems = []
    if generic + rejected != matches:
        problems.append(f"{LEAK_FILE}: approved_generic + rejected ({generic + rejected:g}) != matches ({matches:g})")
    if rejected != 0:
        problems.append(f"{LEAK_FILE}: {rejected:g} matched run(s) rejected; release is blocked")
    return problems


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
    """The known result files that exist, keyed by their path under results_dir. Nothing else is read."""
    loaded = {}
    for name in RESULT_FILES:
        path = results_dir / name
        if path.is_file():
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def key_text(key: Key) -> str:
    """("candidates", Pick("model", "m"), "x") -> "candidates[model=m].x"."""
    if isinstance(key, str):
        return key
    return "".join(str(p) if isinstance(p, Pick) else ("." if i else "") + p for i, p in enumerate(key))


def lookup(data: Any, key: Key) -> Any:
    """Follow a key; None if any step is missing."""
    for part in key.split(".") if isinstance(key, str) else key:
        if isinstance(part, Pick):
            items = data if isinstance(data, list) else []
            data = next((i for i in items if isinstance(i, dict) and i.get(part.field) == part.value), None)
        elif isinstance(data, dict):
            data = data.get(part)
        else:
            return None
        if data is None:
            return None
    return data


def resolve(results: dict[str, Any], numbers: list[Number] = NUMBERS) -> Resolved:
    values: dict[str, str | None] = {}
    missing_keys: list[str] = []
    missing: list[str] = []

    def note_missing(file: str, key: str, required: bool = True) -> None:
        where = f"{file}: {key}"
        if file in results:
            missing_keys.append(where)
        if required:
            missing.append(where)

    for number in numbers:
        raw = lookup(results.get(number.file), number.key)
        if raw is None:
            note_missing(number.file, key_text(number.key), number.required)
            values[number.name] = None
        else:
            values[number.name] = number.fmt(raw)

    rows = []
    selection = results.get(SELECTION_FILE)
    candidates = lookup(selection, "candidates")
    if candidates is None:
        note_missing(SELECTION_FILE, "candidates")
    for i, candidate in enumerate(candidates or []):
        # Comparison rows may stay pending at release; the chosen base's row must be complete.
        is_base = lookup(candidate, "model") == lookup(selection, "chosen")
        row: list[str | None] = []
        for col in ["model", "license", *SELECTION_COLUMNS]:
            raw = lookup(candidate, col)
            if not isinstance(raw, (int, float, str)) or isinstance(raw, bool):
                raw = None  # e.g. greeting_gate "pending" on a dropped row is a string, not a dict
            if raw is None:
                note_missing(SELECTION_FILE, f"candidates[{i}].{col}", is_base)
            row.append(None if raw is None else SELECTION_FORMATS.get(col, text)(raw))
        rows.append(row)
    return Resolved(values, rows, missing_keys, missing)


def _tex_cell(value: str | None) -> str:
    return rf"\textbf{{{PENDING}}}" if value is None else latex_escape(value)


def render_tex(resolved: Resolved) -> str:
    lines = ["% Generated by paper/make_numbers.py from results/. Do not edit by hand."]
    for name, value in resolved.values.items():
        lines.append(rf"\newcommand{{\{latex_name(name)}}}{{{_tex_cell(value)}}}")
    if resolved.rows:
        body = r" \\ ".join(" & ".join(_tex_cell(v) for v in row) for row in resolved.rows) + r" \\"
    else:
        width = 2 + len(SELECTION_COLUMNS)
        body = rf"\multicolumn{{{width}}}{{c}}{{\textbf{{{PENDING}}}}} \\"
    lines.append(rf"\newcommand{{\{MACRO_PREFIX}SelectionRows}}{{{body}}}")
    return "\n".join(lines) + "\n"


def _md_value(values: dict[str, str | None], name: str) -> str:
    if name not in values:
        raise ValueError(f"unknown number {name!r}")
    return values[name] or PENDING


def render_readme_table(values: dict[str, str | None]) -> str:
    def cell(name: str | None) -> str:
        return "-" if name is None else _md_value(values, name)

    lines = ["| Measure | Base model | fy-m1 |", "|---|---|---|"]
    lines += [f"| {label} | {cell(base)} | {cell(ours)} |" for label, base, ours in README_ROWS]
    return "\n".join(lines)


def inject_readme(readme: str, values: dict[str, str | None]) -> str:
    """Refresh the results table, inline values and YAML values in one markdown file."""
    start, end = readme.find(README_START), readme.find(README_END)
    if start == -1 or end < start:
        raise ValueError(f"file needs {README_START} ... {README_END} markers")
    out = readme[: start + len(README_START)] + "\n" + render_readme_table(values) + "\n" + readme[end:]
    out = INLINE.sub(lambda m: f"<!-- n:{m[1]} -->{_md_value(values, m[1])}<!-- /n -->", out)
    return YAML_LINE.sub(lambda m: m["head"] + _md_value(values, m["name"]) + m["tail"], out)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=here.parent / "results")
    parser.add_argument("--out", type=Path, default=here / "numbers.tex")
    parser.add_argument("--markdown", type=Path, nargs="*", default=[], help="also refresh these files")
    parser.add_argument("--strict", action="store_true", help="fail if any number is missing or the leak gate fails (release)")
    args = parser.parse_args()

    results = load_results(args.results)
    resolved = resolve(results)
    for where in resolved.missing_keys:
        print(f"warning: missing key {where}", file=sys.stderr)
    if args.strict:
        problems = [f"not measured yet: {where}" for where in resolved.missing] + leak_gate_problems(results)
        if problems:
            sys.exit("--strict failed:\n  " + "\n  ".join(problems))

    args.out.write_text(render_tex(resolved), encoding="utf-8")
    for path in args.markdown:
        path.write_text(inject_readme(path.read_text(encoding="utf-8"), resolved.values), encoding="utf-8")
    measured = sum(v is not None for v in resolved.values.values())
    print(f"{args.out}: {measured}/{len(resolved.values)} numbers measured, {len(resolved.values) - measured} pending")


if __name__ == "__main__":
    main()
