"""Pure helpers for the base-model baseline (notebooks/00_baseline.ipynb)."""

import re
from collections.abc import Callable, Iterable

import sacrebleu

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def belebele_prompt(row: dict) -> str:
    """0-shot Belebele prompt: passage, question, numbered options, answer-with-a-number instruction."""
    options = "\n".join(f"{i}. {row[f'mc_answer{i}']}" for i in range(1, 5))
    return (
        f"{row['flores_passage']}\n\n"
        f"Question: {row['question']}\n\n"
        f"{options}\n\n"
        "Answer with the number of the correct option only (1, 2, 3 or 4)."
    )


def accuracy(preds: list[int], golds: list[str | int]) -> float:
    """Share of predictions equal to the gold answer number (Belebele stores golds as strings)."""
    if len(preds) != len(golds) or not preds:
        raise ValueError(f"need equal, non-empty lists, got {len(preds)} preds and {len(golds)} golds")
    return sum(int(p) == int(g) for p, g in zip(preds, golds)) / len(preds)


def translation_prompt(text: str, src: str, tgt: str) -> str:
    return f"Translate the following sentence from {src} to {tgt}. Reply with the translation only.\n\n{text}"


def first_line(text: str) -> str:
    """First non-blank line, stripped; models often add notes after the translation."""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def sample_sentences(texts: Iterable[str], n: int, min_words: int = 5, max_words: int = 60) -> list[str]:
    """First `n` sentences with min_words..max_words words, in input order."""
    out = []
    for text in texts:
        for sentence in _SENTENCE_END.split(text.replace("\n", " ")):
            sentence = sentence.strip()
            if min_words <= len(sentence.split()) <= max_words:
                out.append(sentence)
                if len(out) == n:
                    return out
    return out


def tokens_per_word(sentences: list[str], count_tokens: Callable[[str], int]) -> float:
    """Total tokens over total whitespace-separated words."""
    words = sum(len(s.split()) for s in sentences)
    if not words:
        raise ValueError("no words to count")
    return sum(count_tokens(s) for s in sentences) / words


def chrf_pp(hyps: list[str], refs: list[str]) -> float:
    """Corpus chrF++ (chrF with word bigrams), 0-100."""
    return sacrebleu.CHRF(word_order=2).corpus_score(hyps, [refs]).score


def selection_row(result: dict, license: str) -> dict:
    """One `candidates[]` row of results/model_selection.json from a results/baseline/<slug>.json file.

    Sections the run did not produce (skipped model, FLORES pending) stay None.
    `sniff_natural` is filled later from the owner's ratings.
    """
    flores = result.get("flores", {})
    smoke = result.get("qlora_smoke", {})
    return {
        "model": result["model"]["hub_id"],
        "license": license,
        "belebele_acc": result.get("belebele", {}).get("acc"),
        "chrf_plt_eng": flores.get("plt-eng_chrf++"),
        "chrf_eng_plt": flores.get("eng-plt_chrf++"),
        "tokens_per_word": result.get("tokens_per_word", {}).get("value"),
        "sniff_natural": None,
        "peak_vram_gb": smoke.get("peak_vram_gb"),
        "sec_per_step": smoke.get("sec_per_step"),
        "qlora_fits": smoke.get("ok"),
    }


def belebele_completion_prompt(row: dict) -> str:
    """Plain-text 0-shot prompt for base models (no chat template); score the next token over " A"-" D"."""
    options = "\n".join(f"{letter}: {row[f'mc_answer{i}']}" for i, letter in enumerate("ABCD", start=1))
    return f"P: {row['flores_passage']}\nQ: {row['question']}\n{options}\nAnswer:"
