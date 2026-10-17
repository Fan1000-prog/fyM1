---
# TODO(final): set license, base_model and model-index from results/model_selection.json.
language:
  - mg
  - fr
license: other
base_model: pending
library_name: transformers
pipeline_tag: text-generation
tags:
  - malagasy
  - chat
  - qlora
  - gguf
---

# fyM1

fyM1 chats in everyday Malagasy: short, friendly, mixing in French, with texting spelling and emoji.
It is a QLoRA fine-tune of *pending* (base model), made by Fan's lab.

**Status: v0.1.** Some results are pending. They will land in v0.2.

- Demo: [fyM1-chat](https://huggingface.co/spaces/Fan1000-Prog/fyM1-chat)
- Code, notebooks and paper: [github.com/Fan1000-prog/fyM1](https://github.com/Fan1000-prog/fyM1)
- Help rate it: [fyM1-arena](https://huggingface.co/spaces/Fan1000-Prog/fyM1-arena)

## Files

| File | What |
|---|---|
| *pending* | Merged 4-bit weights |
| *pending* | LoRA adapter |
| *pending* | GGUF Q4_K_M, for llama.cpp / Ollama / CPU |

## Intended use

- Casual chat in everyday Malagasy.
- Research on low-resource, informal-register language models.

## Not for

- Medical, legal, financial or safety-critical advice.
- Facts you need to trust. It can make things up.
- Formal Malagasy writing. It is tuned for casual chat.

## How to use

```bash
# TODO(final): exact file name of the GGUF.
llama-cli -hf Fan1000-Prog/fyM1-<size> -p "Sali ahn"
```

## System prompt

The model was trained with this system prompt. Use it for best results.

- Malagasy: *pending (owner-approved wording)*
- English gloss: "You are a helpful Malagasy assistant trained by Fan's lab."

## Training data

Described here, not shared. It contains private conversations.

- Real everyday chats from a native speaker's Messenger export. Everyone in the chats agreed to this use.
  Names, phone numbers, emails, links and handles were removed on the owner's laptop. A native speaker
  approved samples. The cleaned data is kept in a private dataset and is not released.
- A small slice of Malagasy Wikipedia, for facts and spelling.
- A few identity answers and polite refusals, written by the owner.
- No outputs from closed-API models.

Counts are in [`results/data_stats.json`](https://github.com/Fan1000-prog/fyM1/blob/main/results/data_stats.json).

## Training procedure

QLoRA (4-bit NF4 base, LoRA on all linear layers) with Unsloth on a free Colab T4 GPU. Loss on assistant
turns only. Full settings and the loss curve:
[`results/train_log.json`](https://github.com/Fan1000-prog/fyM1/blob/main/results/train_log.json).

## Evaluation

All numbers are written by scripts in the GitHub repo. *pending* means not measured yet.

<!-- numbers:start -->
| Measure | Base model | fyM1 |
|---|---|---|
| Replies that drift to full French/English (lower is better) | pending | pending |
| Naturalness, LLM judge, 1-5 | pending | pending |
| Correctness, LLM judge, 1-5 | pending | pending |
| Belebele Malagasy reading (accuracy) | pending | pending |
| FLORES chrF++ Malagasy to English | pending | pending |
| FLORES chrF++ English to Malagasy | pending | pending |
| Blind human votes: fyM1 win rate (v0.2) | - | pending |
<!-- numbers:end -->

Leak gate: the model does not ship if it repeats sentences from the training chats
(`leak_rate` in `results/eval/scores.json`).

## Example chats

*pending* — taken only from the public test prompts in `eval/prompts.jsonl`, never from private chats.

## Limits

- The style comes mostly from one person's chats: everyday Antananarivo-style speech, not every dialect.
- The blind human comparison is still running (v0.2).
- It can be wrong, and it can invent things.
- It has only a basic safety check. It is not a safety-tested product.

## License

Follows the base model's license: *pending*. Code in the GitHub repo is Apache-2.0.

## Citation

```bibtex
@misc{fym1,
  title  = {fyM1: Teaching an Open Model to Chat in Everyday Malagasy},
  author = {{Fan's lab}},
  year   = {2026},
  url    = {https://github.com/Fan1000-prog/fyM1}
}
```
