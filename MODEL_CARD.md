---
# Values tagged "# n:NAME" are filled by paper/make_numbers.py from results/.
language:
  - mg
  - fr
license: pending # n:base-license-hf
base_model: pending # n:base-model
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
It is a QLoRA fine-tune of <!-- n:base-model -->pending<!-- /n -->, made by Fan's lab.

**Status: v0.1.** *pending* means not measured yet. Coming in v0.2: the full panel's blind human
ratings and each model's answer to each test prompt.

- Demo: [fyM1-chat](https://huggingface.co/spaces/Fan1000-Prog/fyM1-chat)
- Code, notebooks and paper: [github.com/Fan1000-prog/fyM1](https://github.com/Fan1000-prog/fyM1)
- Help rate it: [fyM1-arena](https://huggingface.co/spaces/Fan1000-Prog/fyM1-arena)

## Files

| File | What |
|---|---|
| TODO(final) | Merged 4-bit weights |
| TODO(final) | LoRA adapter |
| TODO(final) | GGUF Q4_K_M, for llama.cpp / Ollama / CPU |

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
llama-cli -hf Fan1000-Prog/fyM1-<size>:Q4_K_M -p "Sali ahn"
```

## System prompt

The model was trained with this system prompt. Use it for best results.

- Malagasy: TODO(final): the owner-approved wording
- English gloss: "You are a helpful Malagasy assistant trained by Fan's lab."

## Training data

Described here, not shared. It contains private conversations.

- Real everyday chats from one person's Messenger export (both sides of each conversation). The owner
  confirmed everyone in the chats agreed to this use.
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

All numbers are written by scripts in the GitHub repo.

<!-- numbers:start -->
| Measure | Base model | fyM1 |
|---|---|---|
| Replies that drift to full French/English (lower is better) | pending | pending |
| Naturalness, LLM judge, 1-5 | pending | pending |
| Correctness, LLM judge, 1-5 | pending | pending |
| Belebele Malagasy reading (accuracy) | pending | pending |
| FLORES chrF++ Malagasy to English | pending | pending |
| FLORES chrF++ English to Malagasy | pending | pending |
| Blind human votes so far (early votes): fyM1 win rate | - | pending |
<!-- numbers:end -->

Leak gate: any answer that shares a run of 8 words with the training chats blocks release.
The owner sees only the matched 8 words, never the chat, and it ships only if every match is a generic phrase.
At release: <!-- n:leak-matches -->pending<!-- /n --> matches, <!-- n:leak-generic -->pending<!-- /n --> approved as generic,
<!-- n:leak-rejected -->pending<!-- /n --> rejected (`results/eval/leak_review.json`).

## Example chats

TODO(final): taken only from the public test prompts in `eval/prompts.jsonl`, never from private chats.

## Limits

- The style comes mostly from one person's chats: everyday Antananarivo-style speech, not every dialect.
- The blind human comparison has only early votes (<!-- n:arena-votes -->pending<!-- /n --> votes from
  <!-- n:arena-raters -->pending<!-- /n --> raters). The full panel's numbers come in v0.2.
- It can be wrong, and it can invent things.
- It has only a basic safety check. It is not a safety-tested product.

## License

Follows the base model's license: <!-- n:base-license -->pending<!-- /n -->. Code in the GitHub repo is Apache-2.0.

## Citation

```bibtex
@misc{fym1,
  title  = {fyM1: Teaching an Open Model to Chat in Everyday Malagasy},
  author = {{Fan's lab}},
  year   = {2026},
  url    = {https://github.com/Fan1000-prog/fyM1}
}
```
