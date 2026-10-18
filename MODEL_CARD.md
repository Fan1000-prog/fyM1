---
# Values tagged "# n:NAME" are filled by paper/make_numbers.py from results/.
language:
  - mg
  - fr
license: cc-by-4.0 # n:base-license-hf
base_model: McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1 # n:base-model
library_name: transformers
pipeline_tag: text-generation
tags:
  - malagasy
  - chat
  - qlora
  - gguf
---

# fy-m1

fy-m1 chats in everyday Malagasy: short, friendly, mixing in French, with texting spelling and emoji.
It is a QLoRA fine-tune of <!-- n:base-model -->McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1<!-- /n -->,
by Fandresena Raharinaivo.

**Status: v0.1.** *pending* means not measured yet. Coming in v0.2: the full panel's blind human
ratings and each model's answer to each test prompt.

- Demo: [fy-m1-chat](https://huggingface.co/spaces/Fan1000-Prog/fy-m1-chat)
- Code, notebooks and paper: [github.com/Fan1000-prog/fy-m1](https://github.com/Fan1000-prog/fy-m1)
- Help rate it: [fy-m1-arena](https://huggingface.co/spaces/Fan1000-Prog/fy-m1-arena)

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
# TODO(final): exact repo id (model-repo number) and GGUF file name.
llama-cli -hf Fan1000-Prog/fy-m1-<size>:Q4_K_M -p "Salama e!"
```

## System prompt

The model was trained with this system prompt. Use it for best results.

- Malagasy: TODO(final): the owner-approved wording
- English gloss: "You are fy-m1, a helpful Malagasy assistant trained by Fan's lab. Reply in everyday Malagasy."

## Training data

Described here, not shared. Part of it is private conversations.

- **Real chats:** one family's Messenger export. Everyone in it agreed to this use, and each person's side
  is used as the assistant in turn. Names, nicknames, phone numbers, emails, links and handles were
  replaced on the owner's laptop. No one read the cleaned text: a strict automatic gate kept
  <!-- n:gate-kept -->192<!-- /n --> of <!-- n:gate-before -->260<!-- /n --> conversations and replaced
  every word it did not know with a fake name. The cleaned data is in a private dataset and is not released.
- **Synthetic chats:** <!-- n:synth-kept -->pending<!-- /n --> everyday chats written by
  <!-- n:synth-generator -->pending<!-- /n --> (licence <!-- n:synth-licence -->pending<!-- /n -->), run by us,
  from cleaned examples. Only permissively licensed generators are allowed.
- **Formal slice:** <!-- n:data-formal -->pending<!-- /n --> short chats from Malagasy Wikipedia, for facts and
  spelling.
- **Identity chats:** <!-- n:data-identity -->pending<!-- /n --> "who are you?" answers and polite refusals,
  written by the owner.
- No outputs from closed-API models. Gemma 3 was only a comparison baseline; none of its outputs are used.

Counts only, never text:
[`results/data_stats.json`](https://github.com/Fan1000-prog/fy-m1/blob/main/results/data_stats.json),
[`results/data_gate.json`](https://github.com/Fan1000-prog/fy-m1/blob/main/results/data_gate.json).

## Training procedure

QLoRA (4-bit NF4 base, LoRA on all linear layers, rank <!-- n:lora-r -->pending<!-- /n -->) with Unsloth
on a free Kaggle T4 GPU (Colab as backup). Loss on assistant turns only, thinking turned off.
<!-- n:train-steps -->pending<!-- /n --> steps, <!-- n:train-time -->pending<!-- /n --> of training. Full settings
and the loss curve:
[`results/train_log.json`](https://github.com/Fan1000-prog/fy-m1/blob/main/results/train_log.json).

## Evaluation

All numbers are written by scripts in the GitHub repo.

<!-- numbers:start -->
| Measure | Base model | fy-m1 |
|---|---|---|
| Replies that drift to full French/English (lower is better) | pending | pending |
| Naturalness, LLM judge, 1-5 | pending | pending |
| Correctness, LLM judge, 1-5 | pending | pending |
| Belebele Malagasy reading (accuracy) | pending | pending |
| FLORES chrF++ Malagasy to English | pending | pending |
| FLORES chrF++ English to Malagasy | pending | pending |
| Blind human votes so far (early votes): fy-m1 win rate | - | pending |
<!-- numbers:end -->

- Test set: <!-- n:eval-prompts -->pending<!-- /n --> fresh everyday prompts, public in `eval/prompts.jsonl`.
- LLM judge (methods disclosure): <!-- n:judge-model -->pending<!-- /n -->; rubric in
  `src/fym1/eval/judge_rubric.md`. A cheap signal, not a substitute for native speakers.
- Leak gate: any fy-m1 answer that shares a run of 8 words with the real training chats blocks release.
  Only the matched words are checked, never the chat. At release: <!-- n:leak-matches -->pending<!-- /n -->
  distinct matched runs, <!-- n:leak-generic -->pending<!-- /n --> generic and
  <!-- n:leak-rejected -->pending<!-- /n --> rejected (`results/eval/leak_review.json`).

## Example chats

TODO(final): taken only from the public test prompts in `eval/prompts.jsonl`, never from private chats.

## Limits

- The real chats come from one family: their everyday speech, not every dialect. Most of the data is synthetic.
- The anonymiser has known limits (see `docs/anonymisation.md`); the strict gate and the leak gate are the backstops.
- The blind human comparison has only early votes (<!-- n:arena-votes -->pending<!-- /n --> votes from
  <!-- n:arena-raters -->pending<!-- /n --> raters). The full panel's numbers come in v0.2.
- It can be wrong, and it can invent things.
- It has only a basic safety check. It is not a safety-tested product.

## License

CC-BY-4.0, the same as the base model (<!-- n:base-license -->CC-BY-4.0<!-- /n -->). Credit fy-m1 and
the base model, AfriqueQwen by McGill-NLP. Code in the GitHub repo is Apache-2.0. Data credits (wordfreq,
Malagasy Wikipedia, HPLT 2.0, INSEE first names, Belebele, FLORES+) are in the GitHub README.

The formal chats are built from Malagasy Wikipedia text, which is CC-BY-SA 4.0. We credit it here. The
model stays CC-BY-4.0, but if you reuse that training text, its ShareAlike terms apply to it.

## Citation

```bibtex
@misc{fym1,
  title  = {fy-m1: Teaching an Open Model to Chat in Everyday Malagasy},
  author = {Fandresena Raharinaivo},
  year   = {2026},
  url    = {https://github.com/Fan1000-prog/fy-m1}
}
```
