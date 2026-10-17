# fyM1

An open AI model that chats in everyday Malagasy. The way people really text — not like a government letter.

"fy" means sweet or pleasant in Malagasy. M is for Malagasy. 1 is the first version.

**[Try it](#try-it)** · [Paper](paper/fym1.pdf) · [Help us rate](https://huggingface.co/spaces/Fan1000-Prog/fyM1-arena)

> **Status: v0.1.** *pending* means not measured yet. Coming in v0.2: the full panel's blind human
> ratings, each model's answer to each test prompt, and the layer-pruning experiment.

## Goals

1. Start from the open model that is already best at Malagasy, and prove it with numbers.
2. Fine-tune it on a free Google Colab GPU so it chats in natural, everyday Malagasy.
3. Make it smaller by removing what Malagasy does not need, without making it worse.
4. Measure the base model against fyM1 honestly, and write it up in a short paper.
5. Share everything: the model, a free demo, the code, the notebooks and the paper.

## What this is

Open models that "speak Malagasy" often sound stiff and formal, or half-translated from French or English.
Ask "De ahoana?" and you get a speech back.

Real Malagasy texting is different:

- French words mix in all the time, even French verbs used the Malagasy way ("lancer-na").
- Spelling is short and phonetic ("zany" for "izany", "bé", "ahn").
- Replies are short, friendly, a bit teasing, with emoji.

fyM1 aims for that style. Mixing Malagasy and French is correct here. A reply only counts as "drifting"
if it is fully French or English with no Malagasy at all.

fyM1 is not a frontier model. The goal is simple: clearly and measurably better than its base model at
everyday Malagasy chat.

## How the model was chosen

We tested a short list of recent open models with 7 to 12 billion parameters. That size knows some
Malagasy already and still fits training on a free GPU.

Each one was scored on:

| Test | Why |
|---|---|
| Belebele Malagasy reading test | Does it understand Malagasy? |
| FLORES+ translation, both ways with English (chrF++) | Can it write correct Malagasy? |
| Tokens per Malagasy word | Fewer tokens = faster and cheaper to run. |
| 20 short everyday chats, rated by a native speaker | Does it already sound natural? |
| License | We prefer a permissive license when quality is close. |

The full table and the winner are in [`results/model_selection.md`](results/model_selection.md) and in
the paper. Winner: <!-- n:base-model -->pending<!-- /n --> (license: <!-- n:base-license -->pending<!-- /n -->).

## How it was trained

Every choice, and why we made it:

| Choice | Why |
|---|---|
| **Real chats** from one person's Messenger export (both sides of each conversation) are the main data. | That is where the everyday style lives. |
| Names, phone numbers, emails, links and handles are removed **on the owner's laptop** before anything leaves it. | Privacy. Raw chats never go to git, Colab or any public place. |
| The owner checks samples and approves the cleaned data. | A native speaker is the only real check for privacy and style. |
| Cleaned data lives only in a **private** dataset. It is never released. | It is still people's private conversations. The owner confirmed everyone in the chats agreed to this use. |
| A **small** slice of formal Malagasy (Wikipedia) is mixed in. | Keeps facts and spelling. Kept small so the style does not turn formal again. |
| Synthetic chats only if real chats are too few, made by an open model we run ourselves. | No paid APIs, no closed-model outputs, no private data sent to anyone. Budget: $0. |
| A few "who are you?" answers and polite refusals, written by the owner. | Meant to make it say it was made by Fan's lab and politely refuse harmful requests. Not tested as a safety feature. |
| **QLoRA**: the model stays frozen in 4-bit, we train small add-on weights (LoRA) on all linear layers. | It lets a 7-12B model train on one free 16 GB T4 GPU ([QLoRA paper](https://arxiv.org/abs/2305.14314)). |
| Learn from the assistant's replies only. | We want it to learn how to answer, not to copy the user. |
| Save a checkpoint to a private repo often, and resume from the latest one. | Free Colab sessions disconnect. No work is lost. |
| Stop early if the held-out loss gets worse. | Reduces overfitting. The leak check below is what tests memorising. |
| Test prompts are written fresh, not taken from the chats. | So the test prompts can be public and anyone can run them. The models' answers to them come out in v0.2. |
| **Leak check** before release: does the model repeat training sentences? If yes, it does not ship. | Privacy again. |
| **Vocabulary trimming**: drop tokens Malagasy, French and English never use. Ships only if it passes its checks. | A smaller download. The checks: held-out Malagasy text comes back unchanged, at least 95% of test answers stay identical (greedy decoding), and Belebele stays within 1 point. |
| No layer pruning yet. | Cutting real layers needs a lot of retraining to heal. Free GPU time cannot pay for that. Planned for v0.2 as an experiment. |
| Demo runs a 4-bit GGUF file on a free CPU. | Free for everyone. It is slow. |

Exact settings and the training log are in [`results/train_log.json`](results/train_log.json).

## Results

The numbers below come from files in [`results/`](results/), written by scripts in this repo. Nothing is
typed by hand.

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

Where each number comes from:

- Automatic scores: [`results/eval/scores.json`](results/eval/) and [`results/eval/judge.json`](results/eval/)
- Blind human votes: [`results/eval/arena.json`](results/eval/) — early votes only: <!-- n:arena-votes -->pending<!-- /n -->
  votes from <!-- n:arena-raters -->pending<!-- /n --> raters at release. The full panel's numbers come in v0.2.
- Model size before and after trimming: [`results/shrink.json`](results/shrink.json)
- Data counts (counts only, never text): [`results/data_stats.json`](results/data_stats.json)

The answers behind these scores are released in v0.2, after the blind human vote closes, so raters can't
see which model wrote what.

The LLM judge is a cheap extra signal. Native speakers are the real judges.

## Limits

- The style comes mostly from one person's chats. It sounds like everyday Antananarivo speech, not
  every region or dialect.
- The human comparison has only early votes so far. Until the full panel is in, the main evidence is automatic.
- It can get facts wrong and make things up. Do not use it for medical, legal or money advice.
- The free demo is slow. It runs on a free CPU.
- Text only. No voice, no images.

## Try it

- **Chat with it:** [fyM1-chat](https://huggingface.co/spaces/Fan1000-Prog/fyM1-chat) — free, no login.
  Try "Sali ahn" or "De ahoana?".
- **Download the model:** <!-- n:model-repo -->pending<!-- /n --> on [Hugging Face](https://huggingface.co/Fan1000-Prog).
  The GGUF file runs with llama.cpp and Ollama.
- **Read the paper:** [paper/fym1.pdf](paper/fym1.pdf)
- **Help us rate (native speakers):** [fyM1-arena](https://huggingface.co/spaces/Fan1000-Prog/fyM1-arena) —
  read two answers, pick the more natural one. No login.

## How to rerun

You need [uv](https://docs.astral.sh/uv/) and a free Google Colab account.

```bash
git clone https://github.com/Fan1000-prog/fyM1 && cd fyM1
uv sync
uv run pytest            # all code tests
```

Then, in order:

1. `notebooks/00_baseline.ipynb` on a Colab T4 — scores the candidate models.
2. `notebooks/01_train.ipynb` — trains fyM1. Rerun it after a disconnect; it resumes.
3. `notebooks/02_eval_export.ipynb` — answers the test prompts, scores both models, exports the GGUF.
4. `notebooks/03_trim.ipynb` — vocabulary trimming and its checks.
5. Rebuild the numbers and the paper:

   ```bash
   uv run python paper/make_numbers.py --markdown README.md MODEL_CARD.md
   cd paper && tectonic fym1.tex
   ```

The training chats are private, so you cannot retrain on the same data. You can rerun everything else,
and you can train on your own chats with `scripts/build_chats.py`.

v0.1 publishes only the total scores, not each model's answer to each prompt. Those answers come out in
v0.2, after the arena closes, so raters cannot see which answer came from which model. Until then you can
make your own answers with `notebooks/02_eval_export.ipynb`.

## License

Code: [Apache-2.0](LICENSE). The model follows the base model's license (see [MODEL_CARD.md](MODEL_CARD.md)).

Made by Fan's lab.
