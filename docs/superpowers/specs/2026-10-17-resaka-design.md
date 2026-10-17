# Resaka — an everyday-Malagasy chat model

Design, 2026-10-17. Owner: Fandresena (native Malagasy speaker). Project name "Resaka" ("conversation") is a placeholder — builder's call unless the owner renames it.

## 1. Intent — why this exists

Open models that "speak Malagasy" speak it like a government letter: formal, stiff, often half-translated from French or English. Malagasy people don't talk like that. The goal is an open model that answers "Sali ahn" or "De ahoana?" the way a friend would.

Two audiences:

- **Non-technical Malagasy speakers.** The win is the moment of surprise: "it talks like a normal person." They interact through a free web demo.
- **Technical people** (ML folks, reviewers, would-be contributors). They want to see how it was built, rerun it, check the numbers, and help. They read the GitHub README, the notebook and the paper.

## 2. Goals, non-goals, anti-goals

**Goals**
1. Pick an open 7–12B model that is already the best available at Malagasy, with the choice justified by measurement.
2. QLoRA fine-tune it on a free Colab T4 so it chats in natural, everyday (colloquial) Malagasy.
3. Make it smaller by removing what Malagasy doesn't need (see §5.4), without measurable quality loss.
4. Measure base vs. fine-tuned honestly and write it up as a short research paper.
5. Publish: model on Hugging Face (with a free interactive demo), code + notebook + paper on a public GitHub repo with a clear README.

**Non-goals**
- Full fluency or parity with frontier models. Target is "clearly and measurably better than the base model at everyday Malagasy."
- Regional dialects beyond common Antananarivo-style everyday speech (future work).
- Speech/audio, images.
- A paid or fast GPU demo.

**Anti-goals (failure even if it "works")**
- Fluent but formal/stiff Malagasy. That is the problem we exist to fix.
- Leaking anyone's private details from the chat data (names, numbers, places, anything identifying).
- Overclaiming in the paper or README. Every claim must trace to a measured number.
- Breaking the base model's license terms.

## 3. Constraints

- **Compute:** Google Colab free tier, T4 (16 GB VRAM). Sessions disconnect after a few hours and daily GPU quota is limited. Training must checkpoint often and resume from the last checkpoint after any disconnect.
- **Precision:** 4-bit (QLoRA NF4) for training; a 4-bit build is what people download/run.
- **Model size:** at least 7B parameters; must fit QLoRA training on a T4.
- **Privacy (hard rule):**
  - Raw Messenger exports stay on the owner's laptop (`~/malagasy-data/raw/`). They never go into git, Colab, or any public place.
  - A local script anonymises chats (names, phone numbers, emails, links, addresses, place names that identify people, account handles) before anything leaves the laptop.
  - The cleaned training data lives only in a **private** HF dataset repo (so Colab can load it). It is never made public.
  - The owner approves samples of the cleaned data before training.
  - The owner confirmed everyone in the chats consents to their messages being used.
- **Licensing:** the base model's license governs the released model's license and naming; prefer a permissive license (e.g. Apache-2.0) when quality is close. Code in the GitHub repo: Apache-2.0.
- **Owner's standing rules:** every change on its own branch + worktree, PR, independent reviewer agent, merge only when clean. No AI attribution anywhere (commits, README, paper, model card). Architectural choices and deletions need the owner's OK.

## 4. Decided (by the owner)

- Two personas as above; prompts are short everyday chat ("Sali ahn", "De ahoana?").
- Native-speaker owner checks samples and supplies real conversations from their Messenger export (export in progress — both the standard "Download your information" JSON export and the end-to-end-encrypted "secure storage" download).
- Both sides of the conversations may be used (consent given).
- GitHub account: `Fan1000-prog` (already authenticated on this machine via `gh`). Hugging Face account: `Fan1000-prog` (not yet authenticated on this machine).
- Free Colab T4 for training. QLoRA, 4-bit.
- Publish on HF for people to interact with; public GitHub repo with README covering goals, what this is, how/why the model was chosen and trained, CTA "try it" linking to the model and the paper.
- Work split across a team of agents.

## 5. Builder's calls (made by me, the designer — owner may override)

### 5.1 Base model selection
- Shortlist the newest open-weight models in the 7–12B range as of Oct 2026 (Gemma, Llama, Qwen, and any newer multilingual family; Aya-style models if they cover Malagasy). Verify what exists now — don't rely on memory.
- Score each candidate on: Malagasy comprehension (Belebele `plt_Latn`), translation (FLORES-200 `plt_Latn` chrF++ both directions), tokenizer efficiency (tokens per Malagasy word), a quick 20-prompt everyday-chat sniff test the owner rates, T4 QLoRA fit, and license.
- Pick the winner by these numbers; record the table in the paper. The owner sees the table before training starts (it's an architectural choice).

### 5.2 Training data mix
1. **Real chats** (gold): from the owner's Messenger export, parsed, mojibake-fixed (Meta exports UTF-8 as escaped Latin-1), anonymised, filtered to Malagasy/Malagasy-French mixed threads, segmented into short multi-turn conversations.
2. **Synthetic everyday chats** modelled on the real ones, to add variety. Owner checks a sample. Generator choice is an open question — see §7.
3. **A small slice of formal Malagasy** (e.g. Malagasy Wikipedia, news) so facts and spelling survive. Kept small so it doesn't pull the style back to formal.
- Format: the base model's own chat template; multi-turn.
- Hold out a test set (§5.5) before any training.

### 5.3 Training
- Unsloth + TRL on Colab T4; QLoRA NF4, LoRA on all linear layers; sensible defaults (r≈16, lr≈2e-4, cosine, ~1–3 epochs), tuned from eval loss.
- Checkpoints + LoRA adapter pushed to a **private** HF repo every N steps so a fresh Colab session resumes where the last one died.
- Owner runs the notebook in their browser (or the team drives Colab via Playwright after the owner logs into Google once — builder picks whichever is reliable).

### 5.4 Making it smaller ("remove non-Malagasy parameters")
- **Ship-eligible: vocabulary trimming.** Drop tokenizer entries (and matching embedding / output rows) that never appear in Malagasy + French + English-loanword text, keeping special and byte-fallback tokens. Must pass: tokenizer round-trip on held-out Malagasy text, and identical eval scores within noise. Report size before/after.
- **Experiment only: layer pruning.** Remove the most redundant layers (e.g. by layer-to-layer similarity), heal with a short QLoRA pass, measure. Ship only if it doesn't hurt the human preference result; otherwise report it as a finding.
- Reason: cutting real model weights without heavy retraining damages quality, and free T4 time can't pay for that retraining.

### 5.5 Measurement (what the paper reports)
- **Primary:** blind side-by-side test on 100–200 held-out everyday prompts. Base vs. fine-tuned answers, order shuffled, labels hidden. Owner (plus any other native speakers they recruit) picks which sounds more like a real person and which is more correct. Report win rate with a confidence interval.
- **Secondary (automatic):** LLM-judge naturalness/colloquiality scores on the same prompts (judge prompt + rubric published); "answers in Malagasy" rate (language ID — not drifting to French/English); held-out chat perplexity.
- **Regression checks:** Belebele `plt_Latn` and FLORES-200 chrF++ — the model must not lose much general Malagasy ability.
- **Efficiency:** parameter count, download size, tokens/sec on the demo hardware, before vs. after trimming.
- All numbers produced by scripts in the repo; the paper only cites numbers from those outputs.

### 5.6 Publishing
- **HF model repo** (public): merged 4-bit weights + LoRA adapter + GGUF (Q4_K_M) for CPU/llama.cpp/Ollama; model card with data description (no private content), limits, eval table, example chats.
- **HF Space** (public, free CPU tier): Gradio chat running the GGUF. Slow but free. Pre-filled example prompts ("Sali ahn", "De ahoana?").
- **GitHub repo** `Fan1000-prog/resaka` (public): code, notebook, eval scripts, paper source + PDF, README (goals, what this is, how and why the model was chosen and trained — every decision with its reason, CTA "Try it" → Space, links to the model and paper). Private data paths are git-ignored.
- **Paper:** short (4–6 page) LaTeX PDF: motivation, model selection, data (described, not shared), method, trimming/pruning, results, limits, ethics/privacy.

### 5.7 Team split (agents)
1. **Scout:** shortlist + baseline eval of candidate models → selection table.
2. **Repos:** create GitHub + HF repos, scaffold, .gitignore for private data, license.
3. **Data:** Messenger parser + anonymiser + sample review sheet for the owner; formal-slice fetcher; synthetic generator (once §7 is answered); dataset upload to private HF.
4. **Eval:** held-out test set, blind A/B tool for the owner, automatic metrics, regression benchmarks.
5. **Training:** Colab notebook with resume; supervises runs; hyperparameter tuning from eval loss.
6. **Shrink:** vocabulary trimming + layer-pruning experiment.
7. **Publish:** merge, quantise, GGUF, model card, Space.
8. **Writer:** paper + README from the measured outputs.

Order: Scout + Repos + Data parser + Eval set can start in parallel. Training waits for the owner's data approval and model pick. Shrink and Publish wait for a trained model. Writer finishes last but drafts throughout.

## 6. Left to the builder
Exact hyperparameters, chunking of conversations, anonymiser technique, LLM-judge choice, Space UI details, paper template, repo layout, and how agents coordinate — as long as the constraints in §3 hold.

## 7. Answers from the owner (after builder check)

### 7.1 What "everyday Malagasy" looks like (owner, with a real chat sample)
Real sample: "Lancement inona?" / "Dia mahay aika rakia nareo ka!" / "Mbola miasa zany ialahy io" / "Mety e. Lancer o tsara bé😂".
- **French mixing is normal and wanted**, including French verbs used Malagasy-style ("Lancer o").
- **Texting spelling is wanted**: shortened/phonetic forms ("zany" for "izany", "bé", "ahn"), slang ("ialahy"), emoji.
- Short replies, friendly, a bit of teasing/humour.
- Therefore: the language metric is "**not drifting into full French/English sentences**" — mixed Malagasy-French is a pass. The judge rubric and A/B rating guide reward this style. The formal slice is for facts, not for spelling style.

### 7.2 Synthetic generator
Owner chose the recommendation: a large **open-weight model with a permissive license, run by us** (e.g. on Colab), few-shot prompted with cleaned real chats. Owner checks a sample. No closed-API outputs in training data; cleaned chats never go to a third-party API. **Budget: $0.**

### 7.3 Raters
Owner will recruit a few other native speakers. They need **one easy link**: a small public HF Space ("arena") showing a prompt and two unlabelled answers in random order; raters pick the more natural one (and flag wrong ones). No login. Votes saved to a private HF dataset. Short instructions in Malagasy.

### 7.4 Timeline: ship today (2026-10-17)
"Today" means **v0.1**, honestly labelled:
- In today: model pick from quick baseline tests; data pipeline; one or two QLoRA runs; automatic metrics + owner's own blind ratings; vocabulary trimming only if it passes its checks; model + demo Space + arena Space + GitHub repo + paper v0.1.
- After today (v0.2): the panel's blind A/B results (raters vote over the following days), layer-pruning experiment, longer training. The paper and README say clearly which results are in and which are pending.

### 7.5 Name and identity
- Name: **fyM1** — "fy" (Malagasy: sweet, pleasant) + M(alagasy) + 1. HF model repo `Fan1000-prog/fyM1-<size>`, GitHub `Fan1000-prog/fyM1`. (Supersedes the "Resaka" placeholder.)
- System prompt (in Malagasy, with an English gloss in the model card): a helpful Malagasy assistant trained by **Fan's lab**. Training data includes a small set of "who are you?" answers consistent with this.

### 7.6 Calls made by the designer after the builder check
- Data volume unknown until the export arrives; the mix adapts — the fewer real chats, the more checked synthetic chats.
- Test prompts are written fresh (owner-approved), not taken from private chats, so the test set is public and rerunnable.
- **Leak gate:** before release, test whether the model reproduces sentences from the real chats; if it does, it doesn't ship.
- **Safety:** small set of polite refusals + identity answers in training; a basic safety check before release; limits stated in the model card.
