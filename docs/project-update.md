# fyM1 project update

Snapshot: 2026-10-17, ~23:30 UTC. Read with the spec (`docs/superpowers/specs/2026-10-17-resaka-design.md`)
and the plan (`docs/superpowers/plans/2026-10-17-fym1-v0.1.md`). This file says where the project stands; the
spec and plan say what it is and how it is built.

## 1. Where we are

The tooling is built and merged. No model is picked or trained yet. Two things block training:

1. **The Messenger export is not on the laptop.** `~/malagasy-data/raw/` does not exist, so there are no real
   chats yet. Without them the model can't learn everyday style.
2. **Colab's free GPU limit was hit** at 23:12 UTC after ~2.7 h of baseline tests. Retries every ~45 min.

Ship date moved from 2026-10-17 to **when the export is in** (owner: tomorrow is fine). The owner chose this
over shipping a model trained only on Wikipedia + identity lines, which would still talk formally.

`main`: 587 tests pass, 1 skipped, 9 expected failures (known anonymiser limits, strict xfail).

## 2. Done

| Plan task | What | PRs |
|---|---|---|
| 1 Repos + scaffold | Private GitHub repo, package layout, `.gitignore` tests, private HF repos `fyM1-chats`, `fyM1-ckpt`, `fyM1-votes` | #1 |
| 2 Data | Messenger parser, anonymiser, segmenter, review sheet, `build_chats.py`, `build_formal.py` | #3 (+ #16 open) |
| 2 Data | Uploaded to private `fyM1-chats`: 1,500 Wikipedia chats (`formal.jsonl`), 16 identity/refusal chats written by the owner, system prompt. Private both before and after upload | — |
| 2 Data | Dry run of cleaner + post-run checks on a made-up export: all planted phones counted, planted names listed | — |
| 3 Eval | `is_malagasy`, `drift_rate`, `leak_rate`, 150 owner-approved test prompts, judge rubric; greeting words "akory", "aby" | #4, #15 |
| 4 Train infra | `latest_checkpoint` resume, Colab smoke notebook, token-safe Colab operator flow, one-shot local token server | #2 |
| 5 Scoring | `score.py`: blind judge batch, judge means, `scores.json`, leak matches, owner leak review, strict leak gate | #6, #10, #11, #15 |
| 6 Spaces | Chat + arena apps, `make_vote`, vote tally, owner-approved Malagasy text, `fy-m-1` Space names | #7, #9, #12 |
| 7 Shrink | Vocabulary trim that keeps BPE merges valid (notebook not run yet) | #8 |
| 8 Writing | Paper, README and model card skeletons (numbers pending) | #5 |

## 3. Decisions made outside the spec and plan

Recorded in the spec (§8) and plan ("Changes from the plan") as well.

**Naming and identity**
- Public name is **`fy-m-1`**: model `Fan1000-Prog/fy-m-1-<size>`, Spaces `fy-m-1-chat` / `fy-m-1-arena`,
  GitHub repo recreated as `Fan1000-prog/fy-m-1` at release. Private repos keep `fyM1-*`; package stays `fym1`.
- Identity/refusal lines (16) are written by the owner in their own words.

**Data and privacy**
- PR #3 merged by owner override with two known anonymiser leaks open (mention fake equal to the real name;
  glued `+261` phones). Owner then reversed course: PR #16 closes both, plus the forms its reviewers found
  (names inside or split across handles, glued bracket/`00261`/`+2610`/no-leading-0 phones).
- Backstops kept even with the fix: owner's 100-sample review, a "leftover names" blocklist the owner ticks,
  a count-only post-run check for phone-like digit runs, and the strict leak gate.
- Raw per-prompt answers and the arena A/B key stay private until the arena closes (v0.2). v0.1 publishes
  aggregates only. The owner's own arena votes count as "early votes".
- Release: the GitHub repo is deleted and recreated from one clean snapshot (owner pre-approved), with no old
  history, screenshots or browser files, then made public.

**Model selection**
- Lead picks the base model (owner delegated). Candidates: Gemma 4 12B, AfriqueQwen-8B, Gemma 3 12B.
- One Belebele protocol for all: plain-text 0-shot, no chat template. The chat-template score is a second
  column, empty for base models. (AfriqueQwen is a base model; its chat-format score of 0.236 was a test error.)
- Smoke training tests cut to 15 steps for the remaining models to save GPU time.
- Baseline runs may use the trainer's branch code (each result records its code sha). Real training runs only on
  code synced from clean `origin/main` (`scripts/sync_code_to_hub.py` refuses otherwise).
- No paid compute. Owner suggested Kaggle (~30 free GPU h/week) as a fallback; lead decides.

**Training details**
- `CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}` shared by train, eval and trim.
- Every assistant turn gets the same empty thinking block the model sees at answer time, so training matches use.
- Own tested early-stop rule (the Hugging Face callback breaks after a Colab restart).
- Gemma 4 on a T4 trains in float32 (Unsloth fallback): if picked, lr 1e-4 or a longer warmup.
- Saving 4-bit weights during the baseline is off (filled Colab RAM and hung).

**Process**
- One extra fix round for PR #2 beyond the usual two, approved by the owner.
- Judge disclosure: "Claude Opus 5.5 (claude-opus-5-5), blind, run as judging agents in Claude Code; rubric in
  `src/fym1/eval/judge_rubric.md`." Methods disclosure, not attribution.

## 4. Baseline numbers so far

| | Gemma 4 12B | AfriqueQwen-8B | Gemma 3 12B |
|---|---|---|---|
| Belebele, chat format, 0-shot | 0.750 | not valid (base model) | pending |
| Belebele, plain text, 0-shot | pending | pending | pending |
| FLORES+ chrF++ | pending | pending | pending |
| Tokens per Malagasy word | 2.79 | 2.95 | pending |
| Seconds per training step (8×2048 tokens) | 126 (float32) | 74 | pending |
| Peak GPU memory | 9.18 GB | 7.79 GB | pending |
| Smoke loss | best at step 9, then rising | smooth, 1.13 → 0.70 | pending |
| 20 greetings | fluent, formal "AI assistant" | unusable (no chat tuning) | pending |

AfriqueQwen-8B has no chat-tuned version. The nearest is
`McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1` (cc-by-4.0, different size and architecture, Unsloth support
unconfirmed). Lead decides whether to test it as a fourth candidate.

Source: `results/baseline/*.json` and `results/model_selection.json` on the `baseline` branch (PR #13).

## 5. Open now

| Item | Owner | State |
|---|---|---|
| PR #16 anonymiser leak fix | main | 2 fix rounds done; final review running |
| PR #13 baseline + 01_train notebook | trainer | draft until the baseline finishes |
| Messenger export | owner | not on the laptop |
| 20-greeting ratings per model | owner | after all three models have answers |
| Kaggle fallback | lead | owner suggested; lead's call |
| Old `data-pipeline` branch + `~/fyM1-data` | main | delete after #16 merges (owner OK) |

## 6. Steps to finish

1. **Merge PR #16**, then delete `data-pipeline` (remote, local, `~/fyM1-data`) and `leak-fix`. `data` re-runs
   its dry run on the new `main`.
2. **Finish the baseline** when a GPU is free (Colab, or Kaggle if lead switches): Gemma 3 in full; plain-text
   Belebele and FLORES+ for all three. Owner rates the 20 greetings per model.
3. **Lead picks the base model** and sends the owner the table and the reason. Merge PR #13.
4. **Messenger export lands** in `~/malagasy-data/raw/`. `data` runs `build_chats.py` locally, then the post-run
   checks; counts only go to lead.
5. **Owner reviews** the 100-sample sheet and ticks real names on the leftover-names list (blocklist).
6. **Upload** approved chats to private `fyM1-chats`; write `results/data_stats.json` (counts only). Decide on
   synthetic chats (only if fewer than 2,000 approved conversations).
7. **Sync code** to `fyM1-ckpt/code/` from clean `origin/main`; **train** with `01_train` (~3 h on a T4),
   supervised by `trainer`; log to `results/train_log.json`.
8. **Eval** (Task 5): answers for 150 prompts from base and fy-m-1, Belebele + FLORES+ rerun, perplexity,
   blind judge, `scores.json`; merge LoRA, 4-bit + GGUF Q4_K_M.
9. **Leak gate**: owner reviews each matched 8-word run; release only if every run is generic. Owner checks the
   20 identity/safety answers.
10. **Trim** (Task 7): run `03_trim`; ship trimmed GGUF only if all three gates pass.
11. **Deploy Spaces** (Task 6) and smoke-test them with Playwright.
12. **Write-up** (Task 8): fill paper, README, model card from `results/` via `make_numbers.py --strict`.
13. **Release** (Task 9): gates green, recreate the GitHub repo clean, flip public (repo, model, Spaces; private
    repos stay private), check every README link, send the owner the arena link and a 2-line Malagasy message.

v0.2 after release: panel arena votes, layer pruning, longer training, raw answers once the arena closes.
