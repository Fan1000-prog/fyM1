# fyM1 project update

Snapshot: 2026-10-18, ~02:30 UTC, `main` at acc6265. Check `gh pr list` for anything newer. Read with the spec
(`docs/superpowers/specs/2026-10-17-resaka-design.md`) and the plan (`docs/superpowers/plans/2026-10-17-fym1-v0.1.md`).
This file says where the project stands and how to pick it up; the spec and plan say what it is and how it is built.

## 0. Start here (new session or new team)

1. Read this file, then `AGENTS.md`, then spec §7–§8 and the plan's "Changes from the plan".
2. Read the private operator handoff on the owner's laptop (not in this repo; it names accounts and machines):
   `~/fyM1-handoff/LEAD-HANDOFF.md` (start at §13 "CURRENT STATE") and `~/fyM1-handoff/TRAINER-HANDOFF.md`.
3. Set up the team: a **lead** (owns calls, merges, run order; talks to the owner through the main session) and,
   as needed, a **trainer** (the only agent that drives Kaggle/Colab), **data** (cleaner + export), **eval**
   (Task 5), **publish** (Spaces), **shrink** (03_trim), **writer** (paper/README/model card).
4. Check what's running before starting anything: the Kaggle kernel status and the result JSONs on the Hub
   (commands in `docs/kaggle_operator.md` and LEAD-HANDOFF §13). Never run the same model on Kaggle and Colab.
5. Rules that bite: one branch + worktree per change, fresh reviewer per PR, at most 2 fix rounds then ask the
   owner, squash-merge only on safe-to-merge + green CI, clean up merged branches, no AI attribution, no deletions
   or architecture changes without the owner, never print or pass the HF/Kaggle tokens, never hand-type a number.

## 1. Where we are

Tooling is built and merged. Baseline tests are half done; no model is picked or trained yet.

- **Blocking training:** the owner's Messenger export (waiting on Meta). `~/malagasy-data/raw/` doesn't exist.
- **GPU:** Kaggle works (owner's account; HF token attached to the baseline kernel). Colab's free quota ran out at
  01:15 UTC and is the backup only.
- **Ship date:** when the export is in (owner chose this over a Wikipedia-only model that would still talk formally).

`main`: 2,077 tests pass, 1 skipped, 14 expected failures with default groups (`uv run pytest -q`); ~2,140 pass
with `uv sync --all-groups`. The 14 strict xfails pin known anonymiser limits.

## 2. Done

| Plan task | What | PRs |
|---|---|---|
| 1 Repos + scaffold | Private GitHub repo, package layout, `.gitignore` tests, private HF repos `fyM1-chats`, `fyM1-ckpt`, `fyM1-votes` | #1 |
| 2 Data | Messenger parser, anonymiser, segmenter, review sheet, `build_chats.py`, `build_formal.py` | #3 |
| 2 Data | Anonymiser leak fixes: a mention's fake never echoes the real name (also inside or split across a handle); glued `+261`/`00261`/`(034)` phones caught | #16 |
| 2 Data | Phone-vs-date follow-up: local numbers with a year-like group, price pairs kept (see the known issue in §5) | #24 |
| 2 Data | Uploaded to private `fyM1-chats`: 1,500 Wikipedia chats, 16 owner-written identity/refusal chats, system prompt | — |
| 2 Data | Dry run on a made-up export with the new cleaner: 0 phone-like numbers left, owner name absent, planted outsider name listed on the leftover sheet | — |
| 3 Eval | `is_malagasy`, `drift_rate`, `leak_rate`, 150 owner-approved prompts, judge rubric | #4, #15 |
| 4 Train infra | Checkpoint resume, smoke notebook, token-safe Colab flow | #2 |
| 4 Baseline + training | `00_baseline`, `01_train`, per-run folders that refuse a resume on changed settings or code, training time across disconnects, own early stop, `sync_code_to_hub.py` | #13 |
| 4 Kaggle | Runtime helper (Kaggle before Colab), `kaggle_push.py`, notebooks run on either, one GPU pinned, fresh code folder per run, notebook builder in the repo | #18, #19, #20 |
| 4 Gemma 3 load | Load `google/gemma-3-12b-it` quantised on load (no silent swap to the oversized copy), out-of-memory guard records `"oom"`, never a number | #21 |
| 4 Results | AfriqueQwen3.5-9B-Instruct baseline + selection table | #23 |
| 5 Scoring | `score.py`: blind judge batch, judge means, `scores.json`, leak matches, owner leak review, strict leak gate | #6, #10, #11, #15 |
| 6 Spaces | Chat + arena apps, votes, owner-approved Malagasy text, `fy-m-1` names | #7, #9, #12 |
| 7 Shrink | Vocabulary trim (notebook not run yet) | #8 |
| 8 Writing | Paper, README, model card skeletons; this update, spec §8, plan status | #5, #17 |

## 3. Decisions made outside the spec and plan

Recorded in spec §8 and the plan's "Changes from the plan" too.

- **Name:** public `fy-m-1`; private repos keep `fyM1-*`; package stays `fym1`. GitHub repo recreated clean as
  `Fan1000-prog/fy-m-1` at release (owner pre-approved deleting and recreating it).
- **Privacy backstops**, kept whatever the cleaner does: owner's 100-sample review, leftover-names blocklist,
  count-only post-run phone check, strict 8-word leak gate. Raw answers and the arena A/B key stay private until v0.2.
- **Anonymiser trade-offs (owner):** glued `tel34…`/`telO34…` and no-0 layouts like `34 123 45 67` stay known
  limits; no narrower rule for `tel341234567`. PR #24 merged "as is" on the owner's word (see §5 known issue).
- **Model pick:** delegated to the lead; same tests for every candidate (plain-text 0-shot Belebele = `belebele_acc`,
  FLORES+ both ways, tokens per word, 15-step QLoRA smoke on a T4) plus the owner's 1–5 rating of 20 greeting answers.
- **Compute:** $0. Kaggle is the main GPU (~30 h/week, 12 h sessions), Colab the backup. On a T4, Unsloth trains
  Gemma 3 and Gemma 4 in float32 (slow); the 9B instruct and AfriqueQwen-8B ran in 16-bit.
- **Training code** reaches the GPU only by syncing clean `origin/main` to the Hub; never during a training run.
- **Judge disclosure:** "Claude Opus 5.5 (claude-opus-5-5), blind, run as judging agents in Claude Code; rubric in
  `src/fym1/eval/judge_rubric.md`." Methods disclosure, not attribution.

## 4. Baseline numbers so far

From `results/model_selection.json` on `main` (written by `scripts/build_model_selection.py`). Gemma 3 has no results file yet; its license is from the script's license table.

| | AfriqueQwen3.5-9B-Instruct | Gemma 4 12B | AfriqueQwen-8B (base) | Gemma 3 12B |
|---|---|---|---|---|
| Belebele, plain text 0-shot | 0.667 | pending | pending | pending |
| Belebele, chat format | 0.742 | 0.750 | not valid (base) | pending |
| FLORES+ chrF++ eng→mg / mg→eng | 49.73 / 56.17 | pending | pending | pending |
| Tokens per Malagasy word | 2.826 | 2.792 | 2.952 | pending |
| Smoke: s/step, peak GPU | 83.3, 9.8 GB | pending (re-run) | 74.3, 7.79 GB | pending |
| Smoke loss | 1.00 → 0.73 (0.77 last) | pending | 1.13 → 0.70 | pending |
| License | CC-BY-4.0 | Apache-2.0 | CC-BY-4.0 | Gemma Terms of Use |

Caveat for the 9B: its answer to "Sali ahn" came back in Hausa and several greetings echo the prompt, so the
owner's rating matters. Gemma 4's earlier unscripted smoke (≈126 s/step, loss rising at lr 2e-4) was removed.

## 5. Open now

| Item | Who | State |
|---|---|---|
| **Known issue: #24 breaks dates followed by more numbers** | owner decides | "15/03/2026 1 500 Ar" → "15/<TEL> Ar"; date ranges broken (~1,080 per 97k everyday lines; a plain date or "le 15/03/2026 à 14h" is fine; not a privacy leak). Options: (a) revert #24 to the #16 cleaner, which brings back the rare leak #24 closed ("038 27 2000/5", "+261 038 27 2000/5" kept whole; the post-run phone count would flag them); (b) fix by masking whole real dates in the sentence before phone matching. Must be settled before `build_chats` runs on the real export |
| Kaggle baseline run | trainer | kernel version 2 running since ~01:48 UTC on Gemma 3 (Hub code 6d13c89); results land on the Hub one section at a time |
| PR #22 baseline guards (draft) | trainer | parked until Gemma 3's result shows `env.loaded_from == "google/gemma-3-12b-it"`; then sync with main, review, merge |
| Messenger export | owner | waiting on Meta |
| 20-greeting ratings per model | owner | after all four candidates have answers |

Run order: Gemma 3 (full) → Gemma 4 (plain-text Belebele, FLORES+, scripted 15-step smoke) → AfriqueQwen-8B
(plain-text Belebele, FLORES+). New result JSONs go in a PR on a fresh branch.

## 6. Steps to finish

1. **Settle the #24 date issue** (owner's call; revert or fix), with its own PR and review.
2. **Finish the baseline** on Kaggle (run order above); merge result PRs; finish or drop PR #22.
3. **Owner rates** the 20 greeting answers per model; **lead picks** the base model and sends the table + reason.
4. **Export lands** in `~/malagasy-data/raw/`: `data` runs `build_chats.py` from `main`, then the post-run checks
   (counts only to the lead).
5. **Owner reviews** the 100-sample sheet and ticks real names on the leftover-names list; rerun with the blocklist.
6. **Upload** approved chats to private `fyM1-chats`; `results/data_stats.json` (counts only). Synthetic chats only
   if fewer than 2,000 approved conversations.
7. **Sync code** from clean `main`; set `MODEL_ID`/`RUN_ID` in `01_train` (no defaults); **train** on Kaggle,
   supervised; log to `results/train_log.json`.
8. **Eval** (Task 5): 150 answers from base and fy-m-1, Belebele + FLORES+ rerun, perplexity, blind judge,
   `scores.json`; merge LoRA, 4-bit + GGUF Q4_K_M.
9. **Leak gate**: owner reviews every matched 8-word run; owner checks the 20 identity/safety answers.
10. **Trim** (Task 7): run `03_trim`; ship trimmed GGUF only if all three gates pass.
11. **Deploy Spaces** (Task 6); Playwright smoke test.
12. **Write-up** (Task 8) from `results/` via `paper/make_numbers.py --strict`.
13. **Release** (Task 9): recreate the GitHub repo clean, flip public (private repos stay private), check every
    README link, send the owner the arena link and a 2-line Malagasy message.

v0.2 after release: panel arena votes, layer pruning, longer training, raw answers once the arena closes.
