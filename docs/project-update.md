# fyM1 project update

Snapshot: 2026-10-19 (UTC), `main` at 4d86fc2. Check `gh pr list` for anything newer. Read with the spec
(`docs/superpowers/specs/2026-10-17-resaka-design.md`) and the plan (`docs/superpowers/plans/2026-10-17-fym1-v0.1.md`).
This file says where the project stands and how to pick it up; the spec and plan say what it is and how it is built.
§0 is the latest state. §1–§6 are older snapshots; where they disagree with §0, §0 wins.

## 0. Start here (new session or new team)

1. Read this file, then `AGENTS.md`, then spec §7–§8 and the plan's "Changes from the plan".
2. Read the private operator handoff on the owner's laptop (not in this repo; it names accounts and machines):
   the lead handoff (§16 "new run" holds; §24 is the latest state and overrides older parts), the trainer
   handoff, and the log of decisions the lead made for the owner.
3. Set up the team: a **lead** (owns calls, merges, run order; talks to the owner through the main session) and,
   as needed, a **trainer** (the only agent that drives Kaggle/Colab), **data** (cleaner + export), **eval**
   (Task 5), **publish** (Spaces), **shrink** (03_trim), **writer** (paper/README/model card). Keep at most 3–4
   agents running at once.
4. Check what's running before starting anything: the Kaggle kernel status and the result JSONs on the Hub
   (commands in `docs/kaggle_operator.md`). Never run the same model on Kaggle and Colab, and never push a new
   Kaggle version over a running one.
5. Rules that bite: one branch + worktree per change, fresh reviewer per PR. Blocked review → a fresh
   ML-engineer agent fixes → a fresh reviewer, repeated until safe-to-merge + green CI (no round cap); the same
   blocker 3 rounds in a row → the lead decides and records it. Before spawning a fixer, check
   `git -C <worktree> status` so no worktree gets two writers. Squash-merge only on safe-to-merge + green CI,
   clean up merged branches, no AI attribution, no deletions or architecture changes without the owner, never
   print or pass the HF/Kaggle tokens, never hand-type a number.
6. Owner-review gates (sample sheet, leftover names, leak gate, identity check, greeting ratings) are strict
   automatic gates run by fresh agents; list what they auto-decided for the owner. Any agent near 25% context
   writes a handoff and is replaced by a fresh one (the first lead handed over to lead-2 this way). Public name is
   `fy-m1` (spec §8.6).

### 0.1 Where things stand (owner takes over here)

In one line: r1 is trained and measured, it speaks good everyday Malagasy but plays a family member instead of
answering; r2 (a new data mix to fix that) finished training and is not measured yet. The owner decided to ship
r2 as fy-m1 v1 (§0.2); §0.3 is the plan.

**r1** (`RUN_ID fym1-9b-a100-r1`, adapter `checkpoint-500`: `final_adapter` in `results/eval/eval_run.json`,
`best_step` in `results/train_log.json`).

- What went wrong: it answers like one of the family in the group chat, not like an assistant.
- Why: `build_chats --all-senders` put each family member in the assistant seat in turn. The synthetic chats
  copied that shape. Names from the fake-name pool sat in assistant turns (fake names, so not a privacy leak).
- r1 vs base, same 150 prompts (r1 sampling was the old temp 0.7 / top_p 0.9, `sampling` in `eval_run.json`):

| | r1 | base | file |
|---|---|---|---|
| Answer rate (judge) | 0.753 | 0.867 | `results/eval/judge.json` `answer_rate` |
| Naturalness, 1–5 | 4.09 | 2.96 | `results/eval/judge.json` |
| Correctness, 1–5 | 3.06 | 3.35 | `results/eval/judge.json` |
| Identity check passed | 24/34 | 19/34 | `results/eval/identity_check.json`, `identity_check_base.json` |
| Drift rate | 0.000 | 0.027 | `results/eval/scores.json` |
| Leak gate (8-word matches) | 0 matches: pass | — | `results/eval/leak_review.json` |

- The identity gate fails for both: under the 90% bar (`threshold`), and both make foreign self-claims
  (`foreign_identity`: r1 2, base 5).

**r2 data mix** (`results/mix_r2.json`, PR #72). 1,345 training rows (`mix.training_rows`):

- synthetic chats with the persona cut: 933 (`synth.kept_chats`)
- identity chats: 16 × 5 = 80 (`identity.chats`, `identity.repeat`)
- real reply pairs: 63, capped at 10% (`real.cap`, `real.max_share`); real share is 0.0468, so about 5%
- Aya: 269, capped at 20% (`aya.cap`, `aya.max_share`). Source `CohereLabs/aya_dataset`, Apache-2.0, Malagasy
  original human annotations only: 11,672 rows, 10,856 pass the gates (`aya.original_annotations`,
  `aya.rows_passed`); 8 dropped because they match eval prompts (`dropped_rows_by_reason.eval_prompt`).
- Left out: the formal Wikipedia set.
- On the Hub: private `fyM1-chats`, `DATA_REVISION 4e5dde2fa8034e055937cf05a05acdb54c5d2454`, `DATA_DIR mix-r2/`
  (from the upload, not a `results/` file).
- At train time one identity chat fell into the eval split: 75 identity rows train, 1,315 train / 26 eval (from
  the r2 run log, not in `results/` yet). Why 1,315 + 26 = 1,341, not 1,345: the split is over 1,281 unique rows
  (933 + 16 + 63 + 269); eval takes 26 of them (1 identity chat, counted once), the 15 identity chats left in
  train are repeated x5, so train = 1,255 + 60 = 1,315.

**Decoding defaults** (PR #71, `src/fym1/generation.py`): temp 0.6, repetition_penalty 1.1, top_p 0.9, top_k 50,
3-exchange chat history. Used by `05_try`, `02_eval_export` and the chat Space.

**r2 run: finished** (run facts from private `fyM1-ckpt` `fym1-9b-a100-r2/train_log.json`, not in `results/` yet;
the r2 eval PR copies them in with `scripts/fetch_train_log.py`):

- `RUN_ID fym1-9b-a100-r2`, Colab Pro A100-SXM4-40GB, code b3d2b35, started about 02:36 UTC 2026-10-19.
- All 330 steps ran, no early stop. Wall time 1,377 s. Final train loss 1.291.
- Eval loss: 1.6454 (step 100), 1.6361 (200), 1.6317 (300), 1.6326 (330). Best is step 300, so the final adapter is
  `checkpoint-300`, saved at `fyM1-ckpt` `fym1-9b-a100-r2/final/`.
- r2's eval loss can't be compared with r1's 1.356 (`best_eval_loss` in `results/train_log.json`): the eval split
  and the mix are different.
- The Colab runtime is already disconnected and deleted.
- Not measured yet (no answers, judge or gates).

### 0.2 Owner verdict on r2 and the ship decision

- Owner's hands-on impression after trying r2 (not a measured number): "r2 is a lot better, but it still doesn't give
  a full assistant answer in an everyday style."
- OWNER DECISION: ship fy-m1 v1 = r2 (`fym1-9b-a100-r2`, checkpoint-300) as is, with a clean system prompt that
  tells it to be a helpful assistant (everyday Malagasy, says it is fy-m1). No r3 before shipping. r3 (a dedicated
  casual Q&A set from `04_synth`, tuning the real-pair share) moves to Future work.

### 0.3 Plan for the next session, in order

1. **System prompt.** Write the clean helpful-assistant system prompt in one place, shared by `05_try`,
   `02_eval_export` and the Spaces. It differs from the training system prompt (`system_prompt.txt` in fyM1-chats);
   say so honestly in the paper.
2. **Measure r2** against base and r1 at the same decoding settings (`src/fym1/generation.py`) with the new system
   prompt: answer rate, naturalness, identity, leak gate, Belebele, FLORES. Score as in PR #73 (`score.py`,
   `identity_check`, the leak gate). Put r2's train log in `results/` (`scripts/fetch_train_log.py`). Watch
   naturalness: real pairs are only about 5% of r2, and r1's naturalness gain came mostly from real chats.
3. **Docs from numbers.** Final README, model card and paper from `paper/make_numbers.py --strict`. r1 is the
   documented failure mode (persona leakage from group-chat data). Disclose that 2 identity training turns contain
   a short identity-check prompt, and one line on the repetition-penalty difference between the notebook and the
   Space (transformers applies it to the whole sequence, llama.cpp to the last 64 tokens; see `generation.py`). The
   README "whole path" diagram and funnel still say "formal" chats; r2 leaves them out and adds Aya.
4. **Ship the model.** `03_trim`, merge the adapter, 4-bit + GGUF, Spaces `fy-m1-chat` and `fy-m1-arena` with the new
   system prompt, and a Playwright smoke test of both Spaces.
5. **Public repo.** Release snapshot checks (the decision 35 rewrite of private ids and links; a fresh agent re-checks
   sender names and secrets), then `Fan1000-prog/fy-m1` as one fresh commit.

Future work (not before v1): r3 with a generated casual assistant-style Q&A set and a tuned real-pair share.

### 0.4 Known limits and open issues

- The identity gate fails for both models so far.
- The persona filter misses names outside the fake pool, and "I'm X" / "Je suis X" lines.
- The Aya licence is a constant in code, not read from the dataset card.
- Eval imports the train split helper (a layering smell).
- `generation.py` says `tests/spaces/test_generation_copy.py` checks the Space copy; that file does not exist.
- The first r2 step took about 14 min (Unsloth 2026.10.6 + torch 2.11 compile failed and fell back).
- `01_train` overwrote the root `final/` in `fyM1-ckpt` with r2's `checkpoint-300`. r1 is still at
  `fym1-9b-a100-r1/final/`. Anything that reads the root `final/` now gets r2.
- The eval split can take identity chats (it did in r2).
- Issue #61 (three quadratic regex paths in the anonymiser) is still open.

### 0.5 Things that could be deleted (owner decides; nothing was deleted)

- Local branch `pr8-review-r2` (31 h old).
- Detached worktree `~/fyM1-sync`.
- Private copy `~/malagasy-data/private/train-0c7ca5d/`: keep while the leak gates need it.
- Old r1-era files in `fyM1-chats` (`chats.jsonl`, `formal.jsonl`, `synth/*`): r2 does not use them; keep them so
  r1 can be rebuilt.
- Note: `~/fyM1-persona-mix` and local branch `data/persona-mix` were deleted by main with the owner's OK.

## 1. Where we are

Tooling, the export reader and the greeting gate are merged. The base model is picked (§4a); nothing is trained.

- **Messenger export:** arrived (custom one-file JSON, 884 messages). The reader (#27) turns it into 267
  conversations with all senders. Dry run of the strict gate (#33, open): 258 pass. Far under 2,000, so
  synthetic chats are needed.
- **Base model:** picked: `McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1` (CC-BY-4.0), lead decision 2026-10-18
  (§4a). Gemma 4 12B stays a comparison row.
- **GPU:** Kaggle ran the Gemma 3 baseline (version 5). The Gemma 4 baseline moved to Colab. Training and eval
  run on Colab Pro (A100, paid; owner order 2026-10-18); a Kaggle T4 is only the fallback.

`main`: run `uv run pytest -q` for the current test count; it must be green. The expected failures
(strict xfails) pin known anonymiser limits.

## 2. Done

| Plan task | What | PRs |
|---|---|---|
| 1 Repos + scaffold | Private GitHub repo, package layout, `.gitignore` tests, private HF repos `fyM1-chats`, `fyM1-ckpt`, `fyM1-votes` | #1 |
| 2 Data | Messenger parser, anonymiser, segmenter, review sheet, `build_chats.py`, `build_formal.py` | #3 |
| 2 Data | Anonymiser leak fixes: a mention's fake never echoes the real name (also inside or split across a handle); glued `+261`/`00261`/`(034)` phones caught | #16 |
| 2 Data | Phone-vs-date follow-up: local numbers with a year-like group, price pairs kept | #24 |
| 2 Data | Real dates masked first, so a phone never starts inside a date (fixes the #24 date side effect) | #29 |
| 2 Data | One-file JSON export reader, `--all-senders`, `build_chats --dry-run`, nickname events from flat exports harvested | #27 |
| 2 Data | Phones with unicode dashes, no-break spaces and a glued letter O | #31 |
| 2 Data | Phones fail closed around dates; no-0 mobiles (`3[2-9]…`) masked | #32 |
| 2 Data | Synthetic everyday chats: `synth.py`, `04_synth` notebook, stats script | #30 |
| 2 Data | Uploaded to private `fyM1-chats`: 1,500 Wikipedia chats, 16 owner-written identity/refusal chats, system prompt | — |
| 3 Eval | `is_malagasy`, `drift_rate`, `leak_rate`, 150 owner-approved prompts, judge rubric | #4, #15 |
| 3 Eval | Blind greeting gate for the base-model pick (`results/greeting_gate.json`) | #34 |
| 4 Results | Gemma 3 and Gemma 4 baselines + greeting gate in the selection table | #35 |
| 4 Train infra | Checkpoint resume, smoke notebook, token-safe Colab flow | #2 |
| 4 Baseline + training | `00_baseline`, `01_train`, per-run folders that refuse a resume on changed settings or code, training time across disconnects, own early stop, `sync_code_to_hub.py` | #13 |
| 4 Kaggle | Runtime helper (Kaggle before Colab), `kaggle_push.py`, notebooks run on either, one GPU pinned, fresh code folder per run, notebook builder in the repo | #18, #19, #20 |
| 4 Gemma 3 load | Load `google/gemma-3-12b-it` quantised on load (no silent swap to the oversized copy), out-of-memory guard records `"oom"`, never a number | #21 |
| 4 Guards | Baseline/train refuse a different model id, record `loaded_from`; chat Belebele `oom` column | #22 |
| 4 Results | AfriqueQwen3.5-9B-Instruct baseline + selection table | #23 |
| 5 Scoring | `score.py`: blind judge batch, judge means, `scores.json`, leak matches, owner leak review, strict leak gate | #6, #10, #11, #15 |
| 6 Spaces | Chat + arena apps, votes, owner-approved Malagasy text, `fy-m1` names | #7, #9, #12 |
| 7 Shrink | Vocabulary trim (notebook not run yet) | #8 |
| 8 Writing | Paper, README, model card skeletons; project updates, spec §8, plan status | #5, #17, #25, #26 |
| 8 Writing | Project update snapshot, operator notes, licence decisions | #36 |
| 2 Data | Synthetic chats: permissive generators only (AfriqueQwen3.5-9B-Instruct, Gemma 4) | #37 |
| 2 Data | `chats.jsonl` rebuilt with the #51 fake-name pool and re-uploaded (`fyM1-chats` a1b58d3): same counts (192 kept, 436 messages); sender-name hits in the file 12 -> 0 | #52 |
| — Name + process | Public name `fy-m1`, open-ended PR loop rules, name guard test | #28 |

## 3. Decisions made outside the spec and plan

Older ones are also in spec §8 and the plan's "Changes from the plan"; the 2026-10-18 ones are recorded here.

- **Name:** `fy-m1` everywhere public. Public ids `Fan1000-Prog/fy-m1-*` (model `fy-m1-<size>`, Spaces
  `fy-m1-chat`, `fy-m1-arena`); GitHub release repo `Fan1000-prog/fy-m1`, recreated clean at release (owner
  pre-approved deleting and recreating it). Private repos keep `fyM1-*`; package and internal ids stay `fym1`.
- **Export (owner):** use ALL senders. Every sender's side is the assistant in turn (all participants
  consented). 884 messages → 267 conversations; the strict-gate dry run keeps 258.
- **Gates and process (owner):** the owner-review gates are now strict automatic gates (§0.6). PR loop has no
  round cap; the same blocker 3 rounds running → the lead decides. Any agent near 25% context hands off (lead →
  lead-2 happened).
- **Licence (owner):** fy-m1 ships under a simple open licence. So the base must be **Gemma 4 12B
  (Apache-2.0)** or **AfriqueQwen3.5-9B-Instruct (CC-BY-4.0)**. Synthetic chats come from a permissive open
  model (the picked base), not Gemma 3. Gemma 3 stays a measured comparison baseline only.
- **wordfreq (owner):** the strict gate (#33) uses the `wordfreq` package to know ordinary French/English words.
  Its word data is CC-BY-SA 4.0: derived word lists in the public repo carry attribution and the licence note
  (README "Third-party data"), or stay out of the public snapshot and get regenerated by script.
- **Greeting ratings:** the owner's 1–5 ratings are replaced by a blind automatic judge over the 20 greeting
  answers per model (`results/greeting_gate.json`, #34).
- **AfriqueQwen-8B dropped:** a base model with no chat-tuned twin; it can't chat. Its partial numbers stay in
  the table.
- **Privacy backstops**, kept whatever the cleaner does: count-only phone check, strict 8-word leak gate,
  fail-closed phone masking (losing a price is cheap, leaking a number is not). Raw answers and the arena A/B
  key stay private until v0.2.
- **Compute (owner, 2026-10-18):** paid compute = Colab Pro. Free T4s (Kaggle, Colab) run the baseline and smoke
  tests; a Colab Pro A100 runs training and eval. Still no closed-API outputs in training data. On a T4 the measured smokes of Gemma 3 and AfriqueQwen3.5-9B-Instruct ran in float16
  (`qlora_smoke.compute_dtype` in `results/baseline/`); Gemma 4 is expected to need float32 under Unsloth (slower),
  not measured yet. Training code reaches
  the GPU only by syncing clean `origin/main` to the Hub, never during a run.
- **Judge disclosure:** "Claude Opus 5.5 (claude-opus-5-5), blind, run in Claude Code; rubric in
  `src/fym1/eval/judge_rubric.md`." Methods disclosure, not attribution.

GPU history: Kaggle versions 2–4 of the baseline failed (no GPU, no internet or no HF token reached the run);
version 5 ran Gemma 3 to the end; version 7 (Gemma 4) was queued, then cancelled and moved to Colab. Training
later moved to Colab Pro (A100).

**What the lead decided for the owner** (full list with reasons in the private operator handoff):

1. Trimmed baseline: Gemma 3 full; Gemma 4 plain-text Belebele + FLORES+; AfriqueQwen-8B dropped.
2. Greeting ratings → blind automatic judge.
3. Internal ids (`fym1_*` data keys, LaTeX macros, file names, Kaggle kernel ids) keep their names; only public
   display names changed.
4. Phone masking fails closed around dates; no-0 mobiles are over-masked on purpose.
5. Nickname events in flat exports are harvested and replaced like Meta's native shape.
6. Strict data gate: drop a conversation with any 7+-digit run (unless a price), an email, link or @handle; an
   unknown capitalised word is replaced with a fake name rather than dropping the conversation. No one reads the
   real text; only counts are reported.
7. #29 merged with unicode dashes still missed (main missed them too); fixed in #31 before any real run.

## 4. Baseline numbers so far

From `results/model_selection.json` and `results/greeting_gate.json` on `main` (written by
`scripts/build_model_selection.py` and the greeting gate); Gemma 3's raw run is
`results/baseline/gemma-3-12b-it.json` (Kaggle version 5). Licences come from the script's licence table.
Gemma 3 is a comparison baseline only: its licence (Gemma Terms of Use) rules it out as the base. The Status row
is the `status` field in `model_selection.json`.

| | AfriqueQwen3.5-9B-Instruct | Gemma 4 12B | Gemma 3 12B (baseline only) | AfriqueQwen-8B (dropped) |
|---|---|---|---|---|
| Belebele, plain text 0-shot | 0.667 | pending (Colab run) | 0.697 | pending |
| Belebele, chat format | 0.742 | 0.750 | 0.751 | not valid (base) |
| FLORES+ chrF++ eng→mg / mg→eng | 49.73 / 56.17 | pending (Colab run) | 36.84 / 51.77 | pending |
| Tokens per Malagasy word | 2.826 | 2.792 | 2.792 | 2.952 |
| Smoke: s/step, peak GPU | 83.3, 9.8 GB | pending (Colab run) | 122.4, 10.26 GB | 74.3, 7.79 GB |
| Smoke loss | 1.00 → 0.73 (0.77 last) | pending (Colab run) | 1.60 → 1.21 (1.24 last) | 1.13 → 0.70 |
| Greeting gate, judge mean (20 answers) | 2.90 | 2.85 | 1.90 | not judged |
| Greeting gate, Malagasy rate | 0.95 | 0.90 | 0.80 | not judged |
| Licence | CC-BY-4.0 | Apache-2.0 | Gemma Terms of Use | CC-BY-4.0 |
| Status | candidate | candidate | dropped (licence; baseline only) | dropped (base model, no chat) |

The 9B and Gemma 4 are close on greetings (a tie in practice).

## 4a. Base-model pick (lead decision, 2026-10-18)

**fy-m1 base = `McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1` (CC-BY-4.0).** Recorded in
`results/model_selection.json` (`chosen`, `chosen_size`, `license`, `hf_license`, `chosen_reason`); rerunning
`scripts/build_model_selection.py` keeps them.

| Field in `model_selection.json` | AfriqueQwen3.5-9B-Instruct | Gemma 4 12B | Gemma 3 12B |
|---|---|---|---|
| `chrf_eng_plt` (FLORES+ eng→mg) | 49.73 | pending | 36.84 |
| `greeting_gate.judge_mean` | 2.9 | 2.85 | 1.9 |
| `belebele_chat_acc` | 0.7422 | 0.75 | 0.7511 |
| `sec_per_step` (T4 smoke) | 83.3 (float16) | pending | 122.4 |
| `license` | CC-BY-4.0 | Apache-2.0 | Gemma Terms of Use |

Why: best eng→mg translation measured; ties Gemma 4 on greetings and chat Belebele (equal within noise); its
T4 smoke ran in float16, while Gemma 4 is expected to need float32 under Unsloth (slower; its smoke is pending);
simple open licence (owner rule). Gemma 4's FLORES+ and smoke were not waited for; it stays a comparison row
(Kaggle version 8 may fill it in). Known risk: one Hausa reply and some prompt echoes in its greeting answers.

**`01_train`:** `MODEL_ID` has no default by design. Set it to exactly
`MODEL_ID = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"`.

## 5. Open now

| Item | Who | State |
|---|---|---|
| PR #33 strict data gate | lead | open; review rounds continue. Must merge before the real `build_chats` run |
| Gemma 4 baseline (plain-text Belebele, FLORES+, smoke) | trainer | comparison row only; Kaggle version 8 may fill it in |
| Base-model pick | lead | done: AfriqueQwen3.5-9B-Instruct (§4a) |
| `synth.py` / `04_synth` default generator | data | AfriqueQwen3.5-9B-Instruct first (the picked base), Gemma 4 fallback (#37) |
| wordfreq-derived word lists | writer, lead | attribution in README; decide at release whether lists ship or get regenerated |
| Private `fyM1-chats` files | data | rename to `fy-m1` in the system prompt and identity lines |

## 6. Steps to finish

1. **Merge #33** (strict gate). Base model picked (§4a).
2. **Real data run** from clean `main`: `build_chats.py --all-senders`, strict gate, counts only into
   `results/` via a PR.
3. **Fix the private dataset names** (`fy-m1`) and **upload** the gated chats to private `fyM1-chats`.
4. **Synthetic chats** with AfriqueQwen3.5-9B-Instruct (first entry of `04_synth` `GENERATORS`): run on Kaggle,
   automatic sample gate, promote.
5. **Sync code** from clean `main`; set `MODEL_ID = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"` and a
   `RUN_ID` in `01_train` (no defaults); **train** on Colab Pro (A100; Kaggle T4 fallback only); log to
   `results/train_log.json`.
6. Optional: Gemma 4 FLORES+/smoke (Kaggle version 8) as a comparison row only.
7. **Eval** (Task 5, Colab Pro A100): 150 answers from base and fy-m1, Belebele + FLORES+ rerun, perplexity, blind judge,
   `scores.json`; merge LoRA, 4-bit + GGUF Q4_K_M.
8. **Leak gate** (automatic, strict): any 8-word match fails → fix data, filter or retrain. Automatic identity
   check (`scripts/identity_check.py`, strict): the 20 identity/safety answers plus the extra identity
   prompts in `eval/identity_check.jsonl`; zero explicit self-claims to be another model (hard fail) and a
   pass rate of at least 90%, where an unexcused foreign mention on an about-self prompt is an ordinary miss
   (two-tier rule, decision 33: the module docstring of `src/fym1/eval/identity_check.py`).
9. **Trim** (Task 7): run `03_trim`; ship trimmed GGUF only if all three gates pass.
10. **Deploy Spaces** `fy-m1-chat` and `fy-m1-arena` (Task 6); Playwright smoke test.
11. **Write-up** (Task 8) from `results/` via `paper/make_numbers.py --strict`; licence = the picked base's.
12. **Release** (Task 9): recreate `Fan1000-prog/fy-m1` from a clean snapshot, secret scan, settle the wordfreq
    lists, flip public (private repos stay private), check every README link, send the owner the arena link and a
    2-line Malagasy message.

v0.2 after release: panel arena votes, layer pruning, longer training, raw answers once the arena closes.
