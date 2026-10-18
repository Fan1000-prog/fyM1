# fyM1 — agent conventions

New session or new team: start at `docs/project-update.md` §0.

Read `docs/superpowers/specs/2026-10-17-resaka-design.md` (§7 and §8 override earlier sections) and
`docs/superpowers/plans/2026-10-17-fym1-v0.1.md` before working.

## Hard rules
- Raw Messenger exports (the owner's export has arrived) live in `~/malagasy-data/raw/` and never enter this repo,
  Colab, or any HF repo.
  Only anonymised, owner-approved data goes to the private HF dataset `Fan1000-Prog/fyM1-chats`.
- Test fixtures are synthetic. Never paste real chat text into tests, commits, PRs, or results.
- Paid compute = Colab Pro (A100 for training + eval; free T4 on Kaggle/Colab for baseline + smoke tests). Still no
  closed-API (Claude/GPT/Gemini API) outputs in training data.
- No AI attribution anywhere: no `Co-Authored-By`, no "Generated with", no 🤖.
- Every number in README / paper / model card comes from a file in `results/` written by a script here.
- One branch + worktree per task (`../fyM1-<topic>`), PR, fresh reviewer. Blocked review → a fresh ML-engineer agent
  fixes → a fresh reviewer, repeated until safe-to-merge + green CI. No round cap. The same blocker 3 rounds in a row →
  the lead decides and records the call + reason in the PR body and the handoff. Never force-push.
- Owner-review gates (sample sheet, leftover names, leak gate, identity check, greeting ratings) are strict automatic
  gates run by fresh agents: drop anything unresolved, fail on any 8-word leak match, list what was auto-decided.
- Any agent near 25% context used: write a handoff, then a fresh agent takes over.
- Public name is `fy-m1` (README, model card, paper, Spaces, identity lines; ids `Fan1000-Prog/fy-m1-*`). Private HF
  repos keep `fyM1-*`, the package stays `fym1`. `tests/test_public_name.py` guards it.

## Layout
- `src/fym1/{data,eval,shrink,train}` — tested Python (`uv run pytest`).
- `scripts/` — CLI entry points. `notebooks/` — Colab/Kaggle notebooks (T4 for baseline/smoke, Colab Pro A100 for
  training/eval); `00_baseline`, `01_train` and `04_synth` are built by `scripts/build_notebooks.py` (edit it, rebuild, commit both; a test checks they match).
- `spaces/{chat,arena}` — Hugging Face Spaces. `paper/` — LaTeX. `results/` — measured outputs.

## Style target
Everyday Malagasy: French mixing, texting spelling, slang, emoji are correct. A reply "drifts" only if it is
full French/English with no Malagasy.
