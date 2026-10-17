# fyM1 — agent conventions

Read `docs/superpowers/specs/2026-10-17-resaka-design.md` (§7 overrides earlier sections) and
`docs/superpowers/plans/2026-10-17-fym1-v0.1.md` before working.

## Hard rules
- Raw Messenger exports live in `~/malagasy-data/raw/` and never enter this repo, Colab, or any HF repo.
  Only anonymised, owner-approved data goes to the private HF dataset `Fan1000-prog/fyM1-chats`.
- Test fixtures are synthetic. Never paste real chat text into tests, commits, PRs, or results.
- No closed-API (Claude/GPT/Gemini API) outputs in training data. Budget $0.
- No AI attribution anywhere: no `Co-Authored-By`, no "Generated with", no 🤖.
- Every number in README / paper / model card comes from a file in `results/` written by a script here.
- One branch + worktree per task (`../fyM1-<topic>`), PR, fresh reviewer, merge when clean. Never force-push.

## Layout
- `src/fym1/{data,eval,shrink,train}` — tested Python (`uv run pytest`).
- `scripts/` — CLI entry points. `notebooks/` — Colab (T4) notebooks.
- `spaces/{chat,arena}` — Hugging Face Spaces. `paper/` — LaTeX. `results/` — measured outputs.

## Style target
Everyday Malagasy: French mixing, texting spelling, slang, emoji are correct. A reply "drifts" only if it is
full French/English with no Malagasy.
