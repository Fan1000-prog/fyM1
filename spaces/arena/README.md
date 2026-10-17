---
title: fyM1 arena
emoji: ⚖️
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Blind A/B rating of Malagasy chat answers
---

# fyM1 arena

Raters see a prompt and two unlabelled answers (base model vs fyM1, order shuffled) and pick the more
natural one: **A**, **B** or **Mitovy** (same). They can also flag an answer as **Diso** (wrong). No login.

- `pairs.jsonl`: one `{"pair_id", "prompt", "a", "b", "a_is"}` per line, `a_is` is `base` or `fym1` and is
  never shown. Built from the eval answers; the app refuses to start without it.
- `pairs.sample.jsonl`: 3 **synthetic** pairs for local runs only (`PAIRS_FILE=pairs.sample.jsonl`).
- Votes are appended to a JSONL file and pushed every 2 minutes to the **private** dataset
  `Fan1000-Prog/fyM1-votes` (the app refuses to run if it is public).

## Vote stuffing (no login, by choice)

The owner wants zero friction for raters, so there is no login. That means one person *can* vote again
from a new browser or a private window. What limits it:

- The server signs each session id with `ARENA_KEY`; ids it did not sign (or over 128 characters) are
  rejected, so a script cannot invent ids. Vote events are not exposed in the public API.
- One vote per session per pair, and at most as many votes per session as there are pairs.
- The tally drops repeat (session, pair) votes and leaves out sessions whose median time between votes is
  under 2 seconds; `arena.json` reports how many sessions and votes were left out, and votes per session.

So the win rate is good evidence from friendly raters, not proof against a determined cheater.

Space settings:

| Name | Kind | Value |
|---|---|---|
| `ARENA_KEY` | secret | long random string that signs session ids; required |
| `HF_TOKEN` | secret | write token for `Fan1000-Prog/fyM1-votes` |
| `PAIRS_FILE` | variable | optional, default `pairs.jsonl` |
| `VOTES_REPO` | variable | optional, default `Fan1000-Prog/fyM1-votes`; empty keeps votes local (dev) |

Run locally: `ARENA_KEY=dev PAIRS_FILE=pairs.sample.jsonl VOTES_REPO= python app.py`.
Build pairs: `uv run python scripts/build_pairs.py --base results/eval/answers_base.jsonl --fym1 results/eval/answers_fym1.jsonl --out spaces/arena/pairs.jsonl --seed 42`.
Tally: `uv run python scripts/tally_votes.py` → `results/eval/arena.json` (win rate, ties excluded, with a
95% Wilson interval). Code: https://github.com/Fan1000-prog/fyM1
