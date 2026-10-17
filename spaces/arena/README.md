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
  `Fan1000-Prog/fyM1-votes` (the app refuses to run if it is public). One vote per browser per pair.

Space settings:

| Name | Kind | Value |
|---|---|---|
| `HF_TOKEN` | secret | write token for `Fan1000-Prog/fyM1-votes` |
| `PAIRS_FILE` | variable | optional, default `pairs.jsonl` |
| `VOTES_REPO` | variable | optional, default `Fan1000-Prog/fyM1-votes`; empty keeps votes local (dev) |

Run locally: `PAIRS_FILE=pairs.sample.jsonl VOTES_REPO= python app.py`.
Tally: `uv run python scripts/tally_votes.py` → `results/eval/arena.json` (win rate, ties excluded, with a
95% Wilson interval). Code: https://github.com/Fan1000-prog/fyM1
