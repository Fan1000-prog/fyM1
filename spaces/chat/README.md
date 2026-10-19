---
title: fy-m1 chat
emoji: 💬
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Chat in everyday Malagasy with fy-m1 (free CPU, slow)
---

# fy-m1 chat

Demo of fy-m1, an everyday-Malagasy chat model, running its GGUF (Q4_K_M) with llama.cpp on the free
2-vCPU Space. Replies stream token by token; expect it to be slow.

Space settings:

| Name | Kind | Value |
|---|---|---|
| `MODEL_REPO` | variable | model repo, e.g. `Fan1000-Prog/fy-m1-8B` |
| `MODEL_FILE` | variable | GGUF file in that repo, e.g. `fy-m1-8B-Q4_K_M.gguf` |
| `SYSTEM_PROMPT` | variable | owner-approved Malagasy system prompt |
| `HF_TOKEN` | secret | read token; only while the model repo is private |

The app refuses to start if a variable is missing. Sampling and history length come from
`generation.py`, a byte-identical copy of `src/fym1/generation.py` (temperature 0.6, top_p 0.9, top_k 50,
repeat_penalty 1.1, 256 new tokens; system prompt plus the last 3 exchanges). Edit the original, then copy it here.
Code: https://github.com/Fan1000-prog/fy-m1
