---
title: fyM1 chat
emoji: 💬
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Chat in everyday Malagasy with fyM1 (free CPU, slow)
---

# fyM1 chat

Demo of fyM1, an everyday-Malagasy chat model, running its GGUF (Q4_K_M) with llama.cpp on the free
2-vCPU Space. Replies stream token by token; expect it to be slow.

Space settings:

| Name | Kind | Value |
|---|---|---|
| `MODEL_REPO` | variable | model repo, e.g. `Fan1000-Prog/fyM1-8B` |
| `MODEL_FILE` | variable | GGUF file in that repo, e.g. `fyM1-8B-Q4_K_M.gguf` |
| `SYSTEM_PROMPT` | variable | owner-approved Malagasy system prompt |
| `HF_TOKEN` | secret | read token; only while the model repo is private |

The app refuses to start if a variable is missing. Sampling matches the eval run (temperature 0.7, top_p 0.9,
256 new tokens). Code: https://github.com/Fan1000-prog/fyM1
