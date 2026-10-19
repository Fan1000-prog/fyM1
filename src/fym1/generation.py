"""Decoding defaults and chat-history trimming, shared by every place fy-m1 generates free text.

Owner-approved (2026-10-18): temperature 0.6 and repetition_penalty 1.1, for steadier, less loopy replies than the
earlier temperature 0.7 with no penalty. top_p stays 0.9. top_k was never set: transformers then used its default
50 (the base model's generation_config sets none) while llama.cpp used its own 40, so the notebooks and the chat
Space sampled differently; 50 is now explicit on both.

Interactive chat keeps the system prompt plus the last HISTORY_EXCHANGES user/assistant exchanges before the
message being answered. Why 3: the owner asked for a short history; the chats fy-m1 is trained on are quick
texting back-and-forth where a reply leans on the last few lines, so 3 exchanges still cover a follow-up like
"ary ianao?". A longer history gives the model more of its own past replies to copy (repetition) and makes the
prompt slower to process on the 2-vCPU chat Space.

Scored eval generation that must stay comparable with 00_baseline (FLORES+ translation, greedy) does not use
these settings.

Stdlib only: spaces/chat ships a byte-identical copy (tests/spaces/test_generation_copy.py checks it).
"""

TEMPERATURE = 0.6
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.1
HISTORY_EXCHANGES = 3

# transformers `model.generate(**GEN_KWARGS, ...)`.
GEN_KWARGS = {
    "do_sample": True,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "repetition_penalty": REPETITION_PENALTY,
}

# llama-cpp-python `create_chat_completion(**LLAMA_CPP_KWARGS, ...)`: same values, llama.cpp names.
# Note: llama.cpp applies repeat_penalty to the last 64 tokens; transformers applies it to the whole sequence.
LLAMA_CPP_KWARGS = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "repeat_penalty": REPETITION_PENALTY,
}


def trim_history(system_prompt: str, history: list[dict], max_exchanges: int = HISTORY_EXCHANGES) -> list[dict]:
    """The system prompt once, then the last `max_exchanges` exchanges before the newest user message.

    `history` is the chat so far, oldest first, usually ending on the user message to answer. Any system message
    in it is dropped, so the result never holds the system prompt twice. An exchange starts at a user message;
    the kept part always starts on a user message, as chat templates require.
    """
    if max_exchanges < 0:
        raise ValueError(f"max_exchanges must be >= 0, got {max_exchanges}")
    turns = [m for m in history if m["role"] != "system"]
    user_starts = [i for i, m in enumerate(turns) if m["role"] == "user"]
    # The newest user message is the one being answered; keep it plus max_exchanges user messages before it.
    keep_users = max_exchanges + (1 if turns and turns[-1]["role"] == "user" else 0)
    kept_starts = user_starts[-keep_users:] if keep_users else []
    kept = turns[kept_starts[0]:] if kept_starts else []
    return [{"role": "system", "content": system_prompt}, *kept]
