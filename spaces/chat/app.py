"""fyM1 chat demo: the GGUF model on a free CPU Space, streamed token by token.

Env (Space variables): MODEL_REPO, MODEL_FILE (the GGUF in that repo), SYSTEM_PROMPT (owner-approved line).
HF_TOKEN (Space secret) is only needed while the model repo is private.
"""

import os

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from prompt import build_messages

GITHUB_URL = "https://github.com/Fan1000-prog/fyM1"
FOOTER = f"[Code]({GITHUB_URL}) · [Paper]({GITHUB_URL}/tree/main/paper) · [Arena](https://huggingface.co/spaces/Fan1000-Prog/fyM1-arena)"
# Same sampling as the eval run, so the demo shows the model that was measured.
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 256}
CONTEXT_TOKENS = 4096
CPU_THREADS = 2  # free CPU Space has 2 vCPU; os.cpu_count() reports the host's cores


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set the {name} variable in the Space settings")
    return value


SYSTEM_PROMPT = required_env("SYSTEM_PROMPT")
llm = Llama(
    model_path=hf_hub_download(required_env("MODEL_REPO"), required_env("MODEL_FILE")),
    n_ctx=CONTEXT_TOKENS,
    n_threads=CPU_THREADS,
    verbose=False,
)


def respond(message: str, history: list[dict]):
    reply = ""
    stream = llm.create_chat_completion(build_messages(SYSTEM_PROMPT, history, message), stream=True, **SAMPLING)
    for chunk in stream:
        reply += chunk["choices"][0]["delta"].get("content") or ""
        yield reply


with gr.Blocks(title="fyM1") as demo:
    gr.ChatInterface(
        respond,
        title="fyM1",
        examples=["Sali ahn", "De ahoana?"],
        cache_examples=False,
        concurrency_limit=1,  # one Llama instance: generations run one at a time, others wait in the queue
    )
    gr.Markdown(FOOTER)

if __name__ == "__main__":
    demo.launch()
