"""Build notebooks/00_baseline.ipynb and notebooks/01_train.ipynb. This script is their source of truth.

Usage: python scripts/build_notebooks.py [--check]
Edit the cells here, run it, and commit both this file and the notebooks. --check exits non-zero when a
committed notebook differs from what this script builds (tests/test_build_notebooks.py runs the same check).
The bootstrap cell carries a verbatim copy of fym1.runtime.hf_token and its helpers, taken from
src/fym1/runtime.py at build time: notebooks need the HF token before they can download fym1.
"""

import argparse
import ast
import sys
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PY = REPO_ROOT / "src" / "fym1" / "runtime.py"
COPIED = ("MISSING_TOKEN", "on_kaggle", "_secret", "hf_token")
METADATA = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def runtime_segments() -> list[str]:
    source = RUNTIME_PY.read_text()
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        if name in COPIED:
            found[name] = ast.get_source_segment(source, node)
    return [found[n] for n in COPIED]


def bootstrap_cell():
    return code(
        "# Runtime bootstrap (Colab or Kaggle). The HF token is needed before fym1 can be downloaded, so this cell\n"
        "# carries a verbatim copy of fym1.runtime.hf_token and its helpers (checked by tests/test_notebook_runtime.py).\n"
        "import os\nimport tempfile\n\n"
        'os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Kaggle\'s T4 runtime may preset "0,1"; always use one GPU\n'
        'CODE_DIR = tempfile.mkdtemp(prefix="fym1code-")  # fresh each run: nothing stale on sys.path\n\n'
        + "\n\n\n".join(runtime_segments()))


def baseline_cells() -> list:
    return [
md("""# fyM1 00 — base-model baseline (T4)
One candidate per run. **Runtime → Run all**; the notebook picks the first candidate in `CANDIDATES` whose
results file on `Fan1000-Prog/fyM1-ckpt` (`results/baseline/<slug>.json`) still misses a required section, and
skips sections already saved there, so a rerun after a disconnect resumes. Restart the runtime between candidates
(**Runtime → Restart session**, then **Run all**). Set `FORCE` to run one candidate explicitly.

Measures, all 4-bit: Belebele `plt_Latn` (900, 0-shot, next-token over 1–4), FLORES+ `plt_Latn`↔`eng_Latn`
devtest first 200 chrF++ (greedy), tokens per Malagasy word (1,000 Wikipedia sentences), 20 greeting answers,
and a short QLoRA smoke train (seq 2048, bs 1 × grad-accum 8) for peak VRAM and sec/step.
Runs on Colab or Kaggle (T4). Needs the `HF_TOKEN` secret (Colab: Secrets pane with notebook access; Kaggle: Add-ons → Secrets) and code synced to `fyM1-ckpt` `code/` (`scripts/sync_code_to_hub.py`)."""),
code("""!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q -U unsloth unsloth_zoo sacrebleu"""),
code("""CANDIDATES = {
    # slug: (hub id, QLoRA smoke steps, push 4-bit weights to fyM1-ckpt base-4bit/<slug>/, base model?)
    # Order = run order. 4-bit push off: on T4 Unsloth keeps gemma4 in fp32 and save_pretrained stalled at
    # ~11.5/12.7 GB CPU RAM. Belebele: plain-text completion for all; chat-template score only for instruct models.
    # Gemma 3 is quantized on load like Gemma 4: Unsloth's pre-quantized bnb-4bit checkpoint keeps many layers
    # unquantized and used 11.9 GB of the T4 right after loading (OOM at the first Belebele batch).
    "gemma-3-12b-it": ("google/gemma-3-12b-it", 15, False, False),
    "afriqueqwen3.5-9b-instruct": ("McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1", 15, False, False),
    "gemma-4-12b-it": ("google/gemma-4-12B-it", 15, False, False),
    "afriqueqwen-8b": ("McGill-NLP/AfriqueQwen-8B", 15, False, True),
}
FORCE = None  # e.g. "afriqueqwen-8b"
REPO = "Fan1000-Prog/fyM1-ckpt"
BELEBELE_BATCH, GEN_BATCH, FLORES_N, TPW_SENTENCES = 4, 8, 200, 1000
FALLBACK_GREETINGS = ["Sali ahn", "De ahoana?", "Manao ahoana ianao?", "De aona ry zalahy?", "Salut, mbola tsy natory ve?"]"""),
bootstrap_cell(),
code("""import sys, json, time, datetime
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

os.environ["HF_TOKEN"] = hf_token()  # never print it
api = HfApi()
code_dir = snapshot_download(REPO, allow_patterns=["code/**"], local_dir=CODE_DIR)
sys.path.insert(0, f"{code_dir}/code/src")
sha_file = f"{code_dir}/code/SHA"
CODE_SHA = open(sha_file).read().strip() if os.path.exists(sha_file) else "unknown"  # git commit of the synced code
from fym1.eval import baseline as bl
from fym1.eval.batching import StillOutOfMemory, run_batched
from fym1.train.chat_format import CHAT_TEMPLATE_KWARGS
from fym1.runtime import output_dir, scratch_dir

OUTPUT, SCRATCH = output_dir(), scratch_dir()  # small wanted results vs big throwaway files


def load_results(slug):
    try:
        return json.load(open(hf_hub_download(REPO, f"results/baseline/{slug}.json", force_download=True)))
    except EntryNotFoundError:
        return {}


def save_results(slug, results):
    for name, section_result in results.items():  # each section records the code and FORCE of its run
        if name not in ("model", "env") and isinstance(section_result, dict):
            section_result.setdefault("code_sha", CODE_SHA)
            section_result.setdefault("force", FORCE)
    path = f"{OUTPUT}/{slug}.json"
    with open(path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    api.upload_file(path_or_fileobj=path, path_in_repo=f"results/baseline/{slug}.json", repo_id=REPO,
                    commit_message=f"baseline {slug}: {', '.join(k for k in results if k not in ('model', 'env'))}")


SLUG = FORCE or next((s for s in CANDIDATES if bl.missing_sections(load_results(s), CANDIDATES[s][3])), None)
assert SLUG, "all candidates complete"
HUB_ID, SMOKE_STEPS, PUSH_4BIT, BASE = CANDIDATES[SLUG]
R = load_results(SLUG)
R.pop("done", None)  # completeness is missing_sections(), not a stored flag
for name, section_result in R.items():  # sections saved before code SHA / FORCE were recorded
    if name not in ("model", "env") and isinstance(section_result, dict):
        section_result.setdefault("code_sha", "unrecorded")
        section_result.setdefault("force", "unrecorded")
R.setdefault("model", {"slug": SLUG, "hub_id": HUB_ID, "quant": "bnb 4-bit nf4 (Unsloth load_in_4bit)"})
print("CANDIDATE", SLUG, "| missing:", bl.missing_sections(R, BASE))"""),
code("""import torch
from unsloth import FastModel

def section(name):
    print(f"== {name}: {'skip (saved)' if name in R else 'run'}")
    return name not in R

t0 = time.time()
try:
    model, processor = FastModel.from_pretrained(HUB_ID, max_seq_length=2048, load_in_4bit=True, full_finetuning=False)
except Exception as e:  # gated/403 or unsupported architecture: record and move on
    R["skipped"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}  # missing_sections() then treats it as complete
    save_results(SLUG, R)
    raise
tok = getattr(processor, "tokenizer", processor)
tok.padding_side = "left"
R["env"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "model_dtype": str(model.dtype),
            "code_sha": CODE_SHA,
            "vram_after_load_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
            "load_s": round(time.time() - t0), "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
print(R["env"], "vram after load GB:", round(torch.cuda.memory_allocated() / 2**30, 2))


def chat(prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)


def encode(texts, special_tokens=False):
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=special_tokens).to("cuda")


def batched(items, batch_size, run):
    \"\"\"On CUDA OOM: empty the cache and retry that batch one item at a time; StillOutOfMemory if that fails.\"\"\"
    return run_batched(items, batch_size, run, is_oom=lambda e: isinstance(e, torch.OutOfMemoryError),
                       clear=torch.cuda.empty_cache)


@torch.inference_mode()
def generate(prompts, max_new_tokens, sample):
    kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)

    def run(batch):
        enc = encode([chat(p) for p in batch])
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        return tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return batched(prompts, GEN_BATCH, run)


@torch.inference_mode()
def last_logits(enc):
    try:
        return model(**enc, logits_to_keep=1).logits[:, -1]
    except TypeError:  # forward without logits_to_keep
        return model(**enc).logits[:, -1]"""),
code("""if PUSH_4BIT and section("base_4bit"):
    t0 = time.time()
    d = f"{SCRATCH}/base-4bit/{SLUG}"
    try:
        model.save_pretrained(d)
        processor.save_pretrained(d)
        api.upload_folder(folder_path=d, path_in_repo=f"base-4bit/{SLUG}", repo_id=REPO,
                          commit_message=f"bnb-4bit weights of {HUB_ID}")
        R["base_4bit"] = {"path": f"base-4bit/{SLUG}", "upload_s": round(time.time() - t0)}
    except Exception as e:  # not fatal: evals still run
        R["base_4bit"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    save_results(SLUG, R)
    print(R["base_4bit"])"""),
code("""from datasets import load_dataset

FastModel.for_inference(model)
if not BASE and section("belebele"):
    t0 = time.time()
    rows = load_dataset("facebook/belebele", "plt_Latn", split="test")
    digit_ids = [tok.encode(d, add_special_tokens=False)[-1] for d in "1234"]
    protocol = "0-shot, chat template, English instruction, argmax of next-token logits over '1'-'4'"
    try:
        scores = torch.tensor(batched([chat(bl.belebele_prompt(r)) for r in rows], BELEBELE_BATCH,
                                      lambda b: last_logits(encode(b))[:, digit_ids].float().tolist()))
        R["belebele"] = {"acc": round(bl.accuracy((scores.argmax(-1) + 1).tolist(), rows["correct_answer_num"]), 4),
                         "n": len(scores), "nan_rows": int(torch.isnan(scores).any(-1).sum()),
                         "seconds": round(time.time() - t0), "protocol": protocol}
    except StillOutOfMemory:
        R["belebele"] = {"oom": True, "protocol": protocol}
    save_results(SLUG, R)
print("BELEBELE", R.get("belebele"))"""),
code("""if section("belebele_completion"):  # the one protocol for every model: plain text, no chat template
    t0 = time.time()
    rows = load_dataset("facebook/belebele", "plt_Latn", split="test")
    letter_ids = [tok.encode(" " + letter, add_special_tokens=False)[-1] for letter in "ABCD"]
    protocol = "0-shot plain-text completion (P/Q/A-D/Answer:), no chat template, argmax of next-token logits over ' A'-' D'"
    try:
        scores = torch.tensor(batched([bl.belebele_completion_prompt(r) for r in rows], BELEBELE_BATCH,
                                      lambda b: last_logits(encode(b, special_tokens=True))[:, letter_ids].float().tolist()))
        R["belebele_completion"] = {"acc": round(bl.accuracy((scores.argmax(-1) + 1).tolist(), rows["correct_answer_num"]), 4),
                                    "n": len(scores), "nan_rows": int(torch.isnan(scores).any(-1).sum()),
                                    "seconds": round(time.time() - t0), "protocol": protocol}
    except StillOutOfMemory:
        R["belebele_completion"] = {"oom": True, "protocol": protocol}
    save_results(SLUG, R)
print("BELEBELE_COMPLETION", R.get("belebele_completion"))"""),
code("""if section("flores"):
    t0 = time.time()
    try:
        mg = load_dataset("openlanguagedata/flores_plus", "plt_Latn", split="devtest")
        en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
        print("flores columns:", mg.column_names)
        mg_txt, en_txt = mg["text"][:FLORES_N], en["text"][:FLORES_N]
        if "id" in mg.column_names:
            assert mg["id"][:FLORES_N] == en["id"][:FLORES_N], "FLORES rows not aligned"
        hyp_mg = [bl.first_line(t) for t in generate([bl.translation_prompt(s, "English", "Malagasy") for s in en_txt], 256, False)]
        hyp_en = [bl.first_line(t) for t in generate([bl.translation_prompt(s, "Malagasy", "English") for s in mg_txt], 256, False)]
        R["flores"] = {"eng-plt_chrf++": round(bl.chrf_pp(hyp_mg, mg_txt), 2),
                       "plt-eng_chrf++": round(bl.chrf_pp(hyp_en, en_txt), 2), "n": FLORES_N,
                       "seconds": round(time.time() - t0),
                       "protocol": "devtest first 200, 0-shot chat prompt, greedy, max 256 new tokens, first line kept"}
        save_results(SLUG, R)
    except StillOutOfMemory:
        R["flores"] = {"oom": True}
        save_results(SLUG, R)
    except Exception as e:  # gated dataset not yet accepted: leave the section unsaved so a rerun retries it
        print("FLORES PENDING:", type(e).__name__, str(e)[:300])
print("FLORES", R.get("flores", "pending"))"""),
code("""if section("tokens_per_word"):
    wiki = load_dataset("wikimedia/wikipedia", "20231101.mg", split="train").shuffle(seed=42)
    sents = bl.sample_sentences((r["text"] for r in wiki), n=TPW_SENTENCES)
    R["tokens_per_word"] = {"value": round(bl.tokens_per_word(sents, lambda s: len(tok.encode(s, add_special_tokens=False))), 3),
                            "n_sentences": len(sents), "source": "wikimedia/wikipedia 20231101.mg, shuffle seed 42, 5-60 words"}
    save_results(SLUG, R)
print("TPW", R["tokens_per_word"])"""),
code("""if section("greetings"):
    try:
        rows = [json.loads(l) for l in open(hf_hub_download(REPO, "code/eval/prompts.jsonl"))]
        greetings = [r["prompt"] for r in rows if r["category"].startswith("greet")][:20]
        source = "eval/prompts.jsonl greetings, first 20"
    except EntryNotFoundError:
        greetings, source = FALLBACK_GREETINGS, "fallback made-up greetings"
    torch.manual_seed(42)
    sampling = "temp 0.7, top_p 0.9, max 128 new tokens, seed 42, no system prompt"
    try:
        answers = generate(greetings, 128, True)
        R["greetings"] = {"source": source, "sampling": sampling,
                          "answers": [{"prompt": p, "answer": a.strip()} for p, a in zip(greetings, answers)]}
    except StillOutOfMemory:
        R["greetings"] = {"oom": True, "source": source, "sampling": sampling}
    save_results(SLUG, R)
for a in R["greetings"].get("answers", []):
    print(repr(a["prompt"]), "->", repr(a["answer"][:200]))"""),
code("""if section("qlora_smoke"):
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    from datasets import Dataset

    wiki = load_dataset("wikimedia/wikipedia", "20231101.mg", split="train").shuffle(seed=7)
    chunks, buf = [], ""
    for r in wiki:  # ~2048-token chunks of Malagasy text: worst-case length for the fit test
        buf += r["text"] + "\\n\\n"
        if len(tok.encode(buf, add_special_tokens=False)) >= 2048:
            chunks.append(buf)
            buf = ""
            if len(chunks) == SMOKE_STEPS * 8:
                break
    multimodal = hasattr(model.config, "vision_config")
    lora = dict(r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=42, use_gradient_checkpointing="unsloth")
    if multimodal:
        model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                         finetune_attention_modules=True, finetune_mlp_modules=True, **lora)
    else:
        model = FastModel.get_peft_model(model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                "gate_proj", "up_proj", "down_proj"], **lora)
    step_times = []

    class StepTimer(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            step_times.append(time.time())

    tok.padding_side = "right"
    compute_dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())  # Unsloth may force fp32 on T4
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=Dataset.from_dict({"text": chunks}),
                         callbacks=[StepTimer()],
                         args=SFTConfig(output_dir=f"{SCRATCH}/smoke_out", per_device_train_batch_size=1,
                                        gradient_accumulation_steps=8, max_steps=SMOKE_STEPS, learning_rate=2e-4,
                                        lr_scheduler_type="cosine", warmup_steps=3, logging_steps=1, optim="adamw_8bit",
                                        fp16=compute_dtype == torch.float16, bf16=False, max_length=2048,
                                        dataset_text_field="text",
                                        packing=False, save_strategy="no", report_to="none", seed=42))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        try:
            trainer.train()
            stopped_early = None
        except KeyboardInterrupt:  # operator stop: what ran so far is still a valid measurement
            stopped_early = "interrupted by the operator"
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        gaps = [b - a for a, b in zip(step_times, step_times[1:])]
        R["qlora_smoke"] = {"ok": True, "steps": len(step_times), "steps_planned": SMOKE_STEPS,
                            "stopped_early": stopped_early, "seq_len": 2048, "batch": "1 x grad-accum 8",
                            "tokens_per_step": 8 * 2048, "lr": 2e-4,
                            "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                            "sec_per_step": round(sum(gaps) / len(gaps), 1) if gaps else None,
                            "seconds": round(time.time() - t0), "loss_curve": losses,
                            "nan_loss": any(l != l for l in losses), "lora": "r16 a16 all linear (language only)",
                            "compute_dtype": str(compute_dtype)}
    except Exception as e:  # OOM or a training error is a result here, not a crash
        R["qlora_smoke"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}", "steps": SMOKE_STEPS,
                            "steps_done": len(step_times), "compute_dtype": str(compute_dtype),
                            "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    save_results(SLUG, R)
print("QLORA", R["qlora_smoke"])"""),
code("""missing = bl.missing_sections(R, BASE)  # Run all again (after Restart session) fills what is missing
print("BASELINE_DONE" if not missing else f"BASELINE_INCOMPLETE {missing}", SLUG,
      json.dumps({k: R[k] for k in R if k not in ("greetings",)}, ensure_ascii=False))"""),
    ]


def train_cells() -> list:
    return [
md("""# fyM1 01 — QLoRA training (T4)
Set `MODEL_ID` (the lead's pick) and `RUN_ID`, then **Runtime → Run all**. Trains a LoRA adapter on the private
`Fan1000-Prog/fyM1-chats` data (every `*.jsonl`, rows `{"id", "messages"}`, plus `system_prompt.txt`) at a pinned
dataset revision. Everything of a run lives under `fyM1-ckpt/<RUN_ID>/`: each `checkpoint-<step>/` carries
`run_config.json` (refused on resume if the notebook's config differs), `session_progress.json` (wall time and
session count so far) and `data_stats.json`. Rerunning after a disconnect resumes the same run; a new lr or data
means a new `RUN_ID`. Training stops when eval loss rises 2 evals in a row (or is NaN) and writes
`<RUN_ID>/stopped.json`; reruns then skip straight to the final cell. The best checkpoint becomes `<RUN_ID>/final/`
and `final/`; the log goes to `<RUN_ID>/train_log.json` and `results/train_log.json`.
Runs on Colab or Kaggle (T4). Needs the `HF_TOKEN` secret (Colab: Secrets pane with notebook access; Kaggle: Add-ons → Secrets) and code synced from clean origin/main (`scripts/sync_code_to_hub.py`)."""),
code("""!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q -U unsloth unsloth_zoo"""),
code("""MODEL_ID = None  # set by the lead's model pick, e.g. "org/model"
RUN_ID = None    # e.g. "r1-lr2e-4"; one folder per run in fyM1-ckpt
LR, LR_SCHEDULER, WARMUP, EPOCHS, MAX_LEN = 2e-4, "cosine", 10, 2, 2048
BATCH, GRAD_ACCUM, SAVE_EVERY = 2, 4, 100
LORA_R, LORA_ALPHA = 16, 16
EVAL_SHARE, SEED = 0.02, 42
CKPT_REPO, DATA_REPO = "Fan1000-Prog/fyM1-ckpt", "Fan1000-Prog/fyM1-chats"
assert MODEL_ID and RUN_ID, "set MODEL_ID and RUN_ID\""""),
bootstrap_cell(),
code("""import sys, json, time, random, shutil
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download, list_repo_files

SESSION_START = time.time()
os.environ["HF_TOKEN"] = hf_token()  # never print it
api = HfApi()
code_dir = snapshot_download(CKPT_REPO, allow_patterns=["code/**"], local_dir=CODE_DIR)
sys.path.insert(0, f"{code_dir}/code/src")
CODE_SHA = open(f"{code_dir}/code/SHA").read().strip()
from fym1.train.resume import latest_checkpoint
from fym1.train.chat_format import CHAT_TEMPLATE_KWARGS, training_text, turn_markers
from fym1.train.supervise import best_eval, eval_losses, should_stop
from fym1.train.run_state import config_mismatch, session_progress
from fym1.runtime import output_dir, scratch_dir

OUTPUT, SCRATCH = output_dir(), scratch_dir()  # small wanted results vs big throwaway files
OUT = f"{SCRATCH}/out"  # checkpoints (pushed to the Hub, never kept as Kaggle output)

# Resume state for this run, read before anything trains.
repo_files = list_repo_files(CKPT_REPO)
resume = latest_checkpoint(CKPT_REPO, repo_files, run=RUN_ID)
STOPPED = f"{RUN_ID}/stopped.json" in repo_files
resumed_progress, saved_config = None, None
if resume:
    snapshot_download(CKPT_REPO, allow_patterns=[f"{resume}/*"], local_dir=OUT)
    saved_config = json.load(open(f"{OUT}/{resume}/run_config.json"))
    resumed_progress = json.load(open(f"{OUT}/{resume}/session_progress.json"))
DATA_REVISION = saved_config["data_revision"] if saved_config else api.dataset_info(DATA_REPO).sha
RUN_CONFIG = {"run_id": RUN_ID, "model": MODEL_ID, "lr": LR, "lr_scheduler": LR_SCHEDULER, "warmup_steps": WARMUP,
              "epochs": EPOCHS, "max_seq_len": MAX_LEN, "batch": BATCH, "grad_accum": GRAD_ACCUM,
              "save_every": SAVE_EVERY, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "eval_share": EVAL_SHARE,
              "seed": SEED, "data_revision": DATA_REVISION, "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
              "code_sha": CODE_SHA}  # a resume on other code is refused: no code sync to the Hub during a run
if saved_config:
    mismatch = config_mismatch(saved_config, RUN_CONFIG)
    assert not mismatch, f"{resume} belongs to a different run config: {mismatch}; use a new RUN_ID"
print("RUN", RUN_ID, "| resume from", resume or "scratch", "| stopped earlier:", STOPPED, "| code", CODE_SHA)"""),
code("""import torch

if not STOPPED:
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(MODEL_ID, max_seq_length=MAX_LEN, load_in_4bit=True,
                                                 full_finetuning=False)
    tok = getattr(processor, "tokenizer", processor)
    lora = dict(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0, bias="none", random_state=SEED,
                use_gradient_checkpointing="unsloth")
    if hasattr(model.config, "vision_config"):
        model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                         finetune_attention_modules=True, finetune_mlp_modules=True, **lora)
    else:
        model = FastModel.get_peft_model(model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                "gate_proj", "up_proj", "down_proj"], **lora)
    compute_dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
    print("model dtype", compute_dtype, "| vram GB", round(torch.cuda.memory_allocated() / 2**30, 2))"""),
code("""if not STOPPED:
    from datasets import Dataset

    def render(messages, add_generation_prompt=False):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                       **CHAT_TEMPLATE_KWARGS)

    files = [f for f in list_repo_files(DATA_REPO, repo_type="dataset", revision=DATA_REVISION) if f.endswith(".jsonl")]
    system_prompt = open(hf_hub_download(DATA_REPO, "system_prompt.txt", repo_type="dataset",
                                         revision=DATA_REVISION)).read().strip()
    rows = []
    for f in files:
        path = hf_hub_download(DATA_REPO, f, repo_type="dataset", revision=DATA_REVISION)
        rows += [json.loads(l) for l in open(path) if l.strip()]
    random.Random(SEED).shuffle(rows)
    texts = [training_text(render, [{"role": "system", "content": system_prompt}] + r["messages"]) for r in rows]
    n_eval = max(1, round(len(texts) * EVAL_SHARE))
    train_ds, eval_ds = Dataset.from_dict({"text": texts[n_eval:]}), Dataset.from_dict({"text": texts[:n_eval]})
    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    DATA_STATS = {"files": files, "revision": DATA_REVISION, "examples": len(texts), "train": len(train_ds),
                  "eval": n_eval, "tokens": sum(lengths), "over_max_len": sum(l > MAX_LEN for l in lengths)}
    print(DATA_STATS)"""),
code("""if not STOPPED:
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    from unsloth.chat_templates import train_on_responses_only

    class PushAndSupervise(TrainerCallback):
        \"\"\"Push every checkpoint (with run config, progress, data stats) to the Hub in one commit; stop when
        eval loss rises 2 evals in a row or is NaN, and put a stopped.json marker in the stop-step commit.\"\"\"

        stop_marker = None

        def on_save(self, args, state, control, **kw):
            name = f"checkpoint-{state.global_step}"
            folder = f"{args.output_dir}/{name}"
            extras = {"run_config.json": RUN_CONFIG, "data_stats.json": DATA_STATS,
                      "session_progress.json": session_progress(resumed_progress, time.time() - SESSION_START)}
            for fname, payload in extras.items():
                with open(f"{folder}/{fname}", "w") as f:
                    json.dump(payload, f, indent=2)
            ops = [CommitOperationAdd(path_in_repo=f"{RUN_ID}/{name}/{os.path.relpath(path, folder)}",
                                      path_or_fileobj=path)
                   for root, _, fnames in os.walk(folder) for path in (os.path.join(root, f) for f in fnames)]
            if self.stop_marker:
                ops.append(CommitOperationAdd(path_in_repo=f"{RUN_ID}/stopped.json",
                                              path_or_fileobj=json.dumps(self.stop_marker, indent=2).encode()))
            api.create_commit(repo_id=CKPT_REPO, operations=ops, commit_message=f"train {RUN_ID}: {name}")

        def on_evaluate(self, args, state, control, **kw):
            if should_stop([l for _, l in eval_losses(state.log_history)]):
                print("STOP: eval loss rose 2 evals in a row (or NaN)")
                control.should_training_stop = True
                control.should_save = True  # the stop-step checkpoint carries the marker (on_save)
                self.stop_marker = {"step": state.global_step, "reason": "eval loss rose 2 evals in a row or NaN",
                                    "eval_curve": eval_losses(state.log_history)}

    tok.padding_side = "right"
    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=train_ds, eval_dataset=eval_ds,
        callbacks=[PushAndSupervise()],
        args=SFTConfig(output_dir=f"{OUT}/{RUN_ID}", per_device_train_batch_size=BATCH,
                       per_device_eval_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,
                       num_train_epochs=EPOCHS, learning_rate=LR, lr_scheduler_type=LR_SCHEDULER,
                       warmup_steps=WARMUP, logging_steps=10, optim="adamw_8bit",
                       fp16=compute_dtype == torch.float16, bf16=False, max_length=MAX_LEN,
                       dataset_text_field="text", packing=False, eval_strategy="steps", eval_steps=SAVE_EVERY,
                       save_strategy="steps", save_steps=SAVE_EVERY, save_total_limit=2, report_to="none",
                       seed=SEED))
    instruction_part, response_part = turn_markers(render)
    trainer = train_on_responses_only(trainer, instruction_part=instruction_part, response_part=response_part)
    trainer.train(resume_from_checkpoint=f"{OUT}/{resume}" if resume else None)
    trainer.save_model(f"{OUT}/{RUN_ID}/last")  # final weights, also when stopped early; not resumable
    api.upload_folder(folder_path=f"{OUT}/{RUN_ID}/last", path_in_repo=f"{RUN_ID}/last", repo_id=CKPT_REPO,
                      commit_message=f"train {RUN_ID}: last")
    print("TRAIN_DONE step", trainer.state.global_step)"""),
code("""if STOPPED:  # trained in an earlier session: read its state from the newest checkpoint
    history = json.load(open(f"{OUT}/{resume}/trainer_state.json"))["log_history"]
    DATA_STATS = json.load(open(f"{OUT}/{resume}/data_stats.json"))
    progress, steps = resumed_progress, int(resume.rsplit("-", 1)[1])
else:
    history = trainer.state.log_history
    progress, steps = session_progress(resumed_progress, time.time() - SESSION_START), trainer.state.global_step
best = best_eval(history)
final_src = f"{RUN_ID}/checkpoint-{best[0]}" if best else f"{RUN_ID}/last"
local = snapshot_download(CKPT_REPO, allow_patterns=[f"{final_src}/*"], local_dir=f"{SCRATCH}/final_src")
adapter_files = [f for f in os.listdir(f"{local}/{final_src}") if f.startswith("adapter_") or f.startswith("tokenizer")]
assert any(f.startswith("adapter_") for f in adapter_files), f"no adapter files in {final_src}"
final_dir = f"{SCRATCH}/final"
shutil.rmtree(final_dir, ignore_errors=True)
os.makedirs(final_dir)
for f in adapter_files:
    shutil.copy(f"{local}/{final_src}/{f}", f"{final_dir}/{f}")
for dest in (f"{RUN_ID}/final", "final"):
    api.upload_folder(folder_path=final_dir, path_in_repo=dest, repo_id=CKPT_REPO,
                      commit_message=f"{dest} adapter = {final_src}")

losses = [h["loss"] for h in history if "loss" in h]
log = {
    "run_id": RUN_ID, "model": MODEL_ID, "code_sha": CODE_SHA, "gpu": torch.cuda.get_device_name(0),
    "steps": steps, "epochs": EPOCHS, "wall_time_s": progress["wall_time_s"], "sessions": progress["sessions"],
    "final_train_loss": losses[-1] if losses else None, "best_eval_loss": best[1] if best else None,
    "best_step": best[0] if best else None, "final_adapter": final_src,
    "stopped_early": f"{RUN_ID}/stopped.json" in list_repo_files(CKPT_REPO),
    "data": DATA_STATS,
    "config": {"lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lr": LR, "lr_scheduler": LR_SCHEDULER,
               "warmup_steps": WARMUP, "max_seq_len": MAX_LEN, "effective_batch": BATCH * GRAD_ACCUM,
               "data_revision": DATA_REVISION, "chat_template_kwargs": CHAT_TEMPLATE_KWARGS},
    "loss_curve": [{"step": h["step"], "loss": h["loss"]} for h in history if "loss" in h],
    "eval_curve": [{"step": s, "eval_loss": l} for s, l in eval_losses(history)],
}
body = json.dumps(log, indent=2).encode()
for dest in (f"{RUN_ID}/train_log.json", "results/train_log.json"):
    api.upload_file(path_or_fileobj=body, path_in_repo=dest, repo_id=CKPT_REPO, commit_message=f"train log {RUN_ID}")
with open(f"{OUTPUT}/train_log.json", "wb") as f:  # also a small run output (Kaggle: kernels output)
    f.write(body)
print("TRAIN_LOG", json.dumps({k: v for k, v in log.items() if not k.endswith("_curve")}))"""),
    ]


NOTEBOOKS = {"00_baseline": baseline_cells, "01_train": train_cells}


def build(name: str) -> str:
    """The notebook's exact file text; cell ids are derived from position so builds are byte-identical."""
    nb = nbf.v4.new_notebook(cells=NOTEBOOKS[name]())
    nb.metadata = METADATA
    for i, cell in enumerate(nb.cells):
        cell.id = f"{name}-{i:02d}".replace("_", "-")
    text = nbf.writes(nb)
    return text if text.endswith("\n") else text + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Colab/Kaggle notebooks.")
    parser.add_argument("--check", action="store_true", help="only report notebooks that differ")
    args = parser.parse_args(argv)
    stale = []
    for name in NOTEBOOKS:
        path = REPO_ROOT / "notebooks" / f"{name}.ipynb"
        text = build(name)
        if path.read_text() != text:
            stale.append(path.name)
            if not args.check:
                path.write_text(text)
    if args.check and stale:
        print("out of date:", ", ".join(stale), "(run python scripts/build_notebooks.py)")
        return 1
    print("rebuilt:" if stale else "up to date", ", ".join(stale))
    return 0


if __name__ == "__main__":
    sys.exit(main())
