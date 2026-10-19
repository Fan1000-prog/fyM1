"""Build notebooks/00_baseline.ipynb, 01_train.ipynb, 02_eval_export.ipynb and 04_synth.ipynb.
This script is their source of truth.

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


ONE_GPU = 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Kaggle\'s T4 runtime may preset "0,1"; always use one GPU\n'
TWO_GPUS = ('os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # both of Kaggle\'s T4s: a 27B model in 4-bit needs ~17 GB; '
            "a single-GPU runtime just sees GPU 0\n")


def bootstrap_cell(gpus: str = ONE_GPU):
    return code(
        "# Runtime bootstrap (Colab or Kaggle). The HF token is needed before fym1 can be downloaded, so this cell\n"
        "# carries a verbatim copy of fym1.runtime.hf_token and its helpers (checked by tests/test_notebook_runtime.py).\n"
        "import os\nimport tempfile\n\n"
        + gpus
        + 'CODE_DIR = tempfile.mkdtemp(prefix="fym1code-")  # fresh each run: nothing stale on sys.path\n\n'
        + "\n\n\n".join(runtime_segments()))


def baseline_cells() -> list:
    return [
md("""# fy-m1 00 — base-model baseline (T4)
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
from fym1.train.hardware import precision_flags
from fym1.train.run_state import check_hub_id
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
check_hub_id(f"saved results/baseline/{SLUG}.json hub_id", R["model"]["hub_id"], HUB_ID)  # new id: new slug
print("CANDIDATE", SLUG, "| missing:", bl.missing_sections(R, BASE))"""),
code("""import torch
from unsloth import FastModel

RETRIED = {"items": 0}  # items re-run one at a time after an OOM batch, in the current section


def section(name):
    print(f"== {name}: {'skip (saved)' if name in R else 'run'}")
    RETRIED["items"] = 0
    return name not in R

t0 = time.time()
try:
    # use_exact_model_name: else Unsloth's name map swaps google/gemma-3-12b-it for its own bnb-4bit checkpoint
    model, processor = FastModel.from_pretrained(HUB_ID, max_seq_length=2048, load_in_4bit=True, full_finetuning=False,
                                                 use_exact_model_name=True)
except Exception as e:  # gated/403 or unsupported architecture: record and move on
    R["skipped"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}  # missing_sections() then treats it as complete
    save_results(SLUG, R)
    raise
check_hub_id("loaded model", model.config._name_or_path, HUB_ID)
tok = getattr(processor, "tokenizer", processor)
tok.padding_side = "left"
R["env"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "model_dtype": str(model.dtype),
            "loaded_from": model.config._name_or_path, "code_sha": CODE_SHA,
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
    def count(n):
        RETRIED["items"] += n

    return run_batched(items, batch_size, run, is_oom=lambda e: isinstance(e, torch.OutOfMemoryError),
                       clear=torch.cuda.empty_cache, on_retry=count)


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
                         "seconds": round(time.time() - t0), "protocol": protocol, "retried_items": RETRIED["items"]}
    except StillOutOfMemory:
        R["belebele"] = {"oom": True, "protocol": protocol, "retried_items": RETRIED["items"]}
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
                                    "seconds": round(time.time() - t0), "protocol": protocol,
                                    "retried_items": RETRIED["items"]}
    except StillOutOfMemory:
        R["belebele_completion"] = {"oom": True, "protocol": protocol, "retried_items": RETRIED["items"]}
    save_results(SLUG, R)
print("BELEBELE_COMPLETION", R.get("belebele_completion"))"""),
code("""if section("flores"):
    t0 = time.time()
    protocol = "devtest first 200, 0-shot chat prompt, greedy, max 256 new tokens, first line kept"
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
                       "seconds": round(time.time() - t0), "protocol": protocol, "retried_items": RETRIED["items"]}
        save_results(SLUG, R)
    except StillOutOfMemory:
        R["flores"] = {"oom": True, "protocol": protocol, "retried_items": RETRIED["items"]}
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
        R["greetings"] = {"source": source, "sampling": sampling, "retried_items": RETRIED["items"],
                          "answers": [{"prompt": p, "answer": a.strip()} for p, a in zip(greetings, answers)]}
    except StillOutOfMemory:
        R["greetings"] = {"oom": True, "source": source, "sampling": sampling, "retried_items": RETRIED["items"]}
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
                                        **precision_flags(str(compute_dtype)), max_length=2048,
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
md("""# fy-m1 01 — QLoRA training (A100 or T4)
Set `MODEL_ID` (the lead's pick) and `RUN_ID`, then **Runtime → Run all**. Trains a LoRA adapter on the private
`Fan1000-Prog/fyM1-chats` data (every `*.jsonl`, rows `{"id", "messages"}`, plus `system_prompt.txt`) at a pinned
dataset revision. Everything of a run lives under `fyM1-ckpt/<RUN_ID>/`: each `checkpoint-<step>/` carries
`run_config.json` (refused on resume if the notebook's config differs), `session_progress.json` (wall time and
session count so far) and `data_stats.json`. Rerunning after a disconnect resumes the same run; a new lr or data
means a new `RUN_ID`. Training stops when eval loss rises 2 evals in a row (or is NaN) and writes
`<RUN_ID>/stopped.json`; reruns then skip straight to the final cell. The best checkpoint becomes `<RUN_ID>/final/`
and `final/`; the log goes to `<RUN_ID>/train_log.json` and `results/train_log.json`.
Runs on Colab A100 (8 rows x 1 step, bf16) or a Colab/Kaggle T4 (2 rows x 4 steps, fp16): the effective batch is 8
on both. Needs the `HF_TOKEN` secret (Colab: Secrets pane with notebook access; Kaggle: Add-ons → Secrets) and code synced from clean origin/main (`scripts/sync_code_to_hub.py`)."""),
code("""!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q -U unsloth unsloth_zoo"""),
code("""MODEL_ID = None  # set by the lead's model pick, e.g. "org/model"
RUN_ID = None    # e.g. "r1-lr2e-4"; one folder per run in fyM1-ckpt
LR, LR_SCHEDULER, WARMUP, EPOCHS, MAX_LEN = 2e-4, "cosine", 10, 2, 2048
SAVE_EVERY = 100  # BATCH x GRAD_ACCUM is picked by GPU below (effective batch 8 everywhere)
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
from fym1.train.run_state import check_hub_id, config_mismatch, session_progress
from fym1.train.hardware import batch_shape, precision_flags
from fym1.runtime import output_dir, scratch_dir
import torch

GPU = torch.cuda.get_device_name(0)
BATCH, GRAD_ACCUM = batch_shape(GPU)
HARDWARE = {"gpu": GPU, "compute_dtype": None}  # dtype is filled in once the model is loaded
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
              "code_sha": CODE_SHA, "hardware": HARDWARE}  # resume on other code is refused: no code sync during a run
if saved_config:
    mismatch = config_mismatch(saved_config, RUN_CONFIG)
    assert not mismatch, f"{resume} belongs to a different run config: {mismatch}; use a new RUN_ID"
print("RUN", RUN_ID, "| resume from", resume or "scratch", "| stopped earlier:", STOPPED, "| code", CODE_SHA,
      "| gpu", GPU, "| batch", BATCH, "x grad-accum", GRAD_ACCUM)"""),
code("""import torch

LOADED_FROM = None  # what this session loaded; stays None when the run stopped in an earlier session
if not STOPPED:
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(MODEL_ID, max_seq_length=MAX_LEN, load_in_4bit=True,
                                                 full_finetuning=False, use_exact_model_name=True)
    LOADED_FROM = model.config._name_or_path
    check_hub_id("loaded model", LOADED_FROM, MODEL_ID)
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
    HARDWARE["compute_dtype"] = str(compute_dtype)
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
                       **precision_flags(str(compute_dtype)), max_length=MAX_LEN,
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
    "run_id": RUN_ID, "model": MODEL_ID, "loaded_from": LOADED_FROM, "code_sha": CODE_SHA,
    "gpu": torch.cuda.get_device_name(0),
    "steps": steps, "epochs": EPOCHS, "wall_time_s": progress["wall_time_s"], "sessions": progress["sessions"],
    "final_train_loss": losses[-1] if losses else None, "best_eval_loss": best[1] if best else None,
    "best_step": best[0] if best else None, "final_adapter": final_src,
    "stopped_early": f"{RUN_ID}/stopped.json" in list_repo_files(CKPT_REPO),
    "data": DATA_STATS,
    "config": {"lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lr": LR, "lr_scheduler": LR_SCHEDULER,
               "warmup_steps": WARMUP, "max_seq_len": MAX_LEN, "batch": BATCH, "grad_accum": GRAD_ACCUM,
               "effective_batch": BATCH * GRAD_ACCUM,
               "hardware": (saved_config if STOPPED else RUN_CONFIG).get("hardware"),
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


def synth_cells() -> list:
    return [
md("""# fy-m1 04 — synthetic everyday chats (T4 x2)
Few-shot prompts an open-weight model with the real, anonymised chats in the private `Fan1000-Prog/fyM1-chats` and
keeps only chats that pass `fym1.data.synth.filter_conversations` (Malagasy assistant turns, length bounds, no
placeholders or unknown names after anonymising, no 8-word run shared with any real chat, no duplicates).
**Runtime → Run all.** Raw outputs go to `fyM1-chats` `synth/raw/part-<first index>.ndjson` every `UPLOAD_EVERY`
chats, so a rerun after a disconnect resumes where the last part ended. Each run then filters everything generated
so far and writes `synth/filtered.ndjson` and counts-only `synth/synth_stats.json`
(fetch it with `scripts/build_synth_stats.py`). Nothing here prints chat text.

`.ndjson`, not `.jsonl`: `01_train` trains on every `*.jsonl` in the repo, so these files stay out of training until
the sample check; then `PROMOTE = True` writes `synth/synth_chats.jsonl`.
Generator: the first entry of `GENERATORS` that loads fully on the GPUs and passes a one-chat smoke check; a part
that hits a float16 inf/nan moves to the next entry and is redone (the stats record which generators ran, with their
licences). Generators are permissive only (owner decision 2026-10-18: fy-m1 ships under a simple open licence):
`synth.generator_licence` refuses any other model, or a model card whose licence changed, before anything loads.
Kaggle: Accelerator **GPU T4 x2**, Internet on, `HF_TOKEN` secret on (no gated-model licence to accept). Code comes from `fyM1-ckpt` `code/` (`scripts/sync_code_to_hub.py`)."""),
code("""!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q -U transformers accelerate bitsandbytes fasttext-numpy2-wheel ftfy"""),
code("""GENERATORS = [  # (hub id, dtype), tried in order; permissive licences only (synth.GENERATOR_LICENCES)
    # float16 is ~2x faster on a T4 (no bfloat16) and ran the 9B's baseline; float32 avoids float16 inf/nan.
    # Gemma 4 runs float32: on a T4 Unsloth keeps gemma4 in float32 (00_baseline, 01_train).
    ("McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1", "float16"),
    ("McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1", "float32"),
    ("google/gemma-4-12B-it", "float32"),
]
TARGET_RAW = 2500         # raw chats to generate; the goal is >= 2,000 kept after filtering
FEW_SHOT_K, SEED = 3, 0   # real examples per prompt; SEED fixes the seed grid, example picks and sampling
GEN_BATCH, MAX_NEW_TOKENS, TEMPERATURE, TOP_P = 8, 400, 0.9, 0.95
UPLOAD_EVERY = 48         # chats per raw part on the Hub: a disconnect loses at most this many
MAX_HOURS = 11.0          # stop generating after this (Kaggle sessions end at 12 h), then filter and save stats
WEIGHTS_GIB_PER_GPU = 11  # weights cap per T4 (15 GB); the rest is left for the KV cache
REAL_FILES = ["chats.jsonl"]  # real chats in fyM1-chats: few-shot examples and the 8-word leak reference
PROMOTE = False           # True only after the sample check: also writes synth/synth_chats.jsonl (01_train input)
CKPT_REPO, DATA_REPO = "Fan1000-Prog/fyM1-ckpt", "Fan1000-Prog/fyM1-chats"
RAW_PREFIX = "synth/raw/\""""),
bootstrap_cell(TWO_GPUS),
code("""import sys, json, time, datetime
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download

SESSION_START = time.time()
os.environ["HF_TOKEN"] = hf_token()  # never print it
api = HfApi()
code_dir = snapshot_download(CKPT_REPO, allow_patterns=["code/**"], local_dir=CODE_DIR)
sys.path.insert(0, f"{code_dir}/code/src")
CODE_SHA = open(f"{code_dir}/code/SHA").read().strip()
from fym1.data import synth
from fym1.eval.batching import run_batched
from fym1.runtime import output_dir, scratch_dir
from fym1.train.chat_format import CHAT_TEMPLATE_KWARGS

OUTPUT, SCRATCH = output_dir(), scratch_dir()
DATA_REVISION = api.dataset_info(DATA_REPO).sha
real = []
for f in REAL_FILES:  # private chat text: kept in memory only, never printed or saved here
    path = hf_hub_download(DATA_REPO, f, repo_type="dataset", revision=DATA_REVISION)
    real += [json.loads(l) for l in open(path) if l.strip()]
seeds = synth.seed_grid(TARGET_RAW, SEED)


def raw_rows():
    files = [f for f in list_repo_files(DATA_REPO, repo_type="dataset") if f.startswith(RAW_PREFIX)]
    if not files:
        return []
    local = snapshot_download(DATA_REPO, repo_type="dataset", allow_patterns=[f"{RAW_PREFIX}*"],
                              local_dir=f"{SCRATCH}/synth")
    return [json.loads(l) for f in sorted(files) for l in open(f"{local}/{f}") if l.strip()]


LICENCES = {g: synth.generator_licence(g, card_licence=getattr(api.model_info(g).card_data, "license", None))
            for g, _ in GENERATORS}  # refuses a non-permissive generator before anything runs
existing = raw_rows()
for g in sorted({row["generator"] for row in existing}):  # parts from an earlier, refused generator
    synth.generator_licence(g)  # raises: delete synth/raw/ in fyM1-chats and rerun
done = {row["index"] for row in existing}
todo = [i for i in range(TARGET_RAW) if i not in done]
print("real chats:", len(real), "| raw done:", len(done), "| to do:", len(todo), "| code", CODE_SHA)"""),
code("""import gc
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

GPUS = torch.cuda.device_count()


def load(hub_id, dtype):
    torch_dtype = getattr(torch, dtype)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch_dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        hub_id, quantization_config=quant, torch_dtype=torch_dtype, device_map="auto",
        max_memory={i: f"{WEIGHTS_GIB_PER_GPU}GiB" for i in range(GPUS)})
    off_gpu = {d for d in model.hf_device_map.values() if not isinstance(d, int)}
    if off_gpu:  # CPU/disk offload would be far too slow: treat as "does not fit"
        raise MemoryError(f"needs offload to {sorted(map(str, off_gpu))}")
    tok = AutoTokenizer.from_pretrained(hub_id)
    tok.padding_side = "left"
    return model.eval(), tok


def prompt_text(i):
    examples = [r["messages"] for r in synth.examples_for(i, real, FEW_SHOT_K, SEED)]
    chat = [{"role": "user", "content": synth.build_prompt(seeds[i], examples)}]
    return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                   **CHAT_TEMPLATE_KWARGS)  # thinking off: Qwen3.5 and Gemma 4 can think


@torch.inference_mode()
def generate(indices):
    \"\"\"(output text, new tokens) per seed index; batches that run out of memory are retried one by one.\"\"\"
    def run(batch):
        enc = tok([prompt_text(i) for i in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE,
                             top_p=TOP_P, pad_token_id=tok.pad_token_id)
        new = ids[:, enc["input_ids"].shape[1]:]
        counts = (new != tok.pad_token_id).sum(dim=1).tolist()
        return list(zip(tok.batch_decode(new, skip_special_tokens=True), counts))

    return run_batched(indices, GEN_BATCH, run, is_oom=lambda e: isinstance(e, torch.OutOfMemoryError),
                       clear=torch.cuda.empty_cache)


def free_model():
    global model, tok
    model = tok = None
    gc.collect()
    torch.cuda.empty_cache()


def load_next_generator():
    \"\"\"Load the next GENERATORS entry that fits on the GPUs and passes a smoke check; raise when none is left.\"\"\"
    global model, tok, GEN_ID, DTYPE, NEXT_GENERATOR
    while NEXT_GENERATOR < len(GENERATORS):
        GEN_ID, DTYPE = GENERATORS[NEXT_GENERATOR]
        NEXT_GENERATOR += 1
        try:
            model, tok = load(GEN_ID, DTYPE)
            torch.manual_seed(SEED)
            smoke = generate(todo[:2])  # two chats, so one unlucky sample does not skip a good model
            if any(synth.parse_conversation(text) for text, _ in smoke):  # outputs never printed
                print("GENERATOR", GEN_ID, DTYPE, LICENCES[GEN_ID], "| skipped:", LOAD_LOG,
                      "| VRAM GB", [round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(GPUS)])
                return
            LOAD_LOG.append(f"{GEN_ID} {DTYPE}: smoke output did not parse")
        except Exception as e:  # does not fit, float16 overflow (inf/nan in sampling), ...
            LOAD_LOG.append(f"{GEN_ID} {DTYPE}: {type(e).__name__}: {str(e)[:200]}")
        free_model()
    raise RuntimeError(f"no generator left: {LOAD_LOG}")


model = tok = None
LOAD_LOG = []  # counts-only reasons a generator was skipped
NEXT_GENERATOR = 0  # index of the next GENERATORS entry to try
GPU_NAMES = f"{GPUS}x {torch.cuda.get_device_name(0)}"
if todo:
    load_next_generator()"""),
code("""start = 0
while start < len(todo):
    if time.time() - SESSION_START > MAX_HOURS * 3600:
        print("time budget reached; filtering what exists")
        break
    part = todo[start:start + UPLOAD_EVERY]
    torch.manual_seed(SEED * 1_000_003 + part[0])  # same part, same sampling on a rerun
    t0 = time.time()
    try:
        results = generate(part)
    except RuntimeError as e:
        if not synth.is_sampling_overflow(e):
            raise
        LOAD_LOG.append(f"{GEN_ID} {DTYPE}: inf/nan in sampling at part {part[0]:05d}")
        free_model()
        load_next_generator()  # then redo this part
        continue
    seconds = (time.time() - t0) / len(part)
    rows = [{"index": i, "seed": seeds[i].as_dict(), "generator": GEN_ID, "dtype": DTYPE,
             "licence": LICENCES[GEN_ID], "code_sha": CODE_SHA, "data_revision": DATA_REVISION, "output": text,
             "new_tokens": n, "gen_seconds": round(seconds, 2)}
            for i, (text, n) in zip(part, results)]
    body = "".join(json.dumps(r, ensure_ascii=False) + "\\n" for r in rows).encode()
    api.upload_file(path_or_fileobj=body, path_in_repo=f"{RAW_PREFIX}part-{part[0]:05d}.ndjson", repo_id=DATA_REPO,
                    repo_type="dataset", commit_message=f"synth raw {part[0]}-{part[-1]}")
    print(f"part {part[0]:05d}: {len(rows)} chats, {sum(r['new_tokens'] for r in rows)} new tokens, "
          f"{seconds:.1f} s/chat | {len(done) + start + len(part)}/{TARGET_RAW}")
    start += UPLOAD_EVERY"""),
code("""from fym1.eval.lang import is_malagasy

rows = sorted(raw_rows(), key=lambda r: r["index"])
kept, stats = synth.filter_raw_rows(  # same step as scripts/filter_synth.py
    rows, real, is_malagasy,
    meta={"seed": SEED, "few_shot_k": FEW_SHOT_K, "target_raw": TARGET_RAW, "gpus": GPU_NAMES if todo else None,
          "code_sha": CODE_SHA, "data_revision": DATA_REVISION})
uploads = {"synth/filtered.ndjson": "".join(json.dumps(r, ensure_ascii=False) + "\\n" for r in kept).encode(),
           "synth/synth_stats.json": (json.dumps(stats, indent=2) + "\\n").encode()}
if PROMOTE:
    uploads["synth/synth_chats.jsonl"] = uploads["synth/filtered.ndjson"]
for dest, body in uploads.items():
    api.upload_file(path_or_fileobj=body, path_in_repo=dest, repo_id=DATA_REPO, repo_type="dataset",
                    commit_message=f"synth filtered: {len(kept)}/{len(rows)} kept")
with open(f"{OUTPUT}/synth_stats.json", "wb") as f:  # counts only (Kaggle: kernels output)
    f.write(uploads["synth/synth_stats.json"])
print("SYNTH_STATS", json.dumps(stats))"""),
    ]


def eval_cells() -> list:
    return [
md("""# fy-m1 02 — eval export (A100)
Scores one finished training run. Set `RUN_ID`, pick an **A100** runtime (Runtime → Change runtime type), then
**Runtime → Run all**. Reads `fyM1-ckpt/<RUN_ID>/train_log.json` for the base model and the data revision, loads the
base model in 4-bit with the LoRA adapter from `<RUN_ID>/final/`, and measures, in this order:
1. answers to every `eval/prompts.jsonl` prompt from the base model (adapter off) and from fy-m1, plus fy-m1's answers
   to the extra identity prompts (`eval/identity_check.jsonl`, set "identity"): data `system_prompt.txt`, temp 0.7,
   top_p 0.9, max 256 new tokens, seed 42;
2. Belebele `plt_Latn` for fy-m1 (both 00_baseline protocols: plain-text completion is `belebele_acc`);
3. FLORES+ `plt_Latn`↔`eng_Latn` devtest first 200, chrF++, for fy-m1 (00_baseline protocol);
4. held-out-chat perplexity on 01_train's 2% eval split (reply tokens only, as eval loss), base and fy-m1.

Everything goes to the **private** `fyM1-ckpt` under `results/eval/`: `answers_base.jsonl`, `answers_fym1.jsonl`,
`answers_fym1_identity.jsonl` (private, never in git), `notebook_metrics.json` (for `scripts/score.py scores`) and
`eval_run.json` (protocols, counts, timings). The base model's Belebele and chrF++ come from its 00_baseline file
(`results/baseline/<slug>.json`, same protocols), so they are not rerun here. `leak_rate` is measured on the owner's
laptop (`scripts/score.py leaks`). Nothing here prints answer or chat text. A rerun skips sections already saved for the
same `RUN_ID` and code; set `FORCE = True` to redo all.
Follow-up, not in this notebook: merge the LoRA, save 4-bit and export GGUF Q4_K_M.
Needs the `HF_TOKEN` secret (Colab Secrets pane with notebook access) and code synced to `fyM1-ckpt` `code/`
(`scripts/sync_code_to_hub.py`)."""),
code("""!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q -U unsloth unsloth_zoo sacrebleu fasttext-numpy2-wheel"""),
code("""RUN_ID = "fym1-9b-a100-r1"  # the training run to score (fyM1-ckpt/<RUN_ID>/)
CKPT_REPO, DATA_REPO = "Fan1000-Prog/fyM1-ckpt", "Fan1000-Prog/fyM1-chats"
OUT_PREFIX = "results/eval"  # in the private fyM1-ckpt; answer files never go to git
SEED, TEMPERATURE, TOP_P, MAX_NEW_TOKENS = 42, 0.7, 0.9, 256
EVAL_SHARE, SPLIT_SEED = 0.02, 42  # 01_train's eval split (checked against train_log.json counts)
GEN_BATCH, BELEBELE_BATCH, PPL_BATCH, FLORES_N = 16, 8, 2, 200
FORCE = False  # True: redo every section; else sections saved for this RUN_ID and code are skipped"""),
bootstrap_cell(),
code("""import sys, json, time, datetime, contextlib
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

os.environ["HF_TOKEN"] = hf_token()  # never print it
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # perplexity needs the full logits
api = HfApi()
assert api.repo_info(CKPT_REPO).private, f"{CKPT_REPO} must be private: the answer files go there"
code_dir = snapshot_download(CKPT_REPO, allow_patterns=["code/**"], local_dir=CODE_DIR)
sys.path.insert(0, f"{code_dir}/code/src")
CODE_SHA = open(f"{code_dir}/code/SHA").read().strip()
from fym1.eval import baseline as bl
from fym1.eval.batching import StillOutOfMemory, run_batched
from fym1.eval.identity_check import check_specs, export_prompts
from fym1.eval.perplexity import held_out_rows, perplexity, response_spans, token_mask
from fym1.eval.scoring import NOTEBOOK_METRICS
from fym1.train.chat_format import CHAT_TEMPLATE_KWARGS, training_text, turn_markers
from fym1.train.run_state import check_hub_id
from fym1.runtime import output_dir, scratch_dir

OUTPUT, SCRATCH = output_dir(), scratch_dir()


def hub_json(path):
    try:
        return json.load(open(hf_hub_download(CKPT_REPO, path, force_download=True)))
    except EntryNotFoundError:
        return None


def jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


TRAIN_LOG = hub_json(f"{RUN_ID}/train_log.json")
assert TRAIN_LOG, f"no {RUN_ID}/train_log.json in {CKPT_REPO}: finish 01_train first"
MODEL_ID, DATA_REVISION = TRAIN_LOG["model"], TRAIN_LOG["config"]["data_revision"]
MAX_LEN = TRAIN_LOG["config"]["max_seq_len"]
SYSTEM_PROMPT = open(hf_hub_download(DATA_REPO, "system_prompt.txt", repo_type="dataset",
                                     revision=DATA_REVISION)).read().strip()
PROMPTS = [{"id": r["id"], "prompt": r["prompt"]} for r in jsonl(f"{code_dir}/code/eval/prompts.jsonl")]
SPECS = jsonl(f"{code_dir}/code/eval/identity_check.jsonl")
check_specs(SPECS)
IDENTITY = export_prompts(SPECS)
assert len({r["id"] for r in PROMPTS + IDENTITY}) == len(PROMPTS) + len(IDENTITY), "prompt ids must be unique"

RUN = hub_json(f"{OUT_PREFIX}/eval_run.json") or {}
if FORCE or RUN.get("run_id") != RUN_ID or RUN.get("code_sha") != CODE_SHA:
    RUN = {}  # another run or other code: start over
RUN.update(run_id=RUN_ID, model=MODEL_ID, adapter=f"{RUN_ID}/final", final_adapter=TRAIN_LOG.get("final_adapter"),
           data_revision=DATA_REVISION, code_sha=CODE_SHA)
RETRIED = {"items": 0}  # items re-run one at a time after an OOM batch, in the current section


def upload(name, body, what):
    api.upload_file(path_or_fileobj=body, path_in_repo=f"{OUT_PREFIX}/{name}", repo_id=CKPT_REPO,
                    commit_message=f"eval {RUN_ID}: {what}")


def save_run(what):
    body = (json.dumps(RUN, ensure_ascii=False, indent=2) + "\\n").encode()
    upload("eval_run.json", body, what)
    with open(f"{OUTPUT}/eval_run.json", "wb") as f:
        f.write(body)


def section(name):
    print(f"== {name}: {'skip (saved)' if name in RUN else 'run'}")
    RETRIED["items"] = 0
    return name not in RUN


print("RUN", RUN_ID, "| model", MODEL_ID, "| data", DATA_REVISION[:9], "| code", CODE_SHA[:9], "| prompts", len(PROMPTS),
      "| identity prompts", len(IDENTITY), "| saved sections", sorted(k for k in RUN if isinstance(RUN[k], dict)))"""),
code("""import torch
from unsloth import FastModel
from peft import PeftModel

t0 = time.time()
# use_exact_model_name: else Unsloth's name map may swap the id for its own pre-quantized checkpoint (as in 01_train)
model, processor = FastModel.from_pretrained(MODEL_ID, max_seq_length=MAX_LEN, load_in_4bit=True,
                                             full_finetuning=False, use_exact_model_name=True)
check_hub_id("loaded model", model.config._name_or_path, MODEL_ID)
local = snapshot_download(CKPT_REPO, allow_patterns=[f"{RUN_ID}/final/*"], local_dir=f"{SCRATCH}/adapter")
model = PeftModel.from_pretrained(model, f"{local}/{RUN_ID}/final")
FastModel.for_inference(model)
tok = getattr(processor, "tokenizer", processor)
tok.padding_side = "left"
RUN["env"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "model_dtype": str(model.dtype),
              "quant": "bnb 4-bit nf4 (Unsloth load_in_4bit)", "load_s": round(time.time() - t0),
              "vram_after_load_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
              "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
print(RUN["env"])


def adapter(on):
    \"\"\"fy-m1 with the LoRA on; the base model with it off.\"\"\"
    return contextlib.nullcontext() if on else model.disable_adapter()


def chat(prompt, system=None):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)


def encode(texts, special_tokens=False):
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=special_tokens).to("cuda")


def batched(items, batch_size, run):
    \"\"\"On CUDA OOM: empty the cache and retry that batch one item at a time; StillOutOfMemory if that fails.\"\"\"
    def count(n):
        RETRIED["items"] += n

    return run_batched(items, batch_size, run, is_oom=lambda e: isinstance(e, torch.OutOfMemoryError),
                       clear=torch.cuda.empty_cache, on_retry=count)


@torch.inference_mode()
def generate(prompts, max_new_tokens, sample, system=None):
    kw = dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P) if sample else dict(do_sample=False)

    def run(batch):
        enc = encode([chat(p, system) for p in batch])
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        return tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return batched(prompts, GEN_BATCH, run)


@torch.inference_mode()
def last_logits(enc):
    try:
        return model(**enc, logits_to_keep=1).logits[:, -1]
    except TypeError:  # forward without logits_to_keep
        return model(**enc).logits[:, -1]"""),
code("""SAMPLING = (f"system prompt: data system_prompt.txt @ {DATA_REVISION[:9]}, temp {TEMPERATURE}, top_p {TOP_P}, "
            f"max {MAX_NEW_TOKENS} new tokens, torch seed {SEED} per file, batch {GEN_BATCH}")
ANSWER_FILES = {  # score.py reads answers_<model>.jsonl; identity_check.py reads the fym1 pair
    "answers_base": (PROMPTS, False),
    "answers_fym1": (PROMPTS, True),
    "answers_fym1_identity": (IDENTITY, True),
}
for name, (items, lora_on) in ANSWER_FILES.items():
    if not section(name):
        continue
    t0 = time.time()
    torch.manual_seed(SEED)
    with adapter(lora_on):
        replies = generate([r["prompt"] for r in items], MAX_NEW_TOKENS, True, SYSTEM_PROMPT)
    rows = [{"id": r["id"], "prompt": r["prompt"], "response": a.strip()} for r, a in zip(items, replies)]
    upload(f"{name}.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\\n" for r in rows).encode(),
           f"{name} ({len(rows)} answers)")
    RUN[name] = {"n": len(rows), "empty": sum(not r["response"] for r in rows), "adapter": lora_on,
                 "sampling": SAMPLING, "seconds": round(time.time() - t0), "retried_items": RETRIED["items"]}
    save_run(name)
    print(name, RUN[name])  # counts only, never answer text"""),
code("""from datasets import load_dataset

PROTOCOLS = {  # the two 00_baseline protocols; "belebele_completion" is the belebele_acc everyone is compared on
    "belebele_completion": "0-shot plain-text completion (P/Q/A-D/Answer:), no chat template, argmax of next-token "
                           "logits over ' A'-' D'",
    "belebele": "0-shot, chat template, English instruction, argmax of next-token logits over '1'-'4'",
}
for name, protocol in PROTOCOLS.items():
    if not section(name):
        continue
    t0 = time.time()
    rows = load_dataset("facebook/belebele", "plt_Latn", split="test")
    if name == "belebele_completion":
        prompts, special = [bl.belebele_completion_prompt(r) for r in rows], True
        option_ids = [tok.encode(" " + letter, add_special_tokens=False)[-1] for letter in "ABCD"]
    else:
        prompts, special = [chat(bl.belebele_prompt(r)) for r in rows], False
        option_ids = [tok.encode(d, add_special_tokens=False)[-1] for d in "1234"]
    try:
        scores = torch.tensor(batched(prompts, BELEBELE_BATCH,
                                      lambda b: last_logits(encode(b, special))[:, option_ids].float().tolist()))
        RUN[name] = {"acc": round(bl.accuracy((scores.argmax(-1) + 1).tolist(), rows["correct_answer_num"]), 4),
                     "n": len(scores), "nan_rows": int(torch.isnan(scores).any(-1).sum()),
                     "seconds": round(time.time() - t0), "protocol": protocol, "retried_items": RETRIED["items"]}
    except StillOutOfMemory:
        RUN[name] = {"oom": True, "protocol": protocol, "retried_items": RETRIED["items"]}
    save_run(name)
    print(name.upper(), RUN[name])"""),
code("""if section("flores"):
    t0 = time.time()
    protocol = "devtest first 200, 0-shot chat prompt, greedy, max 256 new tokens, first line kept"
    try:
        mg = load_dataset("openlanguagedata/flores_plus", "plt_Latn", split="devtest")
        en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
        mg_txt, en_txt = mg["text"][:FLORES_N], en["text"][:FLORES_N]
        if "id" in mg.column_names:
            assert mg["id"][:FLORES_N] == en["id"][:FLORES_N], "FLORES rows not aligned"
        hyp_mg = [bl.first_line(t) for t in generate([bl.translation_prompt(s, "English", "Malagasy") for s in en_txt], 256, False)]
        hyp_en = [bl.first_line(t) for t in generate([bl.translation_prompt(s, "Malagasy", "English") for s in mg_txt], 256, False)]
        RUN["flores"] = {"eng-plt_chrf++": round(bl.chrf_pp(hyp_mg, mg_txt), 2),
                         "plt-eng_chrf++": round(bl.chrf_pp(hyp_en, en_txt), 2), "n": FLORES_N,
                         "seconds": round(time.time() - t0), "protocol": protocol, "retried_items": RETRIED["items"]}
        save_run("flores")
    except StillOutOfMemory:
        RUN["flores"] = {"oom": True, "protocol": protocol, "retried_items": RETRIED["items"]}
        save_run("flores")
    except Exception as e:  # gated dataset not accepted yet: leave unsaved so a rerun retries it
        print("FLORES PENDING:", type(e).__name__, str(e)[:300])
print("FLORES", RUN.get("flores", "pending"))"""),
code("""import torch.nn.functional as F


def render(messages, add_generation_prompt=False):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   **CHAT_TEMPLATE_KWARGS)


@torch.inference_mode()
def reply_nll(texts, spans):
    \"\"\"(summed NLL, scored tokens) per text over reply tokens only, cut at MAX_LEN like training.\"\"\"
    def run(batch):
        enc = tok([texts[i] for i in batch], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN,
                  add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        keep = torch.tensor([token_mask(o, spans[i]) for i, o in zip(batch, offsets)])[:, 1:].to("cuda")
        enc = enc.to("cuda")
        logits = model(**enc).logits[:, :-1].float()
        nll = F.cross_entropy(logits.transpose(1, 2), enc["input_ids"][:, 1:], reduction="none")
        return [(float(n[k].sum()), int(k.sum())) for n, k in zip(nll, keep)]

    return batched(list(range(len(texts))), PPL_BATCH, run)


if section("perplexity"):
    t0 = time.time()
    files = [f for f in list_repo_files(DATA_REPO, repo_type="dataset", revision=DATA_REVISION) if f.endswith(".jsonl")]
    rows = []
    for f in files:  # same files and order as 01_train; private chat text stays in memory, never printed
        rows += jsonl(hf_hub_download(DATA_REPO, f, repo_type="dataset", revision=DATA_REVISION))
    held = held_out_rows(rows, EVAL_SHARE, SPLIT_SEED)
    assert (len(rows), len(held)) == (TRAIN_LOG["data"]["examples"], TRAIN_LOG["data"]["eval"]), \\
        "eval split differs from training: check data_revision, EVAL_SHARE and SPLIT_SEED"
    texts = [training_text(render, [{"role": "system", "content": SYSTEM_PROMPT}] + r["messages"]) for r in held]
    instruction_part, response_part = turn_markers(render)
    spans = [response_spans(t, instruction_part, response_part) for t in texts]
    del rows, held
    tok.padding_side = "right"
    result = {}
    try:
        for name, lora_on in (("base", False), ("fym1", True)):
            with adapter(lora_on):
                per_text = reply_nll(texts, spans)
            nll, n_tok = sum(n for n, _ in per_text), sum(k for _, k in per_text)
            result[name] = {"perplexity": round(perplexity(nll, n_tok), 3), "mean_nll": round(nll / n_tok, 4)}
        RUN["perplexity"] = result | {
            "n_texts": len(texts), "scored_tokens": n_tok,
            "truncated": sum(len(tok(t, add_special_tokens=False)["input_ids"]) > MAX_LEN for t in texts),
            "seconds": round(time.time() - t0), "retried_items": RETRIED["items"],
            "protocol": f"01_train eval split ({EVAL_SHARE:.0%}, shuffle seed {SPLIT_SEED}), training_text with the "
                        f"system prompt, reply tokens only (as train_on_responses_only), cut at {MAX_LEN} tokens"}
    except StillOutOfMemory:
        RUN["perplexity"] = {"oom": True, "retried_items": RETRIED["items"]}
    finally:
        tok.padding_side = "left"
    del texts, spans
    save_run("perplexity")
print("PERPLEXITY", {k: v for k, v in RUN["perplexity"].items() if k != "protocol"})"""),
code("""# notebook_metrics.json for `scripts/score.py scores --metrics` (leak_rate comes from `score.py leaks`, on the laptop).
baseline = None  # the base model's 00_baseline file: same Belebele/FLORES protocols, 4-bit
for f in sorted(f for f in list_repo_files(CKPT_REPO) if f.startswith("results/baseline/") and f.endswith(".json")):
    found = hub_json(f)
    if found and found.get("model", {}).get("hub_id", "").lower() == MODEL_ID.lower():
        baseline = (f, found)
        break
RUN["base_benchmarks_from"] = baseline[0] if baseline else None


def benchmarks(result):
    flores = result.get("flores", {})
    return {"belebele_acc": result.get("belebele_completion", {}).get("acc"),
            "chrf_plt_eng": flores.get("plt-eng_chrf++"), "chrf_eng_plt": flores.get("eng-plt_chrf++")}


ppl = RUN.get("perplexity", {})
values = {
    "base": benchmarks(baseline[1] if baseline else {}) | {"perplexity": ppl.get("base", {}).get("perplexity")},
    "fym1": benchmarks(RUN) | {"perplexity": ppl.get("fym1", {}).get("perplexity")},
}
metrics = {model: {k: v for k, v in vals.items() if v is not None} for model, vals in values.items()}
assert all(set(m) <= NOTEBOOK_METRICS for m in metrics.values())
body = (json.dumps(metrics, indent=2) + "\\n").encode()
upload("notebook_metrics.json", body, "notebook_metrics.json")
with open(f"{OUTPUT}/notebook_metrics.json", "wb") as f:
    f.write(body)
save_run("metrics")
missing = [s for s in (*ANSWER_FILES, *PROTOCOLS, "flores", "perplexity") if s not in RUN]
pending = {model: sorted(NOTEBOOK_METRICS - {"leak_rate"} - set(m)) for model, m in metrics.items()}
print("EVAL_DONE" if not missing else f"EVAL_INCOMPLETE {missing}", RUN_ID, "| pending metrics", pending)
print("NOTEBOOK_METRICS", json.dumps(metrics))"""),
    ]


NOTEBOOKS = {"00_baseline": baseline_cells, "01_train": train_cells, "02_eval_export": eval_cells,
             "04_synth": synth_cells}


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
        if not path.exists() or path.read_text() != text:
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
