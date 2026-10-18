# Kaggle operator guide

Kaggle is the second GPU runtime next to Colab (owner-approved: 30 GPU h/week, 12 h sessions). Runs are
started from the Kaggle web page ("Save & Run All", below); the Kaggle CLI checks status, reads outputs and quota.

`00_baseline` and `01_train` run unchanged on Colab and Kaggle: a bootstrap cell gets the HF token from
Kaggle or Colab secrets (a verbatim copy of `fym1.runtime.hf_token`, checked by a test), pins one GPU, and
later cells put small outputs in `output_dir()` and big files in `scratch_dir()`. Push only notebooks from
a commit that has this wiring (the kaggle-wiring PR); older notebooks read the token from Colab only.

## One-time setup (owner)
- Kaggle account with phone verification. Without it the run gets no GPU and no internet (baseline
  versions 2–4 failed this way).
- Credentials: Kaggle → Settings → API → **Generate New Token**, saved where Kaggle says (the CLI reads
  its default access token). Agents never read or handle it.
- Tell the lead the Kaggle username (not a secret); agents pass it as `--user` or `KAGGLE_USERNAME`.
  It must be the account the saved token belongs to: the push goes to the token's account whatever `--user`
  says (this happened once). The private handoff names the account.
- The first push of each kernel runs before any secret is attached, so that first run fails at the HF
  token step. Then open the kernel on kaggle.com → **Add-ons → Secrets** → add `HF_TOKEN` and tick it for
  this notebook, and push again.

## Run a notebook
1. Code reaches the kernel the same way as on Colab: `fyM1-ckpt` `code/`, synced from clean origin/main
   with `scripts/sync_code_to_hub.py` (baseline runs may use branch code if each result records its SHA).
2. Start the run from the Kaggle UI: open the kernel → check the settings panel (Accelerator **GPU T4**,
   **Internet on**, **Add-ons → Secrets → `HF_TOKEN`** ticked) → **Save Version → Save & Run All (Commit)**.
   This keeps internet and the HF secret on. Pushes from the CLI
   (`python scripts/kaggle_push.py 00_baseline --user <name>`) lost those settings in practice: the run came
   up without internet or the token. Use the CLI push only to create a kernel or upload a new notebook file,
   then start the run from the UI. Every new version starts a run.
3. Poll: `uvx --from kaggle==2.2.4 kaggle kernels status <name>/fym1-00-baseline` until `complete`,
   `error` or `cancelAcknowledged`. Check every few minutes, not in a tight loop.
4. Logs and outputs: `uvx --from kaggle==2.2.4 kaggle kernels output <name>/fym1-00-baseline -p <dir>`
   (put `<dir>` outside the repo). Results that matter are pushed to `fyM1-ckpt` by the notebook itself,
   as on Colab; read them from there.
5. Quota: `uvx --from kaggle==2.2.4 kaggle quota`.

## Rules
- **Never start a new version over a running one.** Check `kernels status` first; a new version can cancel the
  running one or queue behind it, and either way wastes GPU hours.
- **Stop idle interactive sessions.** An open draft (the notebook editor with a GPU session) uses quota even
  when nothing runs: stop it from the editor (power button) or from "Your Work → Active sessions".
- One model on one runtime: never run the same model on Kaggle and Colab at once.

## Disk
`/kaggle/working` is saved with each run (~20 GB cap) and is what `kernels output` downloads: keep it for
small wanted outputs (`fym1.runtime.output_dir()`). Model downloads, checkpoints and merged models go to
`fym1.runtime.scratch_dir()` (a temp folder), and checkpoints are pushed to `fyM1-ckpt` as on Colab.

## GPU
- Prefer T4. T4 (Turing) has no bf16, so Unsloth forces float32 training for gemma3, gemma4 and qwen3_5,
  as on Colab. P100 (Pascal) has no bf16 either and is older; do not use it.

## To verify on the first run (then update this page)
- The `machine_shape` value `NvidiaTeslaT4` is accepted (the CLI does not list the allowed names).
- Whether the T4 runtime shows two GPUs (the bootstrap sets `CUDA_VISIBLE_DEVICES=0` either way).
- The `HF_TOKEN` secret stays attached across CLI pushes: **no** in practice (see "Run a notebook", step 2).
- `kernels output` returns the run log.
- The temp folder used by `scratch_dir()` has room for a 12B model download plus checkpoints.
