# Kaggle operator guide

Kaggle is the second GPU runtime next to Colab (owner-approved: 30 GPU h/week, 12 h sessions). It is
driven from the terminal with the Kaggle CLI, not a browser.

Status: prep only. `fym1.runtime` (HF token from Kaggle or Colab secrets, output and scratch folders) and
`scripts/kaggle_push.py` exist, but the notebooks (`00_baseline`, `01_train`, in PR #13) do not use them
yet; that wiring is a follow-up PR after #13 merges. Until then this page describes the intended flow.

## One-time setup (owner)
- Kaggle account with phone verification (needed for GPU and internet).
- Credentials: Kaggle → Settings → API → **Generate New Token**, saved where Kaggle says (the CLI reads
  its default access token). Agents never read or handle it.
- Tell the lead the Kaggle username (not a secret); agents pass it as `--user` or `KAGGLE_USERNAME`.
- The first push of each kernel runs before any secret is attached, so that first run fails at the HF
  token step. Then open the kernel on kaggle.com → **Add-ons → Secrets** → add `HF_TOKEN` and tick it for
  this notebook, and push again.

## Run a notebook
1. Code reaches the kernel the same way as on Colab: `fyM1-ckpt` `code/`, synced from clean origin/main
   with `scripts/sync_code_to_hub.py` (baseline runs may use branch code if each result records its SHA).
2. Push and start a run: `python scripts/kaggle_push.py 00_baseline --user <name>` (or `01_train`). This
   writes `kernel-metadata.json` (private, GPU, internet on, `machine_shape` T4) and calls
   `uvx --from kaggle==2.2.4 kaggle kernels push`. Every push is a new kernel version and starts a run.
3. Poll: `uvx --from kaggle==2.2.4 kaggle kernels status <name>/fym1-00-baseline` until `complete`,
   `error` or `cancelAcknowledged`. Check every few minutes, not in a tight loop.
4. Logs and outputs: `uvx --from kaggle==2.2.4 kaggle kernels output <name>/fym1-00-baseline -p <dir>`
   (put `<dir>` outside the repo). Results that matter are pushed to `fyM1-ckpt` by the notebook itself,
   as on Colab; read them from there.
5. Quota: `uvx --from kaggle==2.2.4 kaggle quota`.

## Disk
`/kaggle/working` is saved with each run (~20 GB cap) and is what `kernels output` downloads: keep it for
small wanted outputs (`fym1.runtime.output_dir()`). Model downloads, checkpoints and merged models go to
`fym1.runtime.scratch_dir()` (a temp folder), and checkpoints are pushed to `fyM1-ckpt` as on Colab.

## GPU
- Prefer T4. T4 (Turing) has no bf16, so Unsloth forces float32 training for gemma3, gemma4 and qwen3_5,
  as on Colab. P100 (Pascal) has no bf16 either and is older; do not use it.

## To verify on the first run (then update this page)
- The `machine_shape` value `NvidiaTeslaT4` is accepted (the CLI does not list the allowed names).
- Whether the T4 runtime shows two GPUs; if so the wiring PR sets `CUDA_VISIBLE_DEVICES=0`.
- The `HF_TOKEN` secret stays attached to the kernel across new versions pushed from the CLI.
- `kernels output` returns the run log.
- The temp folder used by `scratch_dir()` has room for a 12B model download plus checkpoints.
