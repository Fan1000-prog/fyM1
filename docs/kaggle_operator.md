# Kaggle operator guide

Kaggle is the second GPU runtime next to Colab (owner-approved: 30 GPU h/week, 12 h sessions). It is
driven from the terminal with the Kaggle CLI, not a browser. Notebooks are the same files as on Colab;
they detect the runtime (`fym1.runtime`) for the HF token and the working directory.

## One-time setup (owner)
- Kaggle account with phone verification (needed for GPU and internet), `~/.kaggle/kaggle.json` on the Mac.
- After the first push of each kernel: open it on kaggle.com → **Add-ons → Secrets** → add `HF_TOKEN`
  and tick it for this notebook. Agents never handle this step.

## Run a notebook
1. Code reaches the kernel the same way as on Colab: `fyM1-ckpt` `code/`, synced from clean origin/main
   with `scripts/sync_code_to_hub.py` (baseline runs may use branch code if each result records its SHA).
2. Push and start a run: `python scripts/kaggle_push.py 00_baseline` (or `01_train`). This writes
   `kernel-metadata.json` (private, GPU, internet on, `machine_shape` T4) and calls
   `uvx --from kaggle==2.2.4 kaggle kernels push`. Every push is a new kernel version and starts a run.
3. Poll: `uvx --from kaggle==2.2.4 kaggle kernels status <user>/fym1-00-baseline` until
   `complete`, `error` or `cancelAcknowledged`. Check every few minutes, not in a tight loop.
4. Logs and outputs: `uvx --from kaggle==2.2.4 kaggle kernels output <user>/fym1-00-baseline -p <dir>`
   (put `<dir>` outside the repo). Results that matter are pushed to `fyM1-ckpt` by the notebook itself,
   as on Colab; read them from there.
5. Quota: `uvx --from kaggle==2.2.4 kaggle quota`.

## GPU
- Prefer T4. T4 (Turing) has no bf16, so Unsloth forces float32 training for gemma3, gemma4 and qwen3_5,
  as on Colab. P100 (Pascal) has no bf16 either and is older; do not use it.
- Kaggle's T4 runtime has two GPUs; the notebooks use one.

## To verify on the first run (then update this page)
- The `machine_shape` value `NvidiaTeslaT4` is accepted (the CLI does not list the allowed names).
- The `HF_TOKEN` secret stays attached to the kernel across new versions pushed from the CLI.
- `kernels output` returns the run log.
