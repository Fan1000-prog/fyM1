# Colab operator guide

How notebooks in `notebooks/` are run on a free Colab T4, driven by Playwright. Verified 2026-10-17 with
`notebooks/smoke.ipynb` (T4, `HF_TOKEN` secret, read + write to private `Fan1000-Prog/fyM1-ckpt`).

## Rules
- Use the owner's **personal** Google account: every Colab URL carries `?authuser=1`. Check the account
  button reads `...@gmail.com`, never the work account.
- The GitHub repo is private: never open notebooks from GitHub. Upload the `.ipynb` file instead.
  Notebooks get repo code from `Fan1000-Prog/fyM1-ckpt` under `code/`, or are self-contained.
- The HF token never appears in a transcript, tool argument, commit, notebook, or cell output.

## Click path
1. Open `https://colab.research.google.com/?authuser=1` → **Upload notebook** → **Browse** → pick the
   `.ipynb`. Colab saves a copy to the owner's Drive and opens it.
2. Runtime: our notebooks carry `accelerator: GPU`, `gpuType: T4` metadata, so the header shows
   **Connect T4**. If not: **Runtime → Change runtime type → T4 GPU → Save**.
3. `HF_TOKEN` secret (once per Google account; it persists across notebooks, but notebook access is
   granted per notebook):
   - Left sidebar key icon (**Secrets**) → **Add new secret** → name `HF_TOKEN`.
   - Value: run `python scripts/serve_hf_token_once.py <random-path>` in the background, then in
     `browser_run_code_unsafe`:
     `const t = (await (await page.request.get('http://127.0.0.1:8765/<random-path>')).text()).trim();`
     `await page.getByRole('textbox', { name: 'Value of new secret' }).fill(t);`
     Return only booleans/length, never `t`. (`fs`/`process` are not available in that sandbox;
     `page.request` runs in Node, so there is no CORS or mixed-content issue.)
   - Toggle **notebook access** for `HF_TOKEN` (switch in the secret's row) and check it is on.
   - **Close the Secrets pane before any snapshot.** The accessibility snapshot of the open pane
     contains the secret value. Check with
     `/hf_[A-Za-z0-9_.-]{20,}/.test(await page.locator('body').ariaSnapshot())` → must be `false`.
4. **Runtime → Run all** (`Ctrl+F9`). Wait for "Allocating runtime" to finish (~30 s on T4).
   If a "Run anyway" (notebook not authored by Google) or "Grant access" dialog appears, accept it.
5. Read outputs with `browser_find` on a marker the notebook prints (smoke prints `SMOKE_OK {...}`), or
   evaluate `innerText` of the output areas. Download-progress output lives in an iframe; `browser_find`
   sees it.
6. Verify results from the Mac, not just from the page:
   `uv run python -c "from huggingface_hub import HfApi; print(HfApi().list_repo_files('Fan1000-Prog/fyM1-ckpt'))"`.

## Limits
Free-tier GPU quota messages ("usage limits", "cannot connect to GPU backend") stop the run: report the
exact text to the lead. Long runs push checkpoints to `fyM1-ckpt` and resume with
`fym1.train.resume.latest_checkpoint` after a disconnect.
