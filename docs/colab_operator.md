# Colab operator guide

How notebooks in `notebooks/` are run on a free Colab T4, driven by Playwright. Verified 2026-10-17 with
`notebooks/smoke.ipynb` (T4, `HF_TOKEN` secret, read + write to private `Fan1000-Prog/fyM1-ckpt`).

## Rules
- Use the owner's **personal** Google account: every Colab URL carries `?authuser=1`. Check the account
  button reads `...@gmail.com`, never the work account.
- The GitHub repo is private: never open notebooks from GitHub. Upload the `.ipynb` file instead.
  Notebooks get repo code from `Fan1000-Prog/fyM1-ckpt` under `code/`, or are self-contained.
- The HF token never appears in a transcript, tool argument, commit, notebook, or cell output.
- Nothing browser-related enters the repo: Playwright snapshots, screenshots and console logs stay in
  `~/.playwright-mcp/` (outside every worktree; `.playwright-mcp/` is also gitignored).

## Click path
1. Open `https://colab.research.google.com/?authuser=1` → **Upload notebook** → **Browse** → pick the
   `.ipynb`. Colab saves a copy to the owner's Drive and opens it.
2. Runtime: our notebooks carry `accelerator: GPU`, `gpuType: T4` metadata, so the header shows
   **Connect T4**. If not: **Runtime → Change runtime type → T4 GPU → Save**.
3. `HF_TOKEN` secret (once per Google account; it persists across notebooks, but notebook access is
   granted per notebook):
   - **No snapshot-producing tool** (`browser_click`, `browser_type`, `browser_snapshot`,
     `browser_find`, screenshots) may touch the page while the Secrets pane is open: they write the
     accessibility tree, which contains the secret value, to `.playwright-mcp/page-*.yml`.
   - Run `python scripts/serve_hf_token_once.py` in the background; it prints a one-time loopback URL.
   - Then ONE `browser_run_code_unsafe` call does everything and returns booleans only
     (`fs`/`process` are not available in that sandbox; `page.request` runs in Node):

     ```js
     async (page) => {
       const t = (await (await page.request.get('<one-time URL>')).text()).trim();
       await page.getByRole('button', { name: 'Secrets' }).click();
       await page.getByRole('button', { name: 'Add new secret' }).click();
       await page.getByRole('textbox', { name: 'Name of new secret' }).last().fill('HF_TOKEN');
       await page.getByRole('textbox', { name: 'Value of new secret' }).last().fill(t);
       await page.keyboard.press('Tab');
       await page.waitForTimeout(1500);
       const sw = page.getByRole('switch', { name: 'Toggle notebook access for secret "HF_TOKEN"' });
       if ((await sw.getAttribute('aria-checked')) !== 'true') await sw.click();
       await page.waitForTimeout(1000);
       const accessOn = (await sw.getAttribute('aria-checked')) === 'true';
       await page.getByRole('button', { name: 'Close left pane' }).click();
       await page.waitForTimeout(500);
       const leaks = /hf_[A-Za-z0-9_.-]{20,}/.test(await page.locator('body').ariaSnapshot());
       return { filled: t.startsWith('hf_'), accessOn, leaks };
     }
     ```

     Continue only if `accessOn` is `true` and `leaks` is `false`. The secret is saved per Google
     account; in a new notebook Colab asks **Grant access** on the first `userdata.get` instead.
   - Finally scan Playwright output for the token pattern and delete any hit:
     `grep -rlE 'hf_[A-Za-z0-9_.-]{20,}' ~/.playwright-mcp/ | xargs rm -f`.
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
