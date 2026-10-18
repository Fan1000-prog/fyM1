# Colab operator guide

How notebooks in `notebooks/` are run on a free Colab T4, driven by Playwright. Verified 2026-10-17 with
`notebooks/smoke.ipynb` (T4, `HF_TOKEN` secret, read + write to private `Fan1000-Prog/fyM1-ckpt`).

Since 2026-10-18 Colab also runs the Gemma 4 baseline (plain-text Belebele, FLORES+, smoke). Training and eval
run on Colab Pro (paid, A100 runtime; owner order); a Kaggle T4 is the fallback only. Same click path as below,
with the A100 picked as the runtime's GPU; the result JSON goes to `fyM1-ckpt` as usual.

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
   - Then ONE `browser_run_code_unsafe` call does everything and returns booleans and counts only
     (`fs`/`process` are not available in that sandbox; `page.request` runs in Node). Nothing can throw:
     a failed or non-200 token fetch, or a body that is not a token, returns `{ ok: false }` before the
     pane opens (so `HF_TOKEN` is never overwritten with an empty value); any later failure also returns
     `{ ok: false }`, never error text, which could echo page content. The pane is always closed (page reloaded if closing fails). On a re-run it
     updates the existing `HF_TOKEN` row instead of adding a second one, because a duplicate row makes
     the switch locator match twice.

     ```js
     async (page) => {
       let t = '';
       try {
         const res = await page.request.get('<one-time URL>');
         if (res.ok()) t = (await res.text()).trim();
       } catch {}
       if (!t.startsWith('hf_')) return { ok: false, filled: false };
       const sw = page.getByRole('switch', { name: 'Toggle notebook access for secret "HF_TOKEN"' });
       const valueBox = page.getByRole('textbox', { name: /value/i });
       let result = { ok: false };
       try {
         await page.getByRole('button', { name: 'Secrets' }).click();
         await page.getByRole('button', { name: 'Add new secret' }).waitFor({ timeout: 10000 });  // pane rendered
         const existing = await sw.count();
         if (existing === 0) {
           await page.getByRole('button', { name: 'Add new secret' }).click();
           await page.getByRole('textbox', { name: 'Name of new secret' }).last().fill('HF_TOKEN');
           await page.getByRole('textbox', { name: 'Value of new secret' }).last().fill(t);
         } else {
           // The HF_TOKEN row: the innermost element holding both its switch and a value field.
           const row = page.locator('*').filter({ has: sw }).filter({ has: valueBox }).last();
           await row.getByRole('textbox', { name: /value/i }).fill(t);
         }
         await page.keyboard.press('Tab');
         await page.waitForTimeout(1500);
         const rows = await sw.count();
         if ((await sw.first().getAttribute('aria-checked')) !== 'true') await sw.first().click();
         await page.waitForTimeout(1000);
         const accessOn = (await sw.first().getAttribute('aria-checked')) === 'true';
         result = { ok: true, filled: true, updatedExisting: existing > 0, rows, accessOn };
       } catch {
         result = { ok: false };
       } finally {
         try {
           await page.getByRole('button', { name: 'Close left pane' }).click({ timeout: 5000 });
         } catch {
           try { await page.reload(); } catch { result = { ok: false }; }
         }
       }
       let leaks = true;  // unknown counts as a leak
       try {
         await page.waitForTimeout(500);
         leaks = /hf_[A-Za-z0-9_.-]{20,}/.test(await page.locator('body').ariaSnapshot());
       } catch {}
       return { ...result, leaks };
     }
     ```

     Continue only if `ok`, `filled` and `accessOn` are `true`, `rows` is `1` and `leaks` is `false`.
     - `ok: false` or `leaks: true` (on any error): reload the page (`page.reload()` inside
       `browser_run_code_unsafe`), do not snapshot, run the grep-and-delete below, then report to the
       lead. A new one-time URL is needed for any retry (the server has exited).
     - `rows` above `1`: an older run left duplicate `HF_TOKEN` rows. Ask the owner to delete the extras
       by hand in the Secrets pane, then re-run.
     The secret is saved per Google account; in a new notebook Colab asks **Grant access** on the first
     `userdata.get` instead.
   - Finally scan Playwright output (home and current directory) for the token pattern and delete any hit:
     `grep -rlE --null 'hf_[A-Za-z0-9_.-]{20,}' ~/.playwright-mcp/ ./.playwright-mcp/ 2>/dev/null | xargs -0 rm -f`
     (`--null`, not `-Z`: on macOS `-Z` means decompress, so nothing would be deleted).
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
