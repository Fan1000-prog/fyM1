"""After anonymising: capitalised words that survived, which may be missed names.

`fym1.data.gate` replaces the ones no known-word list has. The optional sheet shows
tokens and counts only (never sentences); a saved blocklist.txt fed back to build_chats.py
replaces those words everywhere, so fewer conversations are dropped.
"""

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from html import escape
from pathlib import Path

from fym1.data.anonymize import compare_key, is_everyday_word

LETTERS = re.compile(r"(?<![<\w])[^\W\d_]+(?![\w>])")  # whole letter-only words, not <TEL>
MIN_LETTERS = 2


def fake_name_keys(fake_names: Iterable[str]) -> set[str]:
    """Compare keys of every word in `fake_names`."""
    return {compare_key(w) for name in fake_names for w in name.split()}


def leftover_matches(text: str, fake_keys: set[str]) -> Iterator[re.Match]:
    """Capitalised words in `text` that are not common Malagasy words, placeholders or fake names."""
    for match in LETTERS.finditer(text):
        word = match.group()
        if (
            word[0].isupper()
            and len(word) >= MIN_LETTERS
            and not is_everyday_word(word)
            and compare_key(word) not in fake_keys
        ):
            yield match


def leftover_names(texts: Iterable[str], fake_names: Iterable[str]) -> Counter:
    """Capitalised words in `texts` that are not common Malagasy words or our fake names."""
    fake_keys = fake_name_keys(fake_names)
    return Counter(match.group() for text in texts for match in leftover_matches(text, fake_keys))


def read_blocklist(path: Path) -> list[str]:
    """Names the owner ticked on the leftover sheet, one per line."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


PAGE = """<!doctype html>
<html lang="mg">
<head>
<meta charset="utf-8">
<title>fyM1 leftover names</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 40rem; margin: 2rem auto; padding: 0 1rem; }}
td {{ padding: .2rem .6rem; }} td.n {{ text-align: right; color: #666; }}
#export {{ position: sticky; bottom: 1rem; font-size: 1.1rem; padding: .5rem 1rem; }}
</style>
</head>
<body>
<h1>fyM1 — {count} capitalised words left after anonymising</h1>
<p>Tick every word that is a real person's name or nickname. Then save <code>blocklist.txt</code>
and run <code>build_chats.py</code> again with <code>--blocklist</code>.</p>
<table>
{rows}
</table>
<button id="export">Save blocklist.txt</button>
<script>
document.getElementById("export").onclick = () => {{
  const names = [...document.querySelectorAll("input[type=checkbox]:checked")].map(b => b.value);
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([names.join("\\n") + "\\n"], {{type: "text/plain"}}));
  link.download = "blocklist.txt";
  link.click();
}};
</script>
</body>
</html>
"""

ROW = '<tr><td><label><input type="checkbox" value="{word}"> {word}</label></td><td class="n">{n}</td></tr>'


def render_leftover_sheet(counts: Counter) -> str:
    """The tick-the-real-names page: one row per word with its count, most common first."""
    rows = "\n".join(ROW.format(word=escape(word), n=n) for word, n in counts.most_common())
    return PAGE.format(count=len(counts), rows=rows)
