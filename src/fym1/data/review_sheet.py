"""A self-contained HTML page where the owner approves or rejects sample conversations."""

import random
from html import escape

PAGE = """<!doctype html>
<html lang="mg">
<head>
<meta charset="utf-8">
<title>fyM1 review</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 46rem; margin: 2rem auto; padding: 0 1rem; }}
section {{ border: 1px solid #ccc; border-radius: 8px; padding: .75rem 1rem; margin: 1rem 0; }}
.user {{ color: #555; }} .assistant {{ color: #0a4; }}
p {{ white-space: pre-wrap; margin: .3rem 0; }}
#export {{ position: sticky; bottom: 1rem; font-size: 1.1rem; padding: .5rem 1rem; }}
</style>
</head>
<body>
<h1>fyM1 — review {count} conversations</h1>
<p>Tick every conversation that is clean (no real names, numbers or private details) and sounds natural.
Then press the button to save <code>approved_ids.txt</code>.</p>
{sections}
<button id="export">Save approved_ids.txt</button>
<script>
document.getElementById("export").onclick = () => {{
  const ids = [...document.querySelectorAll("input[type=checkbox]:checked")].map(b => b.value);
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([ids.join("\\n") + "\\n"], {{type: "text/plain"}}));
  link.download = "approved_ids.txt";
  link.click();
}};
</script>
</body>
</html>
"""

SECTION = """<section>
<label><input type="checkbox" value="{id}"> approve <code>{id}</code></label>
{turns}
</section>"""

TURN = '<p class="{role}"><b>{role}:</b> {content}</p>'


def sample_for_review(records: list[dict], n: int = 100, seed: int = 0) -> list[dict]:
    """Up to `n` records, picked the same way every run for the same seed."""
    return random.Random(seed).sample(records, min(n, len(records)))


def render_review_sheet(records: list[dict]) -> str:
    """The review page for `records` (`{"id", "messages"}`), all text HTML-escaped."""
    sections = "\n".join(
        SECTION.format(
            id=escape(record["id"]),
            turns="\n".join(
                TURN.format(role=escape(turn["role"]), content=escape(turn["content"]))
                for turn in record["messages"]
            ),
        )
        for record in records
    )
    return PAGE.format(count=len(records), sections=sections)
