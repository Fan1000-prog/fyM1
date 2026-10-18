"""Fetch the counts-only synthetic-chat stats from the private Hub dataset into results/synth_stats.json.

Usage: python scripts/build_synth_stats.py
notebooks/04_synth.ipynb writes `synth/synth_stats.json` to the private Fan1000-Prog/fyM1-chats dataset. This
copies it here after checking it holds only counts and run settings (fym1.data.synth.validate_stats), never text.
"""

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from fym1.data.synth import validate_stats

REPO = "Fan1000-Prog/fyM1-chats"
REMOTE = "synth/synth_stats.json"
OUT = Path(__file__).resolve().parents[1] / "results" / "synth_stats.json"


def fetch() -> dict:
    return json.loads(Path(hf_hub_download(REPO, REMOTE, repo_type="dataset", force_download=True)).read_text())


def main(fetch_stats=fetch, out: Path = OUT) -> int:
    stats = fetch_stats()
    try:
        validate_stats(stats)
    except ValueError as error:
        print(f"refused {REMOTE}: {error}", file=sys.stderr)
        return 1
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {out.name}: {stats['kept']} kept of {stats['raw']} raw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
