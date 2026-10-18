"""build_synth_stats copies only validated, counts-only stats. All values are made up."""

import importlib.util
import json
from pathlib import Path

from fym1.data.synth import DROP_REASONS, synth_stats

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_synth_stats", ROOT / "scripts" / "build_synth_stats.py")
build_synth_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_synth_stats)


def stats(**extra):
    dropped = {reason: 1 for reason in DROP_REASONS}
    return {**synth_stats(raw=10, kept=10 - len(DROP_REASONS), dropped=dropped,
                          by_generator={"google/gemma-4-12B-it": 10}, meta={"seed": 0}), **extra}


def test_writes_valid_stats(tmp_path):
    out = tmp_path / "synth_stats.json"
    assert build_synth_stats.main(lambda: stats(), out) == 0
    assert json.loads(out.read_text()) == stats()


def test_refuses_stats_with_text_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "synth_stats.json"
    assert build_synth_stats.main(lambda: stats(sample="Salama ry namana"), out) == 1
    assert not out.exists()
    assert "Salama" not in capsys.readouterr().err
