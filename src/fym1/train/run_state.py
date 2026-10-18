"""Per-run bookkeeping saved with every checkpoint, so a resumed Colab session continues the same run."""


def config_mismatch(saved: dict, current: dict) -> list[str]:
    """Differences between the run config stored with a checkpoint and the notebook's current one."""
    return [f"{k}: {saved.get(k)} -> {current.get(k)}" for k in sorted(saved.keys() | current.keys())
            if saved.get(k) != current.get(k)]


def session_progress(resumed: dict | None, session_seconds: float) -> dict:
    """Running totals to store with a checkpoint: earlier sessions (from the resumed checkpoint) plus this one."""
    resumed = resumed or {"wall_time_s": 0, "sessions": 0}
    return {"wall_time_s": resumed["wall_time_s"] + round(session_seconds), "sessions": resumed["sessions"] + 1}


def check_hub_id(what: str, found: str, expected: str) -> None:
    """Refuse a model id other than the one the notebook names (hub ids ignore case): results or training
    would otherwise describe a different checkpoint, as when a loader silently remaps a name."""
    if found.lower() != expected.lower():
        raise ValueError(f"{what} is {found!r}, expected {expected!r}")
