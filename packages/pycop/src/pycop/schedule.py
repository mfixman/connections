from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from connections.syntax.logic import Logic
from pycop.settings_codec import LeancopSettingsCodec
from connections.interaction.strategy import Strategy, WeightedStrategy


_SCHEDULE_DIR = Path(__file__).with_name("schedules")
_BUILTIN_SCHEDULE_PATHS = {
    "classical": _SCHEDULE_DIR / "classical.json",
    "intuitionistic": _SCHEDULE_DIR / "intuitionistic.json",
    "modal": _SCHEDULE_DIR / "modal.json",
}


def load_schedule_entries(path_or_name: str | Path) -> list[WeightedStrategy[Strategy]]:
    path = Path(path_or_name)
    if isinstance(path_or_name, str) and path_or_name in _BUILTIN_SCHEDULE_PATHS:
        path = _BUILTIN_SCHEDULE_PATHS[path_or_name]

    data = json.loads(path.read_text(encoding = "utf-8"))
    return schedule_entries(data)


def schedule_entries(data: object) -> list[WeightedStrategy[Strategy]]:
    if isinstance(data, Mapping):
        data = data.get("entries")
    if not isinstance(data, list):
        raise ValueError("schedule file must contain a list or an object with entries")

    entries: list[WeightedStrategy[Strategy]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"schedule entry {index} must be an object")
        tokens = item.get("settings")
        if tokens is None:
            tokens = item.get("tokens", [])
        if not isinstance(tokens, list) or not all(
            isinstance(token, str) for token in tokens
        ):
            raise ValueError(f"schedule entry {index} settings must be a string list")
        weight = item.get("weight", 1)
        if not isinstance(weight, int):
            raise ValueError(f"schedule entry {index} weight must be an integer")
        strategy = LeancopSettingsCodec.from_tokens(tokens)
        entries.append(WeightedStrategy(strategy = strategy, weight = weight))
    return entries


_CLASSICAL_SCHEDULE = load_schedule_entries("classical")
_INTUITIONISTIC_SCHEDULE = load_schedule_entries("intuitionistic")
_MODAL_SCHEDULE = load_schedule_entries("modal")

SCHEDULE_BY_LOGIC: dict[Logic, list[WeightedStrategy[Strategy]]] = {
    "classical": _CLASSICAL_SCHEDULE,
    "intuitionistic": _INTUITIONISTIC_SCHEDULE,
    "D": _MODAL_SCHEDULE,
    "T": _MODAL_SCHEDULE,
    "S4": _MODAL_SCHEDULE,
    "S5": _MODAL_SCHEDULE,
}


__all__ = [
    "load_schedule_entries",
    "SCHEDULE_BY_LOGIC",
    "WeightedStrategy",
]
