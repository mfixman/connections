from __future__ import annotations

from typing import Any

def __getattr__(name: str) -> Any:
    if name in {"AxiomPrediction", "AxiomPredictor"}:
        from . import model

        return getattr(model, name)

    raise AttributeError(name)
