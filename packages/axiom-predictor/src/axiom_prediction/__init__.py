from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import AxiomPrediction, AxiomPredictor


def __getattr__(name: str) -> Any:
    if name in {"AxiomPrediction", "AxiomPredictor"}:
        from . import model

        return getattr(model, name)
    raise AttributeError(name)


__all__ = ["AxiomPrediction", "AxiomPredictor"]
