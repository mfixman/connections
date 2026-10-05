from __future__ import annotations

from importlib import import_module
from pathlib import Path

def __getattr__(name):
    if name == "AxiomPredictionNetwork":
        from .base import AxiomPredictionNetwork

        return AxiomPredictionNetwork

    raise AttributeError(name)

def available_models() -> list[str]:
    return sorted(
            p.stem
            for p in Path(__file__).parent.glob("*.py")
            if p.stem not in ("base", "__init__")
        )

def load_model_class(name: str):
    from .base import AxiomPredictionNetwork

    module_name = name.removesuffix(".py")
    if module_name not in available_models():
        raise ValueError(f"unknown model {name!r}; choose one of {', '.join(available_models())}")

    model = getattr(import_module(f"{__name__}.{module_name}"), module_name, None)
    if not isinstance(model, type) or not issubclass(model, AxiomPredictionNetwork):
        raise ValueError(
            f"{module_name}.py must define {module_name} inheriting AxiomPredictionNetwork"
        )

    return model
