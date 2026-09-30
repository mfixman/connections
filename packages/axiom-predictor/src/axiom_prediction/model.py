from __future__ import annotations

from dataclasses import dataclass
import io
import math
import os
from pathlib import Path

import torch

from connections.syntax.matrix import Matrix

from .graph import build_axiom_graph, collate_axiom_graphs
from .configuration import AxiomModelConfig
from .models import AxiomPredictionNetwork, load_model_class
from .choices import plain_values
from .logs import log

CHECKPOINT_FORMAT = "learncop.axiom-predictor"
CHECKPOINT_VERSION = 3

@dataclass(frozen = True, slots = True)
class AxiomPrediction:
    clause_index: int
    clause_text: str
    probability: float
    rank: int

class AxiomPredictor:
    def __init__(
        self,
        model: AxiomPredictionNetwork,
        *,
        device: str | torch.device = "cpu",
        training_config: dict[str, object] | None = None,
    ):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.training_config = dict(training_config or {})

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        device: str | torch.device = "auto",
    ) -> "AxiomPredictor":
        checkpoint_path = Path(checkpoint)
        log(f"loading checkpoint {checkpoint_path} onto {device}")
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "model.pt"

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"axiom predictor checkpoint not found: {checkpoint_path}")

        resolved_device = resolve_device(device)
        try:
            payload = torch.load(
                checkpoint_path,
                map_location = "cpu",
                weights_only = True,
            )
        except Exception as error:
            raise ValueError(
                f"could not load axiom predictor checkpoint {checkpoint_path}: {error}"
            ) from error

        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"not an axiom predictor checkpoint: {checkpoint_path}")

        if payload.get("version") not in (2, CHECKPOINT_VERSION):
            raise ValueError(
                f"unsupported axiom predictor checkpoint version: {payload.get('version')!r}"
            )

        try:
            config = AxiomModelConfig(**payload["model_config"])
            model_class = (
                load_model_class(payload["model_name"])
                if payload.get("model_name")
                else AxiomPredictionNetwork
            )

            model = model_class(config)
            model.load_state_dict(payload["model_state_dict"])
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError(f"malformed axiom predictor checkpoint {checkpoint_path}: {error}") from error

        training_config = payload.get("training_config")
        log(f"loaded {type(model).__name__} checkpoint (epoch {payload.get('epoch')})")
        return cls(
            model,
            device = resolved_device,
            training_config = training_config
            if isinstance(training_config, dict)
            else None,
        )

    def predict(
        self,
        matrix: Matrix,
        *,
        axiom_clause_ids: tuple[int, ...] | list[int],
        conjecture_clause_ids: tuple[int, ...] | list[int],
    ) -> tuple[AxiomPrediction, ...]:
        example = build_axiom_graph(
            matrix,
            axiom_clause_ids = axiom_clause_ids,
            conjecture_clause_ids = conjecture_clause_ids,
        )

        batch = collate_axiom_graphs([(example, None)])
        with torch.no_grad():
            logits = self.model(batch)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("model produced non-finite logits")

            probabilities = torch.sigmoid(logits).cpu().tolist()

        if any(not math.isfinite(value) for value in probabilities):
            raise FloatingPointError("model produced non-finite probabilities")

        ranked = sorted(
            zip(example.axiom_clause_ids, probabilities, strict = True),
            key = lambda item: (-item[1], item[0]),
        )

        return tuple(
            AxiomPrediction(
                clause_index = clause_index,
                clause_text = str(matrix.clauses[clause_index]),
                probability = float(probability),
                rank = rank,
            )
            for rank, (clause_index, probability) in enumerate(ranked, start = 1)
        )

def save_checkpoint(
    path: str | Path,
    model: AxiomPredictionNetwork,
    *,
    training_config: dict[str, object],
    epoch: int | None = None,
    training_state: dict | None = None,
):
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("refusing to save non-finite model parameters")

    buffer = io.BytesIO()
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,

            "model_name": None
            if type(model) is AxiomPredictionNetwork
            else type(model).__module__.rsplit(".", 1)[-1],

            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),

            "training_config": plain_values(training_config),
            "epoch": epoch,
            "training_state": training_state,
        },
        buffer,
    )

    write_durably(Path(path), buffer.getvalue())

def write_durably(path: Path, data: bytes):
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok = True)

    try:
        directory = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(directory)
    except OSError:
        pass
    finally:
        os.close(directory)

def resolve_device(device: str | torch.device) -> torch.device:
    value = str(device)
    if value in ("auto", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {value}")

    return torch.device(value)

def device_description(device: str | torch.device) -> str:
    resolved = resolve_device(device)
    if resolved.type == "cuda":
        index = resolved.index
        if index is None:
            index = torch.cuda.current_device()

        return f"CUDA ({resolved}; {torch.cuda.get_device_name(index)})"

    if resolved.type == "cpu":
        return "CPU"

    return f"{resolved.type.upper()} ({resolved})"

__all__ = [
    "AxiomModelConfig",
    "AxiomPrediction",
    "AxiomPredictionNetwork",
    "AxiomPredictor",

    "device_description",
    "resolve_device",
    "save_checkpoint",
    "write_durably",
]
