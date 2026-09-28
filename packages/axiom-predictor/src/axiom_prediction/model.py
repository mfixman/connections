from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import os
from pathlib import Path

import torch
from torch import nn

from connections.syntax.matrix import Matrix
from axiom_prediction.encoder import GraphModelConfig, GraphNetwork

from .graph import AxiomGraphBatch, build_axiom_graph, collate_axiom_graphs
from .inputs import GRAPH_INPUTS, select_graph_input

CHECKPOINT_FORMAT = "learncop.axiom-predictor"
CHECKPOINT_VERSION = 2


@dataclass(frozen=True, slots=True)
class AxiomModelConfig:
    hidden_dim: int = 64
    message_rounds: int = 3
    num_hidden_layers: int = 2
    activation: str = "relu"
    graph_input: str = "full"

    def __post_init__(self):
        if self.activation != "relu":
            raise ValueError("the axiom predictor currently requires activation='relu'")
        if self.graph_input not in GRAPH_INPUTS:
            raise ValueError(f"graph_input must be one of {GRAPH_INPUTS}")
        if self.hidden_dim < 1 or self.num_hidden_layers < 0 or self.message_rounds < 0:
            raise ValueError("hidden_dim must be positive; layer and message counts must be nonnegative")

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


MODEL_PRESETS = {
    "small": AxiomModelConfig(hidden_dim=32, message_rounds=2, num_hidden_layers=1),
    "default": AxiomModelConfig(),
    "large": AxiomModelConfig(hidden_dim=128, message_rounds=4, num_hidden_layers=3),
}


class AxiomPredictionNetwork(nn.Module):
    def __init__(self, config: AxiomModelConfig):
        super().__init__()
        self.config = config
        self.encoder = GraphNetwork(
            GraphModelConfig(
                hidden_dim=config.hidden_dim,
                message_rounds=config.message_rounds,
                num_hidden_layers=config.num_hidden_layers,
                activation=config.activation,
            )
        )
        dim = config.hidden_dim
        layers: list[nn.Module] = []
        input_dim = dim * 5
        for _ in range(config.num_hidden_layers):
            layers.extend((nn.Linear(input_dim, dim), nn.ReLU()))
            input_dim = dim

        layers.append(nn.Linear(input_dim, 1))
        self.scorer = nn.Sequential(*layers)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch: AxiomGraphBatch) -> torch.Tensor:
        graph = select_graph_input(batch.graph, self.config.graph_input)
        encoded = self.encoder.encode_matrix(graph)
        clauses = encoded["clause"]
        axiom_indices = batch.axiom_clause_indices.to(self.device)
        conjecture_indices = batch.conjecture_clause_indices.to(self.device)
        axiom_batch = batch.axiom_batch.to(self.device)
        conjecture_batch = batch.conjecture_batch.to(self.device)
        batch_size = len(batch.axiom_counts)
        conjecture_context = _pool_mean(clauses[conjecture_indices], conjecture_batch, batch_size)
        axiom_context = _pool_mean(clauses[axiom_indices], axiom_batch, batch_size)
        cand = clauses[axiom_indices]
        conj = conjecture_context[axiom_batch]
        all_ax = axiom_context[axiom_batch]
        x = torch.cat(
            (
                cand,
                conj,
                all_ax,
                cand * conj,
                cand - conj,
            ),
            dim=1,
        )
        return self.scorer(x).squeeze(1)


def _pool_mean(values: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    output = torch.zeros((size, values.shape[1]), device=values.device)
    output.index_add_(0, batch, values)
    counts = torch.zeros(size, device=values.device)
    counts.index_add_(0, batch, torch.ones(len(batch), device=values.device))
    return output / counts.clamp(min=1).unsqueeze(1)


@dataclass(frozen=True, slots=True)
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
    def load(cls, checkpoint: str | Path, device: str | torch.device = "auto") -> "AxiomPredictor":
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "model.pt"

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"axiom predictor checkpoint not found: {checkpoint_path}")

        resolved_device = resolve_device(device)
        try:
            payload = torch.load(
                checkpoint_path,
                map_location=resolved_device,
                weights_only=True,
            )
        except Exception as error:
            raise ValueError(
                f"could not load axiom predictor checkpoint {checkpoint_path}: {error}"
            ) from error

        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"not an axiom predictor checkpoint: {checkpoint_path}")

        if payload.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported axiom predictor checkpoint version: {payload.get('version')!r}"
            )

        try:
            config = AxiomModelConfig(**payload["model_config"])
            model = AxiomPredictionNetwork(config)
            model.load_state_dict(payload["model_state_dict"])
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError(
                f"malformed axiom predictor checkpoint {checkpoint_path}: {error}"
            ) from error

        training_config = payload.get("training_config")
        return cls(
            model,
            device=resolved_device,
            training_config=training_config if isinstance(training_config, dict) else None,
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
            axiom_clause_ids=axiom_clause_ids,
            conjecture_clause_ids=conjecture_clause_ids,
        )
        batch = collate_axiom_graphs([(example, None)])
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(batch)).cpu().tolist()

        ranked = sorted(
            zip(example.axiom_clause_ids, probabilities, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(
            AxiomPrediction(
                clause_index=clause_index,
                clause_text=str(matrix.clauses[clause_index]),
                probability=float(probability),
                rank=rank,
            )
            for rank, (clause_index, probability) in enumerate(ranked, start=1)
        )


def save_checkpoint(
    path: str | Path,
    model: AxiomPredictionNetwork,
    *,
    training_config: dict[str, object],
    epoch: int | None = None,
):
    buffer = io.BytesIO()
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "training_config": training_config,
            "epoch": epoch,
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
        tmp.unlink(missing_ok=True)
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
    if value == "auto":
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
