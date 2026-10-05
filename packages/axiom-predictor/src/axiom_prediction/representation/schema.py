"""Node features and edge types shared by preprocessing and the model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

node_types: list[str] = [
    "symbol",
    "clause",
    "literal",
    "term",
    "var",
    "goal",
]

# (name, source node type, destination node type, has positional feature)

relations: list[tuple[str, str, str, bool]] = [
    ("contains", "clause", "literal", False),
    ("atom", "literal", "term", False),

    ("arg_term", "term", "term", True),
    ("arg_var", "term", "var", True),

    ("sym", "term", "symbol", False),
    ("lit_sym", "literal", "symbol", False),
    ("complement", "literal", "literal", False),

    ("instance_of", "goal", "literal", False),
    ("parent", "goal", "goal", False),
    ("path", "goal", "goal", False),
]

arg_position_buckets = 6

@dataclass(frozen = True, slots = True)
class GraphInput:
    """Graph tables stored as lists for JSON serialization."""

    nodes: dict[str, list[list[int]]]
    edges: dict[str, list[list[int]]]
    actions: list[list[int]]
    preprocessor: str = "graph"
    version: str = "1"
    metadata: dict[str, Any] = field(default_factory = dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphInput":
        return cls(
            nodes = {
                key: [list(f) for f in val]
                for key, val in payload["nodes"].items()
            },
            edges = {
                key: [list(e) for e in val]
                for key, val in payload["edges"].items()
            },
            actions = [list(a) for a in payload["actions"]],
            preprocessor = payload.get("preprocessor", "graph"),
            version = payload.get("version", "1"),
            metadata = dict(payload.get("metadata", {})),
        )

@dataclass(frozen = True, slots = True)
class GraphTensors:
    """Graph tables converted to tensors for training."""

    nodes: dict[str, Any]  # node type -> LongTensor [n, n_features]
    edges: dict[str, Any]  # relation -> LongTensor [n, 2 or 3]
    actions: Any  # LongTensor [n, 4]
    metadata: dict[str, Any] = field(default_factory = dict)

    def to(self, device: Any) -> "GraphTensors":
        return GraphTensors(
            nodes = {k: v.to(device) for k, v in self.nodes.items()},
            edges = {k: v.to(device) for k, v in self.edges.items()},
            actions = self.actions.to(device),
            metadata = self.metadata,
        )

def arg_position_bucket(position: int) -> int:
    return min(position, arg_position_buckets - 1)
