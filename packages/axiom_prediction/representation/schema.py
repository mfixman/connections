# Node features and edge types shared by preprocessing and the model.

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
    # Graph tables stored as lists for JSON serialization.

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
                key: list(map(list, val))
                for key, val in payload["nodes"].items()
            },
            edges = {
                key: list(map(list, val))
                for key, val in payload["edges"].items()
            },
            actions = list(map(list, payload["actions"])),
            preprocessor = payload.get("preprocessor", "graph"),
            version = payload.get("version", "1"),
            metadata = dict(payload.get("metadata", {})),
        )

def arg_position_bucket(position: int) -> int:
    return min(position, arg_position_buckets - 1)
