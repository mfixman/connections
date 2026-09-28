from .choices import GraphInputKind, plain_values

from dataclasses import asdict, dataclass

@dataclass(frozen = True, slots = True)
class AxiomModelConfig:
    hidden_dim: int = 64
    message_rounds: int = 3
    num_hidden_layers: int = 2
    activation: str = "relu"
    graph_input: GraphInputKind | str = GraphInputKind.Full

    def __post_init__(self):
        if self.activation != "relu":
            raise ValueError("the axiom predictor currently requires activation='relu'")

        object.__setattr__(self, "graph_input", GraphInputKind(self.graph_input))

        if self.hidden_dim < 1 or self.num_hidden_layers < 0 or self.message_rounds < 0:
            raise ValueError(
                "hidden_dim must be positive; layer and message counts must be nonnegative"
            )

    def to_dict(self) -> dict[str, int | str]:
        return plain_values(asdict(self))
