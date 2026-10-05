from .choices import GraphInputKind, plain_values

from dataclasses import asdict, dataclass

@dataclass(slots = True, init = False)
class AxiomModelConfig:
    hidden_dim: int
    message_rounds: int
    num_hidden_layers: int
    activation: str
    graph_input: GraphInputKind | str

    def __init__(
        self,
        hidden_dim: int = 64,
        message_rounds: int = 3,
        num_hidden_layers: int = 2,
        activation: str = "relu",
        graph_input: GraphInputKind | str = GraphInputKind.Full,
    ):
        self.hidden_dim = hidden_dim
        self.message_rounds = message_rounds
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.graph_input = graph_input

        self.validate()

    def validate(self):
        if self.activation != "relu":
            raise ValueError("the axiom predictor currently requires activation='relu'")

        self.graph_input = GraphInputKind(self.graph_input)

        if self.hidden_dim < 1 or self.num_hidden_layers < 0 or self.message_rounds < 0:
            raise ValueError(
                "hidden_dim must be positive; layer and message counts must be nonnegative"
            )

    def to_dict(self) -> dict[str, int | str]:
        return plain_values(asdict(self))
