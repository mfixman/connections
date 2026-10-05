from ..choices import GraphInputKind

from ..configuration import AxiomModelConfig
from .base import AxiomPredictionNetwork

class LargeNoComplements(AxiomPredictionNetwork):
    default_config = AxiomModelConfig(
        hidden_dim = 128,
        message_rounds = 4,
        num_hidden_layers = 3,
        graph_input = GraphInputKind.NoComplements,
    )
