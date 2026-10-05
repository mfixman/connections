from ..choices import GraphInputKind

from ..configuration import AxiomModelConfig
from .base import AxiomPredictionNetwork

class SmallNoComplements(AxiomPredictionNetwork):
    default_config = AxiomModelConfig(
        hidden_dim = 32,
        message_rounds = 2,
        num_hidden_layers = 1,
        graph_input = GraphInputKind.NoComplements,
    )
