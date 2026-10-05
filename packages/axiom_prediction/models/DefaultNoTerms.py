from ..choices import GraphInputKind

from ..configuration import AxiomModelConfig
from .base import AxiomPredictionNetwork

class DefaultNoTerms(AxiomPredictionNetwork):
    default_config = AxiomModelConfig(
        hidden_dim = 64,
        message_rounds = 3,
        num_hidden_layers = 2,
        graph_input = GraphInputKind.NoTerms,
    )
