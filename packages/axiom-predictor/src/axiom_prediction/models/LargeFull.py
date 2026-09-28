from ..configuration import AxiomModelConfig
from .base import AxiomPredictionNetwork

class LargeFull(AxiomPredictionNetwork):
    default_config = AxiomModelConfig(
        hidden_dim = 128,
        message_rounds = 4,
        num_hidden_layers = 3,
        graph_input = "full",
    )
