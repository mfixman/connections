from ..configuration import AxiomModelConfig
from .base import AxiomPredictionNetwork

class ExtraLargeFull(AxiomPredictionNetwork):
    default_config = AxiomModelConfig(
        hidden_dim = 256,
        message_rounds = 5,
        num_hidden_layers = 4,
        graph_input = "full",
    )
