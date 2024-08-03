import torch.nn as nn
from components.layer_normalization import LayerNormalization

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self._layers = layers
        self._norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self._layers:
            x = layer(x, mask)
        return self._norm(x)
