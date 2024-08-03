import torch.nn as nn
from components.layer_normalization import LayerNormalization

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self._layers = layers
        self._norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self._layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self._norm(x)
