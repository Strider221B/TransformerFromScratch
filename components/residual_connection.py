import torch.nn as nn
from components.layer_normalization import LayerNormalization

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self._dropout = nn.Dropout(dropout)
        self._norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self._dropout(sublayer(self._norm(x)))
    