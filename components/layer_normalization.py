import torch
import torch.nn as nn

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self._eps = eps
        self._alpha = nn.Parameter(torch.ones(1))
        self._bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True) # since it's layer norm, all processing is on the last dimension.
        std = x.std(dim = -1, keepdim=True)
        return self._alpha * (x - mean) / (std + self._eps) + self._bias
