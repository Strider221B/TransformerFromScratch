import torch
import torch.nn as nn
from components.layer_normalization import LayerNormalization

class ProjectionLayer(nn.Module):
 
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self._projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, sequence length, d_model) --> (Batch, sequence length, vocab_size)
        return torch.log_softmax(self._projection(x), dim=-1)