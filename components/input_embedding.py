import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self._d_model = d_model
        self._vocab_size = vocab_size
        self._embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self._embedding(x) * math.sqrt(self._d_model)
