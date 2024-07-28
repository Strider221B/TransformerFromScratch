import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff:int, dropout: float) -> None:
        super().__init__()
        self._linear_1 = nn.Linear(d_model, d_ff) # W1 amd B1
        self._dropout = nn.Dropout(dropout)
        self._linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self._linear_2(self._dropout(torch.relu(self._linear_1(x))))
    
