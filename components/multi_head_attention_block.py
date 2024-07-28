import torch
import torch.nn as nn

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, no_of_heads: int, dropout: float) -> None:
        super().__init__()
        self._d_model = d_model
        self._no_of_heads = no_of_heads
        self._dropout = dropout
        if d_model % no_of_heads != 0:
            raise ValueError('Dimension is not divisible ny number of heads. Will not be able to assign them equally to heads.')
        
        self._d_k = d_model // no_of_heads
        self._w_q = nn.Linear(d_model, d_model)
        self._w_k = nn.Linear(d_model, d_model)
        self._w_v = nn.Linear(d_model, d_model)
        self._w_o = nn.Linear(d_model, d_model)

        self._dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        query = self._w_q(q) # (Batch, sequence length, d_model)
        key = self._w_q(k) # (Batch, sequence length, d_model)
        value = self._w_q(v) # (Batch, sequence length, d_model)

        query = self._restructure_matrix_for_heads(query)
        key = self._restructure_matrix_for_heads(key)
        value = self._restructure_matrix_for_heads(value)

    def _restructure_matrix_for_heads(self, matrix):
        # (Batch, Seq_len, d_model) -> (Batch, no. of heads, sequence length, d_model)
        matrix = matrix.view(matrix.shape[0], matrix.shape[1], self._no_of_heads, self._d_k) # Spliting only the last dimension, for the heads
        matrix = matrix.transpose(1, 2) # We want each head to see the whole sentence so just re-ordered seq_len with no of heads.
        return matrix