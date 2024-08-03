import torch.nn as nn
from components.decoder import Decoder
from components.encoder import Encoder
from components.input_embedding import InputEmbeddings
from components.positional_encoding import PositionalEncoding
from components.projection_layer import ProjectionLayer

class Transformer(nn.Module):
 
    def __init__(self, 
                 encoder: Encoder,
                 decoder: Decoder,
                 source_embeddings: InputEmbeddings,
                 target_embeddings: InputEmbeddings,
                 source_positional_encoding: PositionalEncoding,
                 target_positional_encoding: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._source_embeddings = source_embeddings
        self._target_embeddings = target_embeddings
        self._source_positional_encoding = source_positional_encoding
        self._target_positional_encoding = target_positional_encoding
        self._projection_layer = projection_layer
    
    def encode(self, source, source_mask):
        source = self._source_embeddings(source)
        source = self._source_positional_encoding(source)
        return self._encoder(source, source_mask)
    
    def decode(self, encoder_output, source_mask, target, target_mask):
        target = self._target_embeddings(target)
        target = self._target_positional_encoding(target)
        return self._decoder(target, encoder_output, source_mask, target_mask)
    
    def project(self, x):
        return self._projection_layer(x)
    