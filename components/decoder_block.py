import torch.nn as nn
from components.feed_forward_block import FeedForwardBlock
from components.multi_head_attention_block import MultiHeadAttentionBlock
from components.residual_connection import ResidualConnection

class DecoderBlock(nn.Module):

    def __init__(self, 
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()
        self._self_attention_block = self_attention_block
        self._cross_attention_block = cross_attention_block
        self._feed_forward_block = feed_forward_block
        self._residual_connections = nn.ModuleList([ResidualConnection(dropout), 
                                                    ResidualConnection(dropout),
                                                    ResidualConnection(dropout),])
    
    # src_mask is used to hide the interaction of the padding word with other words
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self._residual_connections[0](x, 
                                          lambda x: self._self_attention_block(x, x, x, target_mask))
        x = self._residual_connections[1](x, 
                                          lambda x: self._cross_attention_block(x,
                                                                                encoder_output, 
                                                                                encoder_output, 
                                                                                src_mask))
        
        x = self._residual_connections[2](x, self._feed_forward_block)
        return x
    