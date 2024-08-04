import torch.nn as nn

from components.decoder import Decoder
from components.encoder import Encoder
from components.decoder_block import DecoderBlock
from components.encoder_block import EncoderBlock
from components.feed_forward_block import FeedForwardBlock
from components.input_embedding import InputEmbeddings
from components.multi_head_attention_block import MultiHeadAttentionBlock
from components.positional_encoding import PositionalEncoding
from components.projection_layer import ProjectionLayer
from components.transformer import Transformer

class TranslationModel:
    
    @staticmethod
    def build_transformer(source_vocab_size: int,
                          target_vocab_size: int,
                          source_sequence_length: int,
                          target_sequence_length: int,
                          d_model: int = 512,
                          no_encoder_decoder_blocks: int = 6,
                          no_of_heads: int = 8,
                          dropout: float = 0.1,
                          no_of_hidden_layer_in_ff: int = 2048) -> Transformer:
        source_embedding = InputEmbeddings(d_model, source_vocab_size)
        target_embedding = InputEmbeddings(d_model, target_vocab_size)

        source_positional_encoding = PositionalEncoding(d_model, 
                                                        source_sequence_length,
                                                        dropout)
        target_positional_encoding = PositionalEncoding(d_model, 
                                                        target_sequence_length,
                                                        dropout)
        
        encoder_blocks = []
        for _ in range(no_encoder_decoder_blocks):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model,
                                                                   no_of_heads,
                                                                   dropout)
            feed_forward_block = FeedForwardBlock(d_model,
                                                  no_of_hidden_layer_in_ff,
                                                  dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block,
                                         feed_forward_block)
            encoder_blocks.append(encoder_block)

        decoder_blocks = []
        for _ in range(no_encoder_decoder_blocks):
            decoder_self_attention_block = MultiHeadAttentionBlock(d_model,
                                                                   no_of_heads,
                                                                   dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,
                                                                    no_of_heads,
                                                                    dropout)
            feed_forward_block = FeedForwardBlock(d_model,
                                                  no_of_hidden_layer_in_ff,
                                                  dropout)
            decoder_block = DecoderBlock(decoder_self_attention_block,
                                         decoder_cross_attention_block,
                                         feed_forward_block,
                                         dropout)
            decoder_blocks.append(decoder_block)

        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))
        
        projection_layer = ProjectionLayer(d_model, target_vocab_size)

        transformer = Transformer(encoder,
                                  decoder,
                                  source_embedding,
                                  target_embedding,
                                  source_positional_encoding,
                                  target_positional_encoding,
                                  projection_layer)
        
        # initialize parameters so model doesn't have to start from random. A lot of papers use Xavier
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        return transformer
    