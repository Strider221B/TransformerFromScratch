import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from constants import Constants as const

class BilingualDataset(Dataset):

    def __init__(self, 
                 dataset, 
                 tokenizer_source: Tokenizer, 
                 tokenizer_target: Tokenizer, 
                 language_source: str, 
                 language_target: str, 
                 sequence_length: int) -> None:
        super().__init__()
        self._dataset = dataset
        self._tokenizer_source = tokenizer_source
        self._tokenizer_target = tokenizer_target
        self._language_source = language_source
        self._language_target = language_target
        self._sequence_length = sequence_length

        # could have used target tokenizer in the next line, either is fine.
        self._token_sos = torch.Tensor([tokenizer_source.token_to_id([const.TOKEN_START_OF_SENTENCE])], dtype=torch.int64)
        self._token_eos = torch.Tensor([tokenizer_source.token_to_id([const.TOKEN_END_OF_SENTENCE])], dtype=torch.int64)
        self._token_pad = torch.Tensor([tokenizer_source.token_to_id([const.TOKEN_PADDING])], dtype=torch.int64)

    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, index) -> dict:
        source_target_pair = self._dataset[index]
        source_text = source_target_pair[const.TRANSLATION][const.LANGUAGE_SOURCE]
        target_text = source_target_pair[const.TRANSLATION][const.LANGUAGE_TARGET]
        source_tokens = self._tokenizer_source.encode(source_text).ids
        target_tokens = self._tokenizer_target.encode(target_text).ids
        # source will need to have the EOS and SOS appended later so subtracting 2
        no_of_padding_tokens_source = self._get_number_of_padding_tokens_required_for(source_tokens, 2, source_text)
        # target will either have the EOS and SOS appended later so subtracting 1. We will just start with SOS during training
        # and labels will have EOS
        no_of_padding_tokens_target = self._get_number_of_padding_tokens_required_for(target_tokens, 1, target_text)
        encoder_input = torch.cat([
            self._token_sos,
            torch.tensor(source_tokens, dtype=torch.int64),
            self._token_eos,
            torch.tensor([self._token_pad] * no_of_padding_tokens_source, dtype=torch.int64)
        ])
        decoder_input = torch.cat([
            self._token_sos,
            torch.tensor(source_tokens, dtype=torch.int64),
            torch.tensor([self._token_pad] * no_of_padding_tokens_target, dtype=torch.int64)
        ])
        label = torch.cat([
            torch.tensor(source_tokens, dtype=torch.int64),
            self._token_eos,
            torch.tensor([self._token_pad] * no_of_padding_tokens_target, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self._sequence_length
        assert decoder_input.size(0) == self._sequence_length
        assert label.size(0) == self._sequence_length

        return {
            const.INPUT_ENCODER: encoder_input,
            const.INPUT_DECODER: decoder_input,
            const.LABEL: label,
            const.MASK_ENCODER: self._get_padding_mask(encoder_input), 
            const.MASK_DECODER: self._get_padding_mask(decoder_input) & self._get_causal_mask(decoder_input.size(0)),
            const.TEXT_SOURCE: source_text,
            const.TEXT_TARGET: target_text
        }
    
    def _get_padding_mask(self, tensor: torch.Tensor):
        return (tensor != self._token_pad).unsqueeze(0).unsqueeze(0).int() # (1, 1, sequence_len) => for sequence and batch dimension later
    
    def _get_causal_mask(self, input_size: int):
        # we don't want to use next words for prediction. We just want to use past words.
        mask = torch.triu(torch.ones(1, input_size, input_size), diagonal=1).type(torch.int)
        return mask == 0

    def _get_number_of_padding_tokens_required_for(self, 
                                                   tokens: list,
                                                   tokens_to_be_added_later: int,
                                                   original_str: str = '') -> int:
        no_of_tokens = self._sequence_length - len(tokens) - tokens_to_be_added_later
        if no_of_tokens < 0:
            raise ValueError(f'Sentence is too long. Original string - {original_str}')
        return no_of_tokens
    