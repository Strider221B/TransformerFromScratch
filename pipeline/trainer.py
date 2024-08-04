import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from constants import Constants as const
from pipeline.bilingual_dataset import BilingualDataset

class Trainer:

    @classmethod
    def _get_dataset(cls, config: dict):
        raw_dataset = load_dataset(const.DATASET_NAME, 
                                   f'{config[const.LANGUAGE_SOURCE]}-{config[const.LANGUAGE_TARGET]}',
                                   split=const.TRAIN)
        tokenizer_source = cls._get_or_build_tokenizer(config, 
                                                       raw_dataset, 
                                                       config[const.LANGUAGE_SOURCE])
        tokenizer_target = cls._get_or_build_tokenizer(config, 
                                                       raw_dataset,
                                                       config[const.LANGUAGE_TARGET])
        dataset_size = len(raw_dataset)
        train_dataset_size = int(0.9*dataset_size)
        val_dataset_size = dataset_size - train_dataset_size

        train_dataset_raw, val_dataset_raw = random_split(raw_dataset,
                                                          [train_dataset_size, val_dataset_size])
        
        train_dataset = BilingualDataset(train_dataset_raw, 
                                         tokenizer_source,
                                         tokenizer_target,
                                         config[const.LANGUAGE_SOURCE],
                                         config[const.LANGUAGE_TARGET],
                                         config[const.SEQUENCE_LENGTH])
        validation_dataset = BilingualDataset(val_dataset_raw, 
                                              tokenizer_source,
                                              tokenizer_target,
                                              config[const.LANGUAGE_SOURCE],
                                              config[const.LANGUAGE_TARGET],
                                              config[const.SEQUENCE_LENGTH])
        cls._log_max_token_lengths(raw_dataset, tokenizer_source, tokenizer_target, config) #15435

    @staticmethod
    def _log_max_token_lengths(dataset, tokenizer_source, tokenizer_target, config):
        max_len_source = 0
        max_len_target = 0
        for item in dataset:
            src_ids = tokenizer_source.encode(item[const.TRANSLATION][config[const.LANGUAGE_SOURCE]]).ids
            target_ids = tokenizer_target.encode(item[const.TRANSLATION][config[const.LANGUAGE_TARGET]]).ids
            max_len_source = max(max_len_source, len(src_ids))
            max_len_target = max(max_len_target, len(target_ids))
        print(f'Max length of source: {max_len_source}')
        print(f'Max length of target: {max_len_target}')

    @classmethod
    def _get_or_build_tokenizer(cls, config: dict, dataset, language):
        tokenizer_path = Path(config[const.TOKENIZER_FILE].format(language))
        if not Path.exists(tokenizer_path):
            tokenizer = Tokenizer(WordLevel(unk_token=const.TOKEN_UNKNOWN))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=[const.TOKEN_UNKNOWN, 
                                                       const.TOKEN_PADDING, 
                                                       const.TOKEN_START_OF_SENTENCE,
                                                       const.TOKEN_END_OF_SENTENCE], 
                                       min_frequency=2)
            tokenizer.train_from_iterator(cls._get_all_sentences(dataset, language),
                                          trainer=trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer        
    
    @staticmethod
    def _get_all_sentences(dataset, language):
        for item in dataset:
            yield item[const.TRANSLATION][language]